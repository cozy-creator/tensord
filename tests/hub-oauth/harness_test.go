// Package harness serves a real AuthKit authorization server and a DPoP resource server in
// front of a test Hub, for tests/hub_oauth.rs and tests/machine_surface.rs (th-241). It signs
// run capabilities with an owner's enrolled device key, as the CLI does; the machine trades
// them through AuthKit's JWT-bearer grant, and AuthKit verifies every signature and proof. It
// plays the Hub: the grant decision (which may narrow), and on every token-bearing request
// the device key still live and the route among the token's operations.
package harness

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/open-rails/authkit"
	"github.com/open-rails/authkit/authtest"
	"github.com/open-rails/authkit/devicekey"
	"github.com/open-rails/authkit/iam"
	"github.com/open-rails/authkit/verify"
)

var (
	control  = flag.String("control", "", "where the harness writes its ready document")
	upstream = flag.String("upstream", "", "the test Hub the resource server forwards to")
)

const (
	client   = "tensord"
	resource = "https://hub.example.test"
	// narrowed is the model whose operations the Hub's decision always drops.
	narrowed = "acme/narrowed"
)

// seen is one token-bearing request the resource server verified.
type seen struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	Owner  string `json:"owner"`
	Actor  string `json:"actor"`
	Token  string `json:"token"` // a digest, never the token
}

// op is one capability operation (th-241's vocabulary).
type op struct {
	Type     string `json:"type"`
	Model    string `json:"model"`
	Manifest string `json:"manifest,omitempty"`
}

type hub struct {
	mu        sync.Mutex
	verified  []seen
	exchanges int
	lastToken string
}

// decide is the Hub's JWT-bearer decision: the capability's operations, less any on the
// narrowed model; none left is a refusal.
func (h *hub) decide(_ context.Context, r iam.OAuthGrantRequest) (iam.OAuthGrantDecision, error) {
	if r.Kind != iam.OAuthGrantJWTBearer || r.Capability == nil {
		return iam.OAuthGrantDecision{}, iam.ErrOAuthGrantRefused
	}
	var asked, kept []json.RawMessage
	if json.Unmarshal(r.AuthorizationDetails, &asked) != nil {
		return iam.OAuthGrantDecision{}, iam.ErrOAuthGrantRefused
	}
	for _, raw := range asked {
		var o op
		if json.Unmarshal(raw, &o) == nil && o.Model != narrowed {
			kept = append(kept, raw)
		}
	}
	if len(kept) == 0 {
		return iam.OAuthGrantDecision{}, iam.ErrOAuthGrantRefused
	}
	details, _ := json.Marshal(kept)
	h.mu.Lock()
	h.exchanges++
	h.mu.Unlock()
	return iam.OAuthGrantDecision{AuthorizationDetails: details}, nil
}

// permits answers whether ops name the operation a request to path is: a publication (or
// its checkpoint probe) to a model, or a read of one of its checkpoints.
func permits(ops []op, method, path string) bool {
	rest, ok := strings.CutPrefix(path, "/v1/models/")
	if !ok {
		return false
	}
	model, checkpoint, read := strings.Cut(rest, "/checkpoints/")
	if !read {
		model, _, ok = strings.Cut(rest, "/publications/")
		if !ok {
			return false
		}
	}
	return slices.ContainsFunc(ops, func(o op) bool {
		switch o.Type {
		case "tensorhub_model_publish":
			return o.Model == model
		case "tensorhub_model_read":
			return read && o.Model == model && method == http.MethodGet && strings.SplitN(checkpoint, "/", 2)[0] == o.Manifest
		}
		return false
	})
}

func TestHarness(t *testing.T) {
	if *control == "" {
		t.Skip("run by tests/hub_oauth.rs")
	}
	h := &hub{}
	issuerServer := httptest.NewUnstartedServer(nil)
	issuer := "http://" + issuerServer.Listener.Addr().String()
	auth, outbox := authtest.New(t, authtest.WithConfig(func(c *authkit.Config) {
		c.Token.Issuer = issuer
		c.Token.AllowPrivateNetworkJWKS = true
		c.DeviceKeys.Enabled = true
		c.AuthorizationServer = authkit.AuthorizationServerConfig{
			Resources: []authkit.ResourceServerConfig{{ID: resource}},
			Clients: []authkit.OAuthClientConfig{{
				ID: client, Name: "TensorD", Resources: []string{resource},
				GrantTypes:                []authkit.OAuthGrantType{authkit.GrantJWTBearer},
				AuthorizationDetailsTypes: []string{"tensorhub_model_read", "tensorhub_model_publish"},
			}},
		}
	}), authtest.WithDeps(func(d *authkit.Deps) { d.OAuthGrants = h.decide }))
	issuerServer.Config.Handler = auth.Handler()
	issuerServer.Start()
	t.Cleanup(issuerServer.Close)
	owner := authtest.NewUser(t, auth)
	device := authtest.EnrollDeviceKey(t, auth, outbox, owner)

	// The Hub's verifier: this deployment's own tokens, DPoP proofs spent once, nonces asked.
	nonceKey := make([]byte, 32)
	_, _ = rand.Read(nonceKey)
	v, err := auth.NewVerifier([]string{resource}, verify.WithDPoPNonce(nonceKey), verify.WithPublicURL(resource))
	if err != nil {
		t.Fatal(err)
	}
	target, err := url.Parse(*upstream)
	if err != nil {
		t.Fatal(err)
	}
	forward := httputil.NewSingleHostReverseProxy(target)
	refuse := func(w http.ResponseWriter, status int, code string) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, `{"error":{"code":"`+code+`","message":"refused"}}`)
	}
	authorized := verify.Required(v)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cl, _ := verify.ClaimsFromContext(r.Context())
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "DPoP ")
		sum := sha256.Sum256([]byte(token))
		h.mu.Lock()
		h.verified = append(h.verified, seen{r.Method, r.URL.Path, cl.Subject, cl.Invoker, base64.RawURLEncoding.EncodeToString(sum[:8])})
		h.lastToken = token
		h.mu.Unlock()
		// The Hub's per-request checks: the signing device key still live, then the route
		// among the token's operations.
		if auth.CheckSession(r.Context(), cl) != nil {
			refuse(w, http.StatusUnauthorized, "capability.revoked")
			return
		}
		var ops []op
		if json.Unmarshal(cl.AuthorizationDetails, &ops) != nil || !permits(ops, r.Method, r.URL.Path) {
			refuse(w, http.StatusForbidden, "capability.operation_forbidden")
			return
		}
		r.Header.Del("Authorization")
		r.Header.Del("DPoP")
		r.Header.Set("X-Verified-Owner", cl.Subject)
		forward.ServeHTTP(w, r)
	}))
	resourceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Header.Get("Authorization") != "" || r.Header.Get("DPoP") != "":
			authorized.ServeHTTP(w, r)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/packages/"):
			// Packages are public: read anonymously.
			r.Header.Set("X-Verified-Owner", "anonymous")
			forward.ServeHTTP(w, r)
		default:
			// A private object without a token answers as absent.
			refuse(w, http.StatusNotFound, "model.checkpoint_not_found")
		}
	}))
	t.Cleanup(resourceServer.Close)

	var mu sync.Mutex
	controlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch r.URL.Path {
		case "/capability": // the CLI signing a run's capability, offline
			seconds, _ := strconv.Atoi(r.FormValue("seconds"))
			signed, err := devicekey.SignCapability(device.Key, device.ID, devicekey.Capability{
				UserID: device.UserID, Audience: resource, WorkloadThumbprint: r.FormValue("jkt"),
				AuthorizationDetails: json.RawMessage(r.FormValue("ops")),
				ExpiresAt:            time.Now().Add(time.Duration(seconds) * time.Second),
				Claims:               map[string]any{"run": "run-" + random(6)},
			})
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			_, _ = io.WriteString(w, signed)
		case "/revoke-device-key": // the owner signing out of that device
			authtest.RevokeDeviceKey(t, auth, device)
			w.WriteHeader(http.StatusNoContent)
		case "/enroll-device-key": // the owner signing in on a new device
			device = authtest.EnrollDeviceKey(t, auth, outbox, owner)
			w.WriteHeader(http.StatusNoContent)
		case "/seen":
			h.mu.Lock()
			defer h.mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"verified": h.verified, "exchanges": h.exchanges,
				"owner": owner.ID, "last_token": h.lastToken})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(controlServer.Close)

	ready, _ := json.Marshal(map[string]string{
		"issuer": issuer, "resource": resource, "hub": resourceServer.URL, "control": controlServer.URL,
		"token_endpoint": issuer + "/oauth2/token",
	})
	if err := os.WriteFile(*control+".tmp", ready, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(*control+".tmp", *control); err != nil {
		t.Fatal(err)
	}
	// Serve until the test that started this process closes its stdin.
	_, _ = io.Copy(io.Discard, os.Stdin)
}

func random(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}
