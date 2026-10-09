// Package harness serves a real AuthKit authorization server and a DPoP resource server in
// front of a test Hub, for tests/hub_oauth.rs: the machine redeems and presents real grants
// and AuthKit verifies every proof it makes. It plays the CLI too: a device-key sign-in that
// approves each authorization request (th-238's flow).
package harness

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/open-rails/authkit"
	"github.com/open-rails/authkit/authtest"
	"github.com/open-rails/authkit/iam"
	"github.com/open-rails/authkit/verify"
)

var (
	control  = flag.String("control", "", "where the harness writes its ready document")
	upstream = flag.String("upstream", "", "the test Hub the resource server forwards to")
)

const (
	client      = "cozy-machine"
	redirect    = "http://127.0.0.1/cozy-machine/callback"
	resource    = "https://hub.example.test"
	execution   = "tensorhub_execution"
	publication = "tensorhub_machine_publication"
	accessTTL   = 4 * time.Second
)

// seen is one request the resource server verified.
type seen struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	Type   string `json:"type"`
	Token  string `json:"token"` // a digest, never the token
}

func TestHarness(t *testing.T) {
	if *control == "" {
		t.Skip("run by tests/hub_oauth.rs")
	}
	issuerServer := httptest.NewUnstartedServer(nil)
	issuer := "http://" + issuerServer.Listener.Addr().String()
	auth, outbox := authtest.New(t, authtest.WithConfig(func(c *authkit.Config) {
		c.Token.Issuer = issuer
		c.Token.AllowPrivateNetworkJWKS = true
		c.DeviceKeys.Enabled = true
		c.AuthorizationServer = authkit.AuthorizationServerConfig{
			Resources: []authkit.ResourceServerConfig{{ID: resource}},
			Clients: []authkit.OAuthClientConfig{{
				ID: client, Name: "Cozy machine", RedirectURIs: []string{redirect}, Resources: []string{resource},
				GrantTypes:                []authkit.OAuthGrantType{authkit.GrantAuthorizationCode, authkit.GrantRefreshToken},
				AuthorizationDetailsTypes: []string{execution, publication}, Offline: true, KeyBound: true,
				AccessTokenTTL: accessTTL, RefreshTokenTTL: 7 * 24 * time.Hour,
			}},
		}
	}), authtest.WithDeps(func(d *authkit.Deps) { d.OAuthGrants = (&authtest.GrantAuthorizer{}).Authorize }))
	issuerServer.Config.Handler = auth.Handler()
	issuerServer.Start()
	t.Cleanup(issuerServer.Close)

	// The CLI's sign-in: a device key, as `cozy auth login` holds one.
	owner := authtest.NewUser(t, auth)
	signIn := authtest.EnrollDeviceKey(t, auth, outbox, owner)
	signedIn := signIn.AccessToken

	nonceKey := make([]byte, 32)
	_, _ = rand.Read(nonceKey)
	v := verify.NewVerifier(verify.WithHTTPClient(http.DefaultClient), verify.WithDPoP(memoryReplay()),
		verify.WithDPoPNonce(nonceKey), verify.WithPublicURL(resource))
	if err := v.AddIssuer(issuer, []string{resource}, verify.IssuerOptions{JWKSURI: issuer + iam.JWKSPath}); err != nil {
		t.Fatal(err)
	}
	target, err := url.Parse(*upstream)
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var verified []seen
	forward := httputil.NewSingleHostReverseProxy(target)
	resourceServer := httptest.NewServer(verify.Required(v)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cl, _ := verify.ClaimsFromContext(r.Context())
		var details []struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(cl.AuthorizationDetails, &details)
		kind := ""
		if len(details) == 1 {
			kind = details[0].Type
		}
		// The Hub's rule: publication routes take a publication grant, every other read the
		// owner's execution grant.
		want := execution
		if strings.Contains(r.URL.Path, "/publications/") || strings.Contains(r.URL.Path, "/checkpoints/") {
			want = publication
		}
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "DPoP ")
		sum := sha256.Sum256([]byte(token))
		mu.Lock()
		verified = append(verified, seen{r.Method, r.URL.Path, kind, base64.RawURLEncoding.EncodeToString(sum[:8])})
		mu.Unlock()
		if kind != want {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, `{"error":{"code":"auth.forbidden","message":"wrong grant"}}`)
			return
		}
		r.Header.Del("Authorization")
		r.Header.Del("DPoP")
		r.Header.Set("X-Verified-Type", kind)
		forward.ServeHTTP(w, r)
	})))
	t.Cleanup(resourceServer.Close)

	controlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/authorize":
			answer, err := authorize(issuer, auth.APIBase(), signedIn, r.FormValue("kind"), r.FormValue("jkt"))
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			_ = json.NewEncoder(w).Encode(answer)
		case "/logout":
			req, _ := http.NewRequest(http.MethodDelete, issuer+auth.APIBase()+"/logout", nil)
			req.Header.Set("Authorization", "Bearer "+signedIn)
			res, err := http.DefaultClient.Do(req)
			if err != nil || res.StatusCode != http.StatusNoContent {
				http.Error(w, fmt.Sprintf("logout: %v %v", err, res), http.StatusBadGateway)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		case "/seen":
			mu.Lock()
			defer mu.Unlock()
			_ = json.NewEncoder(w).Encode(verified)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(controlServer.Close)

	ready, _ := json.Marshal(map[string]string{
		"issuer": issuer, "resource": resource, "hub": resourceServer.URL, "control": controlServer.URL,
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

// authorize runs the CLI's half of th-238's flow for a machine key: the authorization request
// (PKCE, resource, dpop_jkt, authorization_details), read without following its redirect,
// then the approval with the CLI's sign-in. It answers the HubAuthorization the CLI hands over.
func authorize(issuer, api, signedIn, kind, jkt string) (map[string]string, error) {
	verifier := random(32)
	challenge := sha256.Sum256([]byte(verifier))
	state := random(12)
	q := url.Values{
		"response_type": {"code"}, "client_id": {client}, "redirect_uri": {redirect}, "state": {state},
		"code_challenge": {base64.RawURLEncoding.EncodeToString(challenge[:])}, "code_challenge_method": {"S256"},
		"resource": {resource}, "dpop_jkt": {jkt},
	}
	switch kind {
	case "execution":
		q.Set("authorization_details", `[{"type":"`+execution+`"}]`)
	case "publication":
		q.Set("authorization_details", `[{"type":"`+publication+`","machine_id":"","repositories":[{"org":"acme","name":"tiny"}],"permissions":["checkpoint"]}]`)
		q.Set("scope", "offline_access")
	default:
		return nil, fmt.Errorf("no grant kind %q", kind)
	}
	noRedirect := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := noRedirect.Get(issuer + iam.OAuthAuthorizePath + "?" + q.Encode())
	if err != nil {
		return nil, err
	}
	res.Body.Close()
	location, _ := url.Parse(res.Header.Get("Location"))
	if res.StatusCode != http.StatusSeeOther || location == nil || location.Query().Get("authorization") == "" {
		return nil, fmt.Errorf("authorize: %d %s", res.StatusCode, res.Header.Get("Location"))
	}
	req, _ := http.NewRequest(http.MethodPost, issuer+api+"/oauth2/authorizations/"+url.PathEscape(location.Query().Get("authorization"))+"/approve", nil)
	req.Header.Set("Authorization", "Bearer "+signedIn)
	res, err = http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	var out struct {
		RedirectTo string `json:"redirect_to"`
	}
	body, _ := io.ReadAll(res.Body)
	if res.StatusCode != http.StatusOK || json.Unmarshal(body, &out) != nil {
		return nil, fmt.Errorf("approve: %d %s", res.StatusCode, body)
	}
	callback, err := url.Parse(out.RedirectTo)
	if err != nil || callback.Query().Get("state") != state || callback.Query().Get("iss") != issuer || callback.Query().Get("code") == "" {
		return nil, fmt.Errorf("approve: redirect %q does not answer the request", out.RedirectTo)
	}
	return map[string]string{
		"issuer": issuer, "code": callback.Query().Get("code"), "code_verifier": verifier,
		"redirect_uri": redirect, "resource": resource,
	}, nil
}

func random(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func memoryReplay() func(context.Context, string, time.Duration) (bool, error) {
	var mu sync.Mutex
	claimed := map[string]bool{}
	return func(_ context.Context, key string, _ time.Duration) (bool, error) {
		mu.Lock()
		defer mu.Unlock()
		if claimed[key] {
			return false, nil
		}
		claimed[key] = true
		return true, nil
	}
}
