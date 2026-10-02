// Real Creator snapshot/manifest helpers against an owned service with ready native callbacks.
package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

func main() {
	machine := flag.String("machine", "", "native Rust service binary")
	output := flag.String("output", "", "owned evidence directory")
	flag.Parse()
	if err := run(*machine, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(machine, output string) error {
	if machine == "" || output == "" {
		return fmt.Errorf("--machine and --output required")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	public, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	otherPublic, otherKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	write := func(name string, body any) error {
		raw, e := json.Marshal(body)
		if e != nil {
			return e
		}
		return os.WriteFile(filepath.Join(output, name), raw, 0600)
	}
	if err = write("keys.json", map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(public), base64.RawURLEncoding.EncodeToString(otherPublic)}}); err != nil {
		return err
	}
	if err = write("secret.json", map[string]any{"key_b64url": base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{9}, 32))}); err != nil {
		return err
	}
	if err = write("config.json", map[string]any{"worker_id": "native-consumer-cpu", "identity_directory": filepath.Join(output, "identity"), "authorized_keys_file": filepath.Join(output, "keys.json"), "readiness_hmac_key_file": filepath.Join(output, "secret.json")}); err != nil {
		return err
	}
	state := filepath.Join(output, "state")
	log, err := os.Create(filepath.Join(output, "service.log"))
	if err != nil {
		return err
	}
	defer log.Close()
	cmd := exec.Command(machine, "serve", "--state", state, "--machine-config", filepath.Join(output, "config.json"), "--listen", "127.0.0.1:0", "--host-bytes", "0")
	cmd.Stdout, cmd.Stderr = log, log
	if err = cmd.Start(); err != nil {
		return err
	}
	defer func() { _ = cmd.Process.Signal(syscall.SIGTERM); _ = cmd.Wait() }()
	var ready struct {
		Address string `json:"address"`
		Worker  string `json:"worker_id"`
		Boot    string `json:"boot_id"`
		Cert    string `json:"cert_pem"`
	}
	limit := time.Now().Add(20 * time.Second)
	for {
		raw, e := os.ReadFile(filepath.Join(state, "api-ready.json"))
		if e == nil && json.Unmarshal(raw, &ready) == nil && ready.Address != "" {
			break
		}
		if time.Now().After(limit) {
			return fmt.Errorf("owned service readiness absent")
		}
		time.Sleep(10 * time.Millisecond)
	}
	leaf, _ := pem.Decode([]byte(ready.Cert))
	if leaf == nil {
		return fmt.Errorf("pinned leaf missing")
	}
	conn, err := grpc.NewClient(ready.Address, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, ServerName: "cozy-worker", VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
		if len(raw) == 0 || !bytes.Equal(raw[0], leaf.Bytes) {
			return fmt.Errorf("leaf changed")
		}
		return nil
	}})))
	if err != nil {
		return err
	}
	defer conn.Close()
	digest := sha256.Sum256(leaf.Bytes)
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: ready.Worker, WorkerBootId: ready.Boot, WorkerTlsCertificateDigest: digest[:]})
	if err != nil {
		return err
	}
	claim := &pb.Claim{RecordOwnerEpoch: 1, WorkerId: ready.Worker, WorkerBootId: ready.Boot, WireMinor: 1, Proof: ed25519.Sign(key, proof)}
	otherClaim := &pb.Claim{RecordOwnerEpoch: 1, WorkerId: ready.Worker, WorkerBootId: ready.Boot, Proof: ed25519.Sign(otherKey, proof)}
	host := pb.NewPodHostClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	described, err := host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: claim}, grpc.WaitForReady(true))
	if err != nil {
		return err
	}
	if described.Runtime == nil || described.RuntimeAbsent != "" {
		return fmt.Errorf("live orchestration incorrectly absent: %v", described)
	}
	if described.Runtime.PythonVersion != "" {
		return fmt.Errorf("one fabricated Python SDK version for Rust orchestration")
	}
	packages, err := host.ListPackages(ctx, &pb.PackageListQuery{Claim: claim})
	if err != nil {
		return err
	}
	if len(packages.Packages) != 0 {
		return fmt.Errorf("fresh owner unexpectedly has package observations")
	}
	tree := filepath.Join(output, "authored-input")
	if err = os.MkdirAll(tree, 0700); err != nil {
		return err
	}
	data := bytes.Repeat([]byte("real Creator native tree bytes\n"), 80000)
	if err = os.WriteFile(filepath.Join(tree, "a.bin"), data, 0600); err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(tree, "duplicate.bin"), data, 0600); err != nil {
		return err
	}
	snapshot, problem := inputasset.CaptureTree(filepath.Join(output, "input-manifest"), "dataset", tree, inputasset.MaxRootInputBytes)
	if problem != nil {
		return problem
	}
	members, problem := resultfiles.ParseTreeManifest(snapshot.Snapshot.Body, snapshot.Snapshot.ContentBytes)
	if problem != nil {
		return problem
	}
	rawManifest, err := canonical.Raw(snapshot.Snapshot.Manifest.Digest)
	if err != nil {
		return err
	}
	header := &pb.InputTreeImportHeader{Claim: claim, RequestId: "native-consumer-request", InputId: "dataset", Manifest: &pb.Ref{Digest: rawManifest, Length: uint64(snapshot.Snapshot.Manifest.Length)}, ManifestCanonicalBytes: snapshot.Snapshot.Body, ContentBytes: uint64(snapshot.Snapshot.ContentBytes)}
	upload := func(abort, withBytes bool) (*pb.NativeByteRetentionResult, error) {
		stream, e := host.ImportInputTree(ctx)
		if e != nil {
			return nil, e
		}
		if e = stream.Send(&pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Header{Header: header}}); e != nil {
			return nil, e
		}
		if withBytes {
			seen := map[string]bool{}
			for _, member := range members {
				if seen[member.Digest] {
					continue
				}
				seen[member.Digest] = true
				raw, _ := canonical.Raw(member.Digest)
				file, e := os.Open(filepath.Join(snapshot.Snapshot.Path+".files", strings.TrimPrefix(member.Digest, "sha256:")))
				if e != nil {
					return nil, e
				}
				buf := make([]byte, 1<<20)
				var offset uint64
				for {
					n, e := file.Read(buf)
					if n > 0 {
						if e = stream.Send(&pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Blob{Blob: &pb.InputTreeImportBlob{Object: &pb.Ref{Digest: raw, Length: uint64(member.Length)}, Offset: offset, Data: append([]byte(nil), buf[:n]...)}}}); e != nil {
							file.Close()
							return nil, e
						}
						offset += uint64(n)
					}
					if e == io.EOF {
						break
					}
					if e != nil {
						file.Close()
						return nil, e
					}
				}
				file.Close()
			}
		}
		if e = stream.Send(&pb.InputTreeImportFrame{Body: &pb.InputTreeImportFrame_Commit{Commit: &pb.InputTreeImportCommit{Abort: abort}}}); e != nil {
			return nil, e
		}
		return stream.CloseAndRecv()
	}
	held, err := upload(false, true)
	if err != nil {
		return err
	}
	if held.Source == nil || held.Released || !bytes.Equal(held.Source.Manifest.Digest, rawManifest) || held.Source.ContentBytes != header.ContentBytes {
		return fmt.Errorf("native input changed real Creator manifest")
	}
	replay, err := upload(false, false)
	if err != nil {
		return err
	}
	if replay.RetentionId != held.RetentionId || replay.Released {
		return fmt.Errorf("input commit replay differs")
	}
	_, err = host.RetainByteTree(ctx, &pb.NativeByteRetentionCall{Claim: otherClaim, Request: &pb.NativeByteRetentionRequest{Source: held.Source, RetentionId: "sha256:" + strings.Repeat("c", 64)}})
	if status.Code(err) != codes.NotFound && status.Code(err) != codes.PermissionDenied {
		return fmt.Errorf("authorized different actor retained another actor's input: %v", err)
	}
	consumerID := "sha256:" + strings.Repeat("b", 64)
	request := &pb.NativeByteRetentionCall{Claim: claim, Request: &pb.NativeByteRetentionRequest{Source: held.Source, RetentionId: consumerID}}
	retained, err := host.RetainByteTree(ctx, request)
	if err != nil {
		return err
	}
	if retained.RetentionId != consumerID || retained.Released {
		return fmt.Errorf("consumer retention differs")
	}
	released, err := upload(true, false)
	if err != nil {
		return err
	}
	if !released.Released {
		return fmt.Errorf("explicit input abort did not release intake")
	}
	late, err := upload(false, false)
	if err != nil {
		return err
	}
	if !late.Released {
		return fmt.Errorf("late input commit reopened durable abort tombstone")
	}
	objectDigest, _ := canonical.Raw(members[0].Digest)
	reader, err := host.ReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: claim, Source: &pb.NativeByteRetentionRequest{Source: retained.Source, RetentionId: retained.RetentionId}, Object: &pb.Ref{Digest: objectDigest, Length: uint64(members[0].Length)}})
	if err != nil {
		return err
	}
	var read []byte
	for {
		chunk, e := reader.Recv()
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		if chunk.Offset != uint64(len(read)) || len(chunk.Data) > 1<<20 {
			return fmt.Errorf("noncontiguous/unbounded native read")
		}
		read = append(read, chunk.Data...)
	}
	if !bytes.Equal(read, data) {
		return fmt.Errorf("consumer hold did not survive producer release")
	}
	dropped, err := host.ReleaseByteTree(ctx, request)
	if err != nil {
		return err
	}
	if !dropped.Released {
		return fmt.Errorf("consumer release not durable")
	}
	_, err = host.RetainByteTree(ctx, &pb.NativeByteRetentionCall{})
	if status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("unsigned custody operation reached owner: %v", err)
	}
	return write("evidence.json", map[string]any{"checks": []string{"actual Creator CaptureTree and ParseTreeManifest", "authenticated1MiB input stream and duplicate-object dedup", "native commit replay without bytes", "independent consumer retention survives intake release", "full contiguous native readback", "explicit consumer release", "truthful Rust runtime/empty per-package inventory", "unsigned custody rejected", "authorized cross-actor retention refused", "late commit preserves abort tombstone"}, "ordinary_cli_qualified": false, "inference_qualified": false})
}
