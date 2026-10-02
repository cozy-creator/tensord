// Imports actual current Creator helpers read-only through an isolated -modfile.
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
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

func main() {
	machine := flag.String("machine", "", "Rust inspection fixture")
	output := flag.String("output", "", "owned test artifact directory")
	flag.Parse()
	if err := run(*machine, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(machine, output string) error {
	if machine == "" || output == "" {
		return fmt.Errorf("--machine and --output are required")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	key := make([]byte, 32)
	if _, err = rand.Read(key); err != nil {
		return err
	}
	keyFile, readyFile := filepath.Join(output, "receipt.key"), filepath.Join(output, "ready.json")
	if err = os.WriteFile(keyFile, key, 0600); err != nil {
		return err
	}
	if err = os.Remove(readyFile); err != nil && !os.IsNotExist(err) {
		return err
	}
	log, err := os.Create(filepath.Join(output, "service.log"))
	if err != nil {
		return err
	}
	defer log.Close()
	command := exec.Command(machine, "--worker-id", "isolated-capture-owner", "--owner-key", base64.RawURLEncoding.EncodeToString(public), "--receipt-key-file", keyFile, "--ready-file", readyFile, "--upload-root", filepath.Join(output, "machine-state"))
	command.Stdout, command.Stderr = log, log
	if err = command.Start(); err != nil {
		return err
	}
	defer func() { _ = command.Process.Signal(syscall.SIGTERM); _ = command.Wait() }()
	var ready struct {
		Address  string `json:"address"`
		WorkerID string `json:"worker_id"`
		BootID   string `json:"boot_id"`
		Cert     string `json:"cert_pem"`
	}
	deadline := time.Now().Add(20 * time.Second)
	for {
		raw, e := os.ReadFile(readyFile)
		if e == nil && json.Unmarshal(raw, &ready) == nil && ready.Address != "" {
			break
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("private fixture readiness absent")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cert, _ := pem.Decode([]byte(ready.Cert))
	if cert == nil {
		return fmt.Errorf("no pinned leaf")
	}
	conn, err := grpc.NewClient(ready.Address, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, ServerName: "cozy.worker", InsecureSkipVerify: true, VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
		if len(raw) == 0 || !bytes.Equal(raw[0], cert.Bytes) {
			return fmt.Errorf("wrong pinned leaf")
		}
		return nil
	}})))
	if err != nil {
		return err
	}
	defer conn.Close()
	digest := sha256.Sum256(cert.Bytes)
	transcript, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: ready.WorkerID, WorkerBootId: ready.BootID, WorkerTlsCertificateDigest: digest[:]})
	if err != nil {
		return err
	}
	claim := &pb.Claim{RecordOwnerEpoch: 1, WorkerId: ready.WorkerID, WorkerBootId: ready.BootID, RecordOwnerId: "untrusted-display-selector", WireMinor: 1, Proof: ed25519.Sign(private, transcript)}
	host := pb.NewPodHostClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	info, err := host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil || info.MinimumWireMinor != 0 {
		return fmt.Errorf("baseline protocol floor: %v %v", info, err)
	}
	source := filepath.Join(output, "authored-source")
	if err = os.MkdirAll(source, 0700); err != nil {
		return err
	}
	for name, data := range map[string][]byte{"pyproject.toml": []byte("[project]\nname='fixture'\nversion='0.0.1'\nrequires-python='>=3.11'\n"), "uv.lock": []byte("version = 1\nrevision = 3\n"), "fixture.py": []byte("raise RuntimeError('source description must never import me')\n"), "source-data.txt": bytes.Repeat([]byte("owned source bytes\n"), 200000)} {
		if err = os.WriteFile(filepath.Join(source, name), data, 0600); err != nil {
			return err
		}
	}
	archive := filepath.Join(output, "source.tar")
	if err = os.Remove(archive); err != nil && !os.IsNotExist(err) {
		return err
	}
	length, problem := packagepublish.WriteSourceArchive(source, archive)
	if problem != nil {
		return problem
	}
	installation := localpackage.Installation{ID: "install-owned-fixture", Package: "local/fixture", Release: "0.0.1", SourceArchive: "source.tar", PythonRequires: ">=3.11", PythonVersion: "3.11", Files: []localpackage.File{{Filename: "source.tar", Kind: "source", Length: length, Path: archive}}}
	selected, problem := orchestrator.LocalPackageSelection("capture-owned-fixture", installation)
	if problem != nil {
		return problem
	}
	header := &pb.LocalPackageUploadHeader{Claim: claim, OperationId: selected.OperationId, File: selected.Files[0]}
	// Interrupt only the observer after one durable chunk; the real uploader resumes it.
	transfer, cancelTransfer := context.WithCancel(ctx)
	stream, err := host.LocalPackageUpload(transfer)
	if err != nil {
		return err
	}
	if err = stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header}}); err != nil {
		return err
	}
	initial, err := stream.Recv()
	if err != nil || initial.ReceivedBytes != 0 {
		return fmt.Errorf("initial prefix: %v %v", initial, err)
	}
	carrier, err := os.ReadFile(archive)
	if err != nil {
		return err
	}
	if err = stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{Chunk: &pb.LocalPackageUploadChunk{Data: carrier[:1<<20]}}}); err != nil {
		return err
	}
	prefix, err := stream.Recv()
	if err != nil || prefix.ReceivedBytes != 1<<20 || prefix.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING {
		return fmt.Errorf("durable prefix: %v %v", prefix, err)
	}
	cancelTransfer()
	_ = stream.CloseSend()
	// Reconnection may race the departing stream's operation lock; no timeout kills it.
	var observed []uint64
	for {
		problem = localpackage.UploadFile(ctx, host, header, archive, func(s *pb.LocalPackageFileStatus) { observed = append(observed, s.ReceivedBytes) })
		if problem == nil {
			break
		}
		if ctx.Err() != nil {
			return problem
		}
		if len(observed) > 0 {
			return problem
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(observed) == 0 || observed[0] != 1<<20 || observed[len(observed)-1] != uint64(length) {
		return fmt.Errorf("real Creator resume offsets: %v", observed)
	}
	// Final replay needs neither the laptop file nor its original unsigned owner label.
	claim.RecordOwnerId = "another-unsigned-label"
	if problem = localpackage.UploadFile(ctx, host, header, filepath.Join(output, "does-not-exist"), nil); problem != nil {
		return problem
	}
	prepare, err := host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: claim, LocalPackageSet: selected})
	if err == nil {
		_, err = prepare.Recv()
	}
	if status.Code(err) != codes.Unimplemented {
		return fmt.Errorf("inspection backend must not pretend installed/invokable: %v", err)
	}
	denied, err := host.LocalPackageUpload(ctx)
	if err != nil {
		return err
	}
	bad := *header
	bad.Claim = nil
	_ = denied.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: &bad}})
	_, err = denied.Recv()
	_ = denied.CloseSend()
	if status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("unauthenticated capture accepted: %v", err)
	}
	evidence := map[string]any{"checks": []string{"actual Creator WriteSourceArchive", "actual Creator LocalPackageSelection", "actual Creator canonical ClaimProof", "durable 1MiB prefix after observer interruption", "actual Creator UploadFile pipelined resume", "verified replay without original laptop carrier", "unsigned owner label does not change actor namespace", "typed RootSet imported before honest unsupported installation", "unsigned capture rejected before resource access", "ProtocolInfo baseline minimum0"}, "acknowledged_offsets": observed, "source_archive_bytes": length, "limits": []string{"source transfer and static archive validation only; no installed package, accepted invocation or inference"}}
	raw, _ := json.MarshalIndent(evidence, "", "  ")
	if err = os.WriteFile(filepath.Join(output, "results.json"), append(raw, '\n'), 0600); err != nil {
		return err
	}
	fmt.Printf("PASS actual Creator capture/upload/resume: %d bytes, %v\n", length, observed)
	return nil
}
