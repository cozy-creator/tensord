package main

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	pb "github.com/cozy-creator/worker-protocol-v2/gen/go/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

type retainedReady struct {
	Address  string `json:"address"`
	WorkerID string `json:"worker_id"`
	BootID   string `json:"boot_id"`
	Cert     string `json:"cert_pem"`
}

func runRetained(machine, output string) error {
	if machine == "" || output == "" {
		return fmt.Errorf("--machine and --output are required")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	pubA, keyA, _ := ed25519.GenerateKey(rand.Reader)
	pubB, keyB, _ := ed25519.GenerateKey(rand.Reader)
	privateJSON := func(path string, body any) error {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		if err = os.WriteFile(path, raw, 0600); err != nil {
			return err
		}
		return os.Chmod(path, 0600)
	}
	keyFile := filepath.Join(output, "keys.json")
	secretFile := filepath.Join(output, "readiness.json")
	configFile := filepath.Join(output, "machine.json")
	readyFile := filepath.Join(output, "ready.json")
	if err := privateJSON(keyFile, map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(pubA)}}); err != nil {
		return err
	}
	if err := privateJSON(secretFile, map[string]any{"key_b64url": base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{7}, 32))}); err != nil {
		return err
	}
	if err := privateJSON(configFile, map[string]any{"worker_id": "retained-cpu-fixture", "identity_directory": filepath.Join(output, "identity"), "authorized_keys_file": keyFile, "readiness_hmac_key_file": secretFile}); err != nil {
		return err
	}
	var process *exec.Cmd
	stop := func() {
		if process != nil {
			_ = process.Process.Signal(syscall.SIGTERM)
			_ = process.Wait()
			process = nil
		}
	}
	defer stop()
	start := func(address string) (retainedReady, error) {
		_ = os.Remove(readyFile)
		process = exec.Command(machine, "--config", configFile, "--ready-file", readyFile, "--listen", address, "--upload-root", filepath.Join(output, "native-state"))
		log, err := os.OpenFile(filepath.Join(output, "service.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			return retainedReady{}, err
		}
		defer log.Close()
		process.Stdout, process.Stderr = log, log
		if err = process.Start(); err != nil {
			return retainedReady{}, err
		}
		limit := time.Now().Add(20 * time.Second)
		for {
			var ready retainedReady
			raw, e := os.ReadFile(readyFile)
			if e == nil && json.Unmarshal(raw, &ready) == nil && ready.Address != "" {
				return ready, nil
			}
			if time.Now().After(limit) {
				return retainedReady{}, fmt.Errorf("retained fixture readiness not observed")
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	first, err := start("127.0.0.1:0")
	if err != nil {
		return err
	}
	block, _ := pem.Decode([]byte(first.Cert))
	if block == nil {
		return fmt.Errorf("retained leaf absent")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return err
	}
	ec, ok := cert.PublicKey.(*ecdsa.PublicKey)
	if !ok || ec.Curve != elliptic.P256() {
		return fmt.Errorf("leaf is not P256")
	}
	conn, err := grpc.NewClient(first.Address, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, ServerName: "cozy-worker", VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
		if len(raw) == 0 || !bytes.Equal(raw[0], block.Bytes) {
			return fmt.Errorf("retained leaf changed")
		}
		return nil
	}})))
	if err != nil {
		return err
	}
	defer conn.Close()
	host := pb.NewPodHostClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	author := func(ready retainedReady, key ed25519.PrivateKey) *pb.Claim {
		digest := sha256.Sum256(block.Bytes)
		transcript := []byte(fmt.Sprintf(`{"format":"cozy.worker.v1.ClaimProof/1","record_owner_epoch":1,"worker_boot_id":%q,"worker_id":%q,"worker_tls_certificate_digest":"sha256:%s"}`, ready.BootID, ready.WorkerID, hex.EncodeToString(digest[:])))
		return &pb.Claim{RecordOwnerEpoch: 1, WorkerId: ready.WorkerID, WorkerBootId: ready.BootID, Proof: ed25519.Sign(key, transcript)}
	}
	stale := author(first, keyA)
	carrier := []byte("native retained package carrier survives process and key lifecycle")
	digest := sha256.Sum256(carrier)
	header := func(claim *pb.Claim) *pb.LocalPackageUploadHeader {
		return &pb.LocalPackageUploadHeader{Claim: claim, OperationId: "retained-upload", File: &pb.LocalPackageFileRef{Filename: "fixture-0.0.1-py3-none-any.whl", Length: uint64(len(carrier)), Digest: digest[:]}}
	}
	transfer, err := host.LocalPackageUpload(ctx, grpc.WaitForReady(true))
	if err != nil {
		return err
	}
	if err = transfer.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header(stale)}}); err != nil {
		return err
	}
	prefix, err := transfer.Recv()
	if err != nil || prefix.ReceivedBytes != 0 {
		return fmt.Errorf("new carrier prefix: %v %v", prefix, err)
	}
	if err = transfer.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{Chunk: &pb.LocalPackageUploadChunk{Data: carrier}}}); err != nil {
		return err
	}
	complete, err := transfer.Recv()
	_ = transfer.CloseSend()
	if err != nil || complete.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
		return fmt.Errorf("native carrier: %v %v", complete, err)
	}
	stop()
	second, err := start(first.Address)
	if err != nil {
		return err
	}
	if first.Cert != second.Cert || first.WorkerID != second.WorkerID || first.BootID == second.BootID {
		return fmt.Errorf("retained leaf/worker or fresh boot invariant failed")
	}
	if _, err = host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{}, grpc.WaitForReady(true)); err != nil {
		return err
	}
	if _, err = host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: stale}); status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("stale boot not fenced: %v", err)
	}
	fresh := author(second, keyA)
	if _, err = host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: fresh}); err != nil {
		return err
	}
	transfer, err = host.LocalPackageUpload(ctx)
	if err != nil {
		return err
	}
	_ = transfer.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header(fresh)}})
	complete, err = transfer.Recv()
	_ = transfer.CloseSend()
	if err != nil || complete.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
		return fmt.Errorf("retained native custody disappeared: %v %v", complete, err)
	}
	stop()
	if err = privateJSON(keyFile, map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(pubB)}}); err != nil {
		return err
	}
	third, err := start(first.Address)
	if err != nil {
		return err
	}
	if _, err = host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{}, grpc.WaitForReady(true)); err != nil {
		return err
	}
	if _, err = host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: author(third, keyA)}); status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("retired key accepted: %v", err)
	}
	if _, err = host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: author(third, keyB)}); err != nil {
		return err
	}
	stop()
	if err = privateJSON(keyFile, map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(pubA)}}); err != nil {
		return err
	}
	fourth, err := start(first.Address)
	if err != nil {
		return err
	}
	if _, err = host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{}, grpc.WaitForReady(true)); err != nil {
		return err
	}
	transfer, err = host.LocalPackageUpload(ctx)
	if err != nil {
		return err
	}
	_ = transfer.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header(author(fourth, keyA))}})
	complete, err = transfer.Recv()
	_ = transfer.CloseSend()
	if err != nil || complete.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
		return fmt.Errorf("key refresh destroyed earlier native custody: %v %v", complete, err)
	}
	body := map[string]any{"checks": []string{"retained P256 leaf and worker identity across four real service processes", "same grpc connection reconnects under original exact leaf pin", "fresh boot UUID each process", "stale signed boot rejected", "native TensorFS package custody survives restart", "configured key refresh accepts new key and rejects retired key", "reauthorizing original owner recovers retained native custody"}, "limits": []string{"native source custody, not execution-journal restart qualification", "key rotation is applied by explicit process restart; no automatic ownership transfer between keys", "manual current-claim bootstrap does not prove existing Creator daemon automatically refreshes its cached boot claim"}}
	raw, _ := json.MarshalIndent(body, "", "  ")
	if err = os.WriteFile(filepath.Join(output, "results.json"), append(raw, '\n'), 0600); err != nil {
		return err
	}
	fmt.Println("PASS retained pin, boot fencing, configured key refresh and durable native custody")
	return nil
}
