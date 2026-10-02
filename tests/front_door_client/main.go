// Real deployed protobuf/gRPC client against an isolated Rust service.
package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	pb "github.com/cozy-creator/worker-protocol-v2/gen/go/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protowire"
)

type ready struct{ Address, WorkerID, BootID, CertPEM string }
type evidence struct {
	Checks []string `json:"checks"`
	Limits []string `json:"limits"`
}

func main() {
	machine := flag.String("machine", "", "path to front-door fixture binary")
	output := flag.String("output", "", "owned evidence directory")
	retained := flag.Bool("retained-restart", false, "qualify retained pin and key refresh across owned fixture restarts")
	helper := flag.String("native-helper", "", "trusted installer interpreter for integrated service gate")
	wheel := flag.String("native-wheel", "", "built machine client wheel")
	source := flag.String("native-source", "", "frozen real classifier source")
	flag.Parse()
	gate := run
	if *helper != "" {
		gate = func(machine, output string) error { return runNative(machine, output, *helper, *wheel, *source) }
	}
	if *retained {
		gate = runRetained
	}
	if err := gate(*machine, *output); err != nil {
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
	receiptKey := make([]byte, 32)
	if _, err = rand.Read(receiptKey); err != nil {
		return err
	}
	keyFile, readyFile := filepath.Join(output, "receipt.key"), filepath.Join(output, "ready.json")
	if err = os.WriteFile(keyFile, receiptKey, 0600); err != nil {
		return err
	}
	// A previous ready record must never select a different process.
	if err = os.Remove(readyFile); err != nil && !os.IsNotExist(err) {
		return err
	}
	log, err := os.Create(filepath.Join(output, "service.log"))
	if err != nil {
		return err
	}
	defer log.Close()
	command := exec.Command(machine, "--worker-id", "isolated-cpu-front-door", "--owner-key", base64.RawURLEncoding.EncodeToString(public), "--receipt-key-file", keyFile, "--ready-file", readyFile)
	command.Stdout, command.Stderr = log, log
	if err = command.Start(); err != nil {
		return err
	}
	// Explicit fixture teardown after the authored checks, never a rental/owner daemon.
	defer func() { _ = command.Process.Signal(syscall.SIGTERM); _ = command.Wait() }()
	var r struct {
		Address  string `json:"address"`
		WorkerID string `json:"worker_id"`
		BootID   string `json:"boot_id"`
		CertPEM  string `json:"cert_pem"`
	}
	deadline := time.Now().Add(20 * time.Second)
	for {
		data, readErr := os.ReadFile(readyFile)
		if readErr == nil && json.Unmarshal(data, &r) == nil && r.Address != "" {
			break
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("fixture readiness was not observed; see service.log")
		}
		time.Sleep(10 * time.Millisecond)
	}
	block, _ := pem.Decode([]byte(r.CertPEM))
	if block == nil {
		return fmt.Errorf("no pinned leaf")
	}
	leaf := block.Bytes
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, ServerName: "cozy.worker", InsecureSkipVerify: true,
		VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
			if len(raw) == 0 || !bytes.Equal(raw[0], leaf) {
				return fmt.Errorf("machine leaf differs from pin")
			}
			return nil
		}}
	connection, err := grpc.NewClient(r.Address, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	if err != nil {
		return err
	}
	defer connection.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	host, control := pb.NewPodHostClient(connection), pb.NewWorkerControlClient(connection)
	e := evidence{Limits: []string{"inspection backend only; no accepted execution, package preparation, ordinary cozy run, Hub provisioning, GPU or browser qualification", "service process is an owned fixture, not the user daemon"}}
	check := func(name string) { e.Checks = append(e.Checks, name); fmt.Println("PASS", name) }
	protocol, err := host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil || protocol.WireMinor != 72 || protocol.MinimumWireMinor != 0 {
		return fmt.Errorf("ProtocolInfo: %v %v", protocol, err)
	}
	check("real TLS/gRPC ProtocolInfo")
	wrongPin := tlsConfig.Clone()
	wrongPin.VerifyPeerCertificate = func(raw [][]byte, _ [][]*x509.Certificate) error {
		return fmt.Errorf("fixture intentionally pinned a different leaf")
	}
	wrongConnection, err := grpc.NewClient(r.Address, grpc.WithTransportCredentials(credentials.NewTLS(wrongPin)))
	if err != nil {
		return err
	}
	_, pinErr := pb.NewPodHostClient(wrongConnection).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	_ = wrongConnection.Close()
	if status.Code(pinErr) != codes.Unavailable {
		return fmt.Errorf("wrong leaf pin was accepted: %v", pinErr)
	}
	check("a different pinned leaf is rejected at TLS")
	digest := sha256.Sum256(leaf)
	transcript := []byte(fmt.Sprintf(`{"format":"cozy.worker.v1.ClaimProof/1","record_owner_epoch":1,"worker_boot_id":%q,"worker_id":%q,"worker_tls_certificate_digest":"sha256:%s"}`, r.BootID, r.WorkerID, hex.EncodeToString(digest[:])))
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: r.WorkerID, WorkerBootId: r.BootID, WireMinor: 72, Proof: ed25519.Sign(private, transcript)}
	stream, err := control.Control(ctx)
	if err != nil {
		return err
	}
	if err = stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return err
	}
	frame, err := stream.Recv()
	if err != nil || frame.GetClaimAck() == nil || !frame.GetClaimAck().Accepted || frame.GetClaimAck().WorkerId != r.WorkerID || frame.GetClaimAck().WorkerBootId != r.BootID {
		return fmt.Errorf("Control ClaimAck: %v %v", frame, err)
	}
	_ = stream.CloseSend()
	check("Creator-style signed Control ClaimAck")
	for _, version := range []uint32{0, 1, 64, 72, ^uint32(0)} {
		claim.WireMinor = version
		described, err := host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: claim})
		if err != nil || described.WorkerId != r.WorkerID || described.Host.Phase != "ready" {
			return fmt.Errorf("version %d DescribeMachine: %v %v", version, described, err)
		}
		workspace, err := host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: claim})
		if err != nil || workspace.AcceleratorBackend != "none" || workspace.RunOutputLog || workspace.ReleaseRootOwner {
			return fmt.Errorf("workspace incorrectly advertises capabilities: %v %v", workspace, err)
		}
	}
	check("old/current/future peers authenticated without global version refusal")
	claim.ProtoReflect().SetUnknown(protowire.AppendVarint(protowire.AppendTag(nil, 999, protowire.VarintType), 17))
	if _, err := host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: claim}); err != nil {
		return err
	}
	check("additive unknown protobuf fields")
	bad := *claim
	bad.Proof = append([]byte(nil), claim.Proof...)
	bad.Proof[0] ^= 1
	tests := []struct {
		name string
		c    *pb.Claim
	}{{"missing Claim", nil}, {"bad signature", &bad}}
	stale := *claim
	stale.WorkerBootId = "previous-boot"
	tests = append(tests, struct {
		name string
		c    *pb.Claim
	}{"stale boot", &stale})
	wrong := *claim
	wrong.WorkerId = "other-machine"
	tests = append(tests, struct {
		name string
		c    *pb.Claim
	}{"wrong machine", &wrong})
	for _, t := range tests {
		if _, err := host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: t.c}); status.Code(err) != codes.Unauthenticated {
			return fmt.Errorf("%s not rejected: %v", t.name, err)
		}
		if _, err := host.SubmitMachineExecution(ctx, &pb.MachineExecutionSubmit{Claim: t.c}); status.Code(err) != codes.Unauthenticated {
			return fmt.Errorf("submit %s not rejected before backend: %v", t.name, err)
		}
	}
	check("missing, bad-signature, stale-boot and wrong-machine claims reject before execution")
	if _, err := host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{}); status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("nested missing Claim: %v", err)
	}
	check("nested operation claims are required")
	if _, err := host.SubmitMachineExecution(ctx, &pb.MachineExecutionSubmit{Claim: claim}); status.Code(err) != codes.Unimplemented {
		return fmt.Errorf("unsupported execution should refuse operation: %v", err)
	}
	if _, err := host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: claim}); err != nil {
		return err
	}
	check("unsupported operation leaves baseline reads usable")
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConfig}, Timeout: 10 * time.Second}
	response, err := client.Get("https://" + r.Address + "/v1/bootstrap/receipt")
	if err != nil {
		return err
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 65537))
	_ = response.Body.Close()
	if err != nil || response.StatusCode != 200 {
		return fmt.Errorf("receipt: status %d %v", response.StatusCode, err)
	}
	var envelope struct {
		Payload []byte `json:"payload"`
		HMAC    string `json:"hmac_sha256"`
	}
	if err = json.Unmarshal(data, &envelope); err != nil {
		return err
	}
	mac := hmac.New(sha256.New, receiptKey)
	mac.Write([]byte("cozy.pod-readiness/1\x00"))
	mac.Write(envelope.Payload)
	actual, err := hex.DecodeString(envelope.HMAC)
	if err != nil || !hmac.Equal(actual, mac.Sum(nil)) {
		return fmt.Errorf("readiness HMAC failed")
	}
	var payload struct {
		Boot        string `json:"pod_boot_id"`
		Certificate []byte `json:"tls_certificate_der_base64"`
		Port        int    `json:"worker_internal_port"`
	}
	if err = json.Unmarshal(envelope.Payload, &payload); err != nil || payload.Boot != r.BootID || !bytes.Equal(payload.Certificate, leaf) {
		return fmt.Errorf("receipt leaf/boot mismatch")
	}
	check("existing HMAC readiness envelope names exact TLS leaf and boot")
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+r.Address+"/v1/health", nil)
	request.Header.Set("Authorization", "Bearer foreign")
	response, err = client.Do(request)
	if err != nil {
		return err
	}
	_ = response.Body.Close()
	if response.StatusCode != 401 {
		return fmt.Errorf("health accepted foreign bearer")
	}
	check("health rejects foreign credentials")
	maps, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", command.Process.Pid))
	if err != nil {
		return err
	}
	for _, name := range []string{"libcuda", "libnvidia-ml", "libcudart"} {
		if strings.Contains(string(maps), name) {
			return fmt.Errorf("CPU server loaded %s", name)
		}
	}
	check("CPU server mappings exclude CUDA/NVML")
	fds, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", command.Process.Pid))
	if err != nil {
		return err
	}
	for _, fd := range fds {
		target, readErr := os.Readlink(fmt.Sprintf("/proc/%d/fd/%s", command.Process.Pid, fd.Name()))
		if readErr == nil && (strings.HasPrefix(target, "/dev/nvidia") || strings.HasPrefix(target, "/dev/dri")) {
			return fmt.Errorf("CPU server opened GPU device %s", target)
		}
	}
	check("CPU server descriptors exclude GPU devices")
	out, _ := json.MarshalIndent(e, "", "  ")
	return os.WriteFile(filepath.Join(output, "results.json"), append(out, '\n'), 0600)
}
