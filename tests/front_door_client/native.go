package main

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
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
	"google.golang.org/protobuf/proto"
)

// Production work has no test deadlines or implicit cancellation. These budgets
// only bound this observer; cleanup releases this gate's own fixture process.
func runNative(machine, output, helper, wheel, source string, assets bool) error {
	if wheel == "" || source == "" {
		return fmt.Errorf("native wheel and frozen source required")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	writeJSON := func(name string, value any) error {
		b, err := json.Marshal(value)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(output, name), b, 0600)
	}
	if err = writeJSON("keys.json", map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(public)}}); err != nil {
		return err
	}
	if err = writeJSON("readiness.json", map[string]any{"key_b64url": base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{7}, 32))}); err != nil {
		return err
	}
	if err = writeJSON("machine.json", map[string]any{"worker_id": "native-cpu-gate", "identity_directory": filepath.Join(output, "identity"), "authorized_keys_file": filepath.Join(output, "keys.json"), "readiness_hmac_key_file": filepath.Join(output, "readiness.json")}); err != nil {
		return err
	}
	state := filepath.Join(output, "state")
	log, err := os.Create(filepath.Join(output, "machine.log"))
	if err != nil {
		return err
	}
	defer log.Close()
	command := exec.Command(machine, "serve", "--state", state, "--machine-config", filepath.Join(output, "machine.json"), "--listen", "127.0.0.1:0", "--installer-python", helper, "--client-wheel", wheel, "--package-python", "3.12", "--host-bytes", "0")
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
	limit := time.Now().Add(20 * time.Second)
	for {
		raw, e := os.ReadFile(filepath.Join(state, "api-ready.json"))
		if e == nil && json.Unmarshal(raw, &ready) == nil && ready.Address != "" {
			break
		}
		if time.Now().After(limit) {
			return fmt.Errorf("service readiness observer budget exhausted")
		}
		time.Sleep(20 * time.Millisecond)
	}
	block, _ := pem.Decode([]byte(ready.Cert))
	if block == nil {
		return fmt.Errorf("certificate absent")
	}
	leaf := block.Bytes
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, VerifyConnection: func(s tls.ConnectionState) error {
		if len(s.PeerCertificates) != 1 || !bytes.Equal(s.PeerCertificates[0].Raw, leaf) {
			return fmt.Errorf("pin differs")
		}
		_, e := x509.ParseCertificate(leaf)
		return e
	}}
	connection, err := grpc.NewClient(ready.Address, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	if err != nil {
		return err
	}
	defer connection.Close()
	host := pb.NewPodHostClient(connection)
	tlsDigest := sha256.Sum256(leaf)
	transcript := []byte(fmt.Sprintf(`{"format":"cozy.worker.v1.ClaimProof/1","record_owner_epoch":1,"worker_boot_id":%q,"worker_id":%q,"worker_tls_certificate_digest":"sha256:%s"}`, ready.BootID, ready.WorkerID, hex.EncodeToString(tlsDigest[:])))
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "mutable-label", WorkerId: ready.WorkerID, WorkerBootId: ready.BootID, WireMinor: 1, Proof: ed25519.Sign(private, transcript)}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	workspace, err := host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: claim})
	if err != nil {
		return err
	}
	if workspace.ExecutionWorkspaceId == "" || !workspace.RunOutputLog || !workspace.ReleaseRootOwner {
		return fmt.Errorf("native execution capabilities absent")
	}
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	err = filepath.WalkDir(source, func(path string, entry os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if entry.IsDir() {
			return nil
		}
		info, e := entry.Info()
		if e != nil {
			return e
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("fixture member is not regular")
		}
		name, e := filepath.Rel(source, path)
		if e != nil {
			return e
		}
		header, e := tar.FileInfoHeader(info, "")
		if e != nil {
			return e
		}
		header.Name = filepath.ToSlash(name)
		if e = writer.WriteHeader(header); e != nil {
			return e
		}
		data, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		_, e = writer.Write(data)
		return e
	})
	if err != nil {
		return err
	}
	if err = writer.Close(); err != nil {
		return err
	}
	packageMetadata := &pb.DevelopmentPackage{Package: "local/cozy-machine-cpu-classifier", Release: "0.1.0", InstallationId: "install-native-classifier"}
	if assets {
		packageMetadata.Package = "local/cozy-machine-cpu-assets"
	}
	header := &pb.LocalPackageUploadHeader{Claim: claim, OperationId: "native-source", File: &pb.LocalPackageFileRef{Filename: "source.tar", Length: uint64(archive.Len())}}
	upload, err := host.LocalPackageUpload(ctx)
	if err != nil {
		return err
	}
	if err = upload.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header}}); err != nil {
		return err
	}
	if _, err = upload.Recv(); err != nil {
		return err
	}
	if err = upload.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{Chunk: &pb.LocalPackageUploadChunk{Data: archive.Bytes()}}}); err != nil {
		return err
	}
	if _, err = upload.Recv(); err != nil {
		return err
	}
	_ = upload.CloseSend()
	selection := &pb.DesiredLocalPackageSet{OperationId: header.OperationId, Package: packageMetadata, Files: []*pb.LocalPackageFileRef{header.File}, PythonRequires: ">=3.12,<3.15", PythonVersion: "3.12", SourceArchive: header.File.Filename}
	preparation, err := host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: claim, LocalPackageSet: selection})
	if err != nil {
		return err
	}
	prepared, err := preparation.Recv()
	if err != nil {
		return err
	}
	if prepared.InstalledPackage == nil {
		return fmt.Errorf("no installed package: %v", prepared)
	}
	if err = writeJSON("prepared.json", prepared); err != nil {
		return err
	}
	request := &pb.MachineExecutionSubmit{Claim: claim, SubmissionId: "native-submission", Offer: &pb.AttemptOffer{RequestId: "native-request"}, ExpectedExecutionWorkspaceId: workspace.ExecutionWorkspaceId, PayloadCanonicalBytes: []byte(`{"iterations":2,"samples":[[5.1,3.5,1.4,0.2],[6,2.7,5.1,1.6],[6.7,3.1,4.7,1.5]],"seed":19}`), ReleaseRoot: &pb.ReleaseRoot{Package: packageMetadata.Package, InstallationId: packageMetadata.InstallationId, Entrypoint: "classify"}}
	receipt, err := host.SubmitMachineExecution(ctx, request)
	if err != nil {
		return err
	}
	claim.RecordOwnerId = "different-label"
	replay, err := host.SubmitMachineExecution(ctx, request)
	if err != nil || !proto.Equal(receipt, replay) {
		return fmt.Errorf("replay receipt changed: %v", err)
	}
	query := &pb.MachineExecutionQuery{Claim: claim, RequestId: receipt.RequestId, ExpectedExecutionWorkspaceId: workspace.ExecutionWorkspaceId}
	var terminal *pb.AttemptOutcome
	var product *pb.RunProduct
	var products []*pb.RunProduct
	var after uint64
	for terminal == nil {
		page, e := host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: after, Wait: true})
		if e != nil {
			return e
		}
		for _, event := range page.Events {
			if event.Product != nil {
				product = event.Product
				if len(event.BodyCanonicalBytes) == 0 {
					return fmt.Errorf("product canonical document absent")
				}
				products = append(products, product)
			}
			if event.Outcome != nil {
				terminal = event.Outcome
			}
		}
		after = page.NextAfter
	}
	digest := sha256.Sum256(terminal.OutcomeCanonicalBytes)
	if !bytes.Equal(digest[:], terminal.OutcomeDigest) {
		return fmt.Errorf("outcome digest mismatch")
	}
	if err = os.WriteFile(filepath.Join(output, "outcome.json"), terminal.OutcomeCanonicalBytes, 0600); err != nil {
		return err
	}
	var body struct {
		Status int `json:"status"`
		Result struct {
			Inline string `json:"inline_result"`
		} `json:"result"`
	}
	if err = json.Unmarshal(terminal.OutcomeCanonicalBytes, &body); err != nil {
		return err
	}
	if body.Status != 1 {
		return fmt.Errorf("inference not successful: %s", terminal.OutcomeCanonicalBytes)
	}
	value, err := base64.StdEncoding.DecodeString(body.Result.Inline)
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(output, "result.json"), value, 0600); err != nil {
		return err
	}
	if assets {
		if len(products) != 3 || products[0].Output != "report" || products[0].Op != pb.RunProductOp_RUN_PRODUCT_OP_SET || products[1].Output != "reports" || products[1].Op != pb.RunProductOp_RUN_PRODUCT_OP_APPEND || products[1].Index != 0 || products[2].Output != "reports" || products[2].Index != 1 {
			return fmt.Errorf("declared scalar/list fold differs: %v", products)
		}
		var result struct {
			Opaque map[string]string `json:"opaque"`
		}
		if err = json.Unmarshal(value, &result); err != nil {
			return err
		}
		if result.Opaque["asset_ref"] != "authored user metadata" || result.Opaque["digest"] != "unchanged" {
			return fmt.Errorf("opaque user data was treated as an asset")
		}
	}
	if product == nil || product.Source == nil || product.Content == nil {
		return fmt.Errorf("native product absent")
	}
	read := func(object *pb.Ref, offset uint64) ([]byte, error) {
		stream, e := host.ReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: claim, Source: product.Source, Object: object, Offset: offset})
		if e != nil {
			return nil, e
		}
		var data []byte
		cursor := offset
		for {
			chunk, e := stream.Recv()
			if e == io.EOF {
				break
			}
			if e != nil {
				return nil, e
			}
			if chunk.Offset != cursor {
				return nil, fmt.Errorf("range cursor mismatch")
			}
			data = append(data, chunk.Data...)
			cursor += uint64(len(chunk.Data))
		}
		return data, nil
	}
	report, err := read(product.Content, 0)
	if err != nil {
		return err
	}
	reportDigest := sha256.Sum256(report)
	if !bytes.Equal(reportDigest[:], product.Content.Digest) {
		return fmt.Errorf("product digest mismatch")
	}
	if err = os.WriteFile(filepath.Join(output, "report.json"), report, 0600); err != nil {
		return err
	}
	suffix, err := read(product.Content, 10)
	if err != nil || !bytes.Equal(suffix, report[10:]) {
		return fmt.Errorf("suffix differs: %v", err)
	}
	manifest, err := read(product.Source.Source.Manifest, 0)
	if err != nil {
		return err
	}
	manifestDigest := sha256.Sum256(manifest)
	if !bytes.Equal(manifestDigest[:], product.Source.Source.Manifest.Digest) {
		return fmt.Errorf("manifest differs")
	}
	otherPub, otherKey, _ := ed25519.GenerateKey(rand.Reader)
	_ = otherPub
	unauthorized := proto.Clone(claim).(*pb.Claim)
	unauthorized.Proof = ed25519.Sign(otherKey, transcript)
	if _, err = host.GetMachineExecution(ctx, &pb.MachineExecutionQuery{Claim: unauthorized, RequestId: receipt.RequestId, ExpectedExecutionWorkspaceId: workspace.ExecutionWorkspaceId}); status.Code(err) != codes.Unauthenticated {
		return fmt.Errorf("wrong key read allowed: %v", err)
	}
	page, err := host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query})
	if err != nil || len(page.Events) < 2 {
		return fmt.Errorf("terminal replay unavailable: %v", err)
	}
	ack := &pb.AttemptOutcomeAck{WorkerBootId: terminal.WorkerBootId, RequestId: terminal.RequestId, AttemptOrdinal: terminal.AttemptOrdinal, InvocationSpecDigest: terminal.InvocationSpecDigest, OutcomeId: terminal.OutcomeId, OutcomeDigest: terminal.OutcomeDigest}
	wrongBoot := proto.Clone(ack).(*pb.AttemptOutcomeAck)
	wrongBoot.WorkerBootId = "different-outcome-boot"
	if _, err = host.AcknowledgeMachineExecutionCollection(ctx, &pb.MachineExecutionCollectionAck{Execution: query, Outcome: wrongBoot}); status.Code(err) != codes.InvalidArgument {
		return fmt.Errorf("wrong outcome boot acknowledged: %v", err)
	}
	// Deployed Creator omits this additive field; its authenticated Claim binds the live boot.
	ack.WorkerBootId = ""
	badAck := proto.Clone(ack).(*pb.AttemptOutcomeAck)
	badAck.OutcomeId = "different-outcome"
	if _, err = host.AcknowledgeMachineExecutionCollection(ctx, &pb.MachineExecutionCollectionAck{Execution: query, Outcome: badAck}); status.Code(err) != codes.InvalidArgument {
		return fmt.Errorf("wrong acknowledgement accepted: %v", err)
	}
	for index := 0; index < 2; index++ {
		state, e := host.AcknowledgeMachineExecutionCollection(ctx, &pb.MachineExecutionCollectionAck{Execution: query, Outcome: ack})
		if e != nil || !state.Collected {
			return fmt.Errorf("collection acknowledgement not durable: %v", e)
		}
	}
	if _, err = read(product.Content, 0); err != nil {
		return fmt.Errorf("collection acknowledgement destroyed native hold: %v", err)
	}
	return writeJSON("evidence.json", map[string]any{"receipt": receipt, "report_sha256": hex.EncodeToString(reportDigest[:]), "checks": []string{"retained pinned TLS", "older peer wire minor1", "frozen source preparation", "SDK real classifier inference", "same-key label change remains same actor", "identical durable replay receipt", "outcome content identity", "native manifest/full/suffix readback", "wrong-key refusal", "terminal log replay"}, "gpu_qualified": false, "ordinary_cli_qualified": false})
}
