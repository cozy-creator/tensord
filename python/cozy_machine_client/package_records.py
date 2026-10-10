"""Consumed package metadata and opaque SDK interface records."""
import msgspec


class Dependency(msgspec.Struct, frozen=True):
    name: str
    version: str


class Callee(msgspec.Struct, frozen=True):
    """Another package's App this environment holds as a dependency: a job of the root calls
    its invocables as child runs here."""
    distribution: str
    version: str
    application: str
    interface: msgspec.Raw
    source_digest: str = ""
    package: str = ""


class Generation(msgspec.Struct, frozen=True):
    identity: str
    package: str
    version: str
    application: str
    python: str
    dependencies: list[Dependency]
    interface: msgspec.Raw
    # Empty: the CPU runner's SDK adapter imports in this environment; else why it cannot.
    cpu_bridge: str = ""
    # The package's own installed files, hashed: what keys its memoized calls across machines.
    source_digest: str = ""
    callees: list[Callee] = []
    # Empty when this machine's own Runtime/TensorFS pair installed; else why the package's own runs.
    sdk_fallback: str = ""


class ApplicationEntry(msgspec.Struct, frozen=True):
    default: str


class EntryPoints(msgspec.Struct, frozen=True):
    application: ApplicationEntry = msgspec.field(name="cozy.application")


class Project(msgspec.Struct, frozen=True):
    name: str
    version: str
    entry_points: EntryPoints = msgspec.field(name="entry-points")


class Pyproject(msgspec.Struct, frozen=True):
    project: Project


class PackageMetadata(msgspec.Struct, frozen=True):
    name: str
    version: str
    application: str


class Describe(msgspec.Struct, tag="describe", tag_field="kind"):
    project: str
    environment_python: str | None = None


class DescribeInstalled(msgspec.Struct, tag="describe_installed", tag_field="kind"):
    distribution: str
    environment_python: str


class DescribeEnvironment(msgspec.Struct, tag="describe_environment", tag_field="kind"):
    root: str
    packages: dict[str, str] = {}


class DescribedEnvironment(msgspec.Struct, tag="described_environment", tag_field="kind"):
    source_digest: str
    callees: list[Callee]


class Described(msgspec.Struct, tag="described", tag_field="kind"):
    interface: msgspec.Raw


class DescribeFailed(msgspec.Struct, tag="describe_failed", tag_field="kind"):
    code: str
    detail: str


class InstallFailed(msgspec.Struct, tag="install_failed", tag_field="kind"):
    code: str
    detail: str


GENERATION_DECODER = msgspec.json.Decoder(Generation)
DESCRIPTION_DECODER = msgspec.json.Decoder(Described | DescribeFailed)
ENVIRONMENT_DECODER = msgspec.json.Decoder(DescribedEnvironment | DescribeFailed)
