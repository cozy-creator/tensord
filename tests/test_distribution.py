"""A callee's identity is the one the machine read from its release's lock, else local."""
import importlib.metadata
import json

import pytest

from cozy_machine_client.distribution import package_name


@pytest.mark.parametrize("url", [
    "https://hub/v1/index/first/files/abc/cpu_memo.whl",
    "https://hub/v1/index/first/cpu-memo/1.0/cpu_memo.whl",
    "file:///wheels/cpu_memo.whl",
])
def test_a_callee_is_named_by_the_lock_never_by_its_wheel_url(tmp_path, url):
    record = tmp_path / "cpu_memo-1.0.dist-info"
    record.mkdir()
    (record / "METADATA").write_text("Name: CPU_Memo\nVersion: 1.0\n")
    (record / "direct_url.json").write_text(json.dumps({"url": url, "archive_info": {}}))
    installed = importlib.metadata.Distribution.at(record)
    assert package_name(installed, {}) == "local/cpu-memo"
    assert package_name(installed, {"cpu-memo": "first/cpu-memo"}) == "first/cpu-memo"
    with pytest.raises(ValueError, match="different package"):
        package_name(installed, {"cpu-memo": "first/another-package"})
