"""Callee identities come from explicit capture or standard installed provenance."""
import importlib.metadata
import json

import pytest

from cozy_machine_client.distribution import package_name


@pytest.mark.parametrize("url,expected", [
    ("https://hub/v1/index/first/files/abc/cpu_memo.whl", "first/cpu-memo"),
    ("https://hub/v1/index/second/files/def/cpu_memo.whl", "second/cpu-memo"),
    ("https://pypi.org/packages/cpu_memo.whl", "local/cpu-memo"),
    ("file:///wheels/cpu_memo.whl", "local/cpu-memo"),
    ("https://hub:444/v1/index/first/files/abc/cpu_memo.whl", "local/cpu-memo"),
    ("https://hub:0/v1/index/first/files/abc/cpu_memo.whl", "local/cpu-memo"),
    ("https://user@hub/v1/index/first/files/abc/cpu_memo.whl", "local/cpu-memo"),
    ("https://@hub/v1/index/first/files/abc/cpu_memo.whl", "local/cpu-memo"),
])
def test_installed_wheel_provenance_scopes_its_package_without_import(tmp_path, url, expected):
    record = tmp_path / "cpu_memo-1.0.dist-info"
    record.mkdir()
    (record / "METADATA").write_text("Name: CPU_Memo\nVersion: 1.0\n")
    (record / "direct_url.json").write_text(json.dumps({"url": url, "archive_info": {}}))
    installed = importlib.metadata.Distribution.at(record)
    assert package_name(installed, {}, "https://hub:443") == expected
    assert package_name(installed, {}) == "local/cpu-memo"
    assert package_name(installed, {}, "https://other-hub") == "local/cpu-memo"
    assert package_name(installed, {}, "http://hub") == "local/cpu-memo"
    assert package_name(installed, {"cpu-memo": "local/cpu-memo"}) == "local/cpu-memo"
    assert package_name(installed, {"cpu-memo": "first/cpu-memo"}) == "first/cpu-memo"
    with pytest.raises(ValueError, match="different package"):
        package_name(installed, {"cpu-memo": "first/another-package"})
