"""The CLI names its data sources explicitly (no accidental 2 GB download). No data needed."""

from __future__ import annotations

import pytest

from footymind.cli import expand_groups, main, resolve_sources
from footymind.data import sources


def test_groups_expand_to_registered_ids_in_order() -> None:
    metrica = resolve_sources(["metrica"])
    assert metrica == ["metrica-sample-1", "metrica-sample-2", "metrica-sample-3"]
    sc = resolve_sources(["skillcorner"])
    assert len(sc) == 20 and all(m.startswith("skillcorner-") for m in sc)
    assert resolve_sources(["all"]) == list(sources.SOURCES)
    # explicit ids and groups combine without duplicates
    assert resolve_sources(["metrica-sample-2", "metrica"]) == [
        "metrica-sample-2",
        "metrica-sample-1",
        "metrica-sample-3",
    ]


@pytest.mark.parametrize("cmd", ["download", "ingest"])
def test_no_source_or_an_unknown_one_is_refused_before_any_work(cmd: str) -> None:
    with pytest.raises(SystemExit, match="name the sources"):
        main([cmd])
    with pytest.raises(SystemExit, match="unknown source 'nope'"):
        main([cmd, "nope"])


def test_documented_download_and_ingest_commands_name_valid_sources() -> None:
    """Every `footymind download|ingest` command in the setup docs names its sources, and they
    resolve (a bare command would stop; an unknown id would fail)."""
    import re

    from footymind.paths import REPO_ROOT

    docs = [
        "README.md",
        "CLAUDE.md",
        "web/README.md",
        "docs/CURRENT_STATE.md",
        "docs/release/VERIFICATION.md",
    ]
    cmd = re.compile(r"footymind (download|ingest)((?:[ \t]+[a-z0-9-]+)*)")
    found = 0
    for rel in docs:
        text = (REPO_ROOT / rel).read_text()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.splitlines():
                code = line.split("#", 1)[0]
                for _, args in cmd.findall(code):
                    names = args.split()
                    assert names, f"{rel}: `{line.strip()}` names no sources"
                    resolve_sources(names)
                    found += 1
    assert found >= 6


def test_ingest_without_raw_data_names_the_download_command(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from footymind import paths

    root = Path(str(tmp_path))
    monkeypatch.setattr(paths, "RAW_DIR", root / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", root / "processed")
    monkeypatch.setattr(sources, "RAW_DIR", root / "raw")
    with pytest.raises(SystemExit, match="run `footymind download metrica-sample-1` first"):
        main(["ingest", "metrica-sample-1"])


def test_expand_groups_keeps_unknown_names_for_the_entrypoint_to_report() -> None:
    # the Docker entrypoint expands groups, then reports and skips unregistered ids
    assert expand_groups(["metrica", "nope"]) == [
        "metrica-sample-1",
        "metrica-sample-2",
        "metrica-sample-3",
        "nope",
    ]
