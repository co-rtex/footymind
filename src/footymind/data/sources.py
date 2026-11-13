"""Registry of external datasets with pinned checksums, plus a verified downloader.

Raw provider data is never committed to the repository. ``footymind download``
fetches it into ``data/raw/<provider>/`` and verifies SHA-256 checksums so that
every downstream artefact can name the exact input bytes it was built from.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from footymind.paths import RAW_DIR

METRICA_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
METRICA_LICENSE_NOTE = (
    "Metrica Sports sample data (https://github.com/metrica-sports/sample-data). "
    "No explicit licence file. The provider asks users to 'be responsible with the "
    "use of this data' and to acknowledge the source for public use. Raw files are "
    "not redistributed by FootyMind."
)


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class MatchSource:
    match_id: str
    provider: str
    files: tuple[RemoteFile, ...]
    fmt: str  # "metrica_csv" | "metrica_epts" | "skillcorner"

    def local_dir(self, root: Path = RAW_DIR) -> Path:
        return root / self.provider


def _metrica_csv(game: int, hashes: tuple[str, str, str]) -> MatchSource:
    names = ("RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv")
    files = tuple(
        RemoteFile(
            url=f"{METRICA_BASE}/Sample_Game_{game}/Sample_Game_{game}_{n}",
            filename=f"Sample_Game_{game}_{n}",
            sha256=h,
        )
        for n, h in zip(names, hashes, strict=True)
    )
    return MatchSource(f"metrica-sample-{game}", "metrica", files, "metrica_csv")


SOURCES: dict[str, MatchSource] = {
    s.match_id: s
    for s in (
        _metrica_csv(
            1,
            (
                "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
                "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
                "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
            ),
        ),
        _metrica_csv(
            2,
            (
                "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
                "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
                "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
            ),
        ),
        MatchSource(
            "metrica-sample-3",
            "metrica",
            tuple(
                RemoteFile(
                    url=f"{METRICA_BASE}/Sample_Game_3/Sample_Game_3_{n}",
                    filename=f"Sample_Game_3_{n}",
                    sha256=h,
                )
                for n, h in (
                    (
                        "tracking.txt",
                        "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
                    ),
                    (
                        "events.json",
                        "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
                    ),
                    (
                        "metadata.xml",
                        "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
                    ),
                )
            ),
            "metrica_epts",
        ),
    )
}


SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
SKILLCORNER_LFS = (
    "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
)
SKILLCORNER_LICENSE_NOTE = (
    "SkillCorner open data (https://github.com/SkillCorner/opendata), MIT licence. "
    "The provider asks users to credit SkillCorner. Broadcast tracking: positions of players "
    "outside the camera view are provider extrapolations (is_detected = false)."
)
_SC_MANIFEST = Path(__file__).with_name("skillcorner_manifest.json")


def _skillcorner_sources() -> dict[str, MatchSource]:
    """SkillCorner matches whose file hashes are pinned in ``skillcorner_manifest.json``."""
    if not _SC_MANIFEST.exists():
        return {}
    import json

    out: dict[str, MatchSource] = {}
    for mid, files in json.loads(_SC_MANIFEST.read_text()).items():
        rf = []
        for name, digest in sorted(files.items()):
            base = SKILLCORNER_LFS if name.endswith(".jsonl") else SKILLCORNER_RAW
            rf.append(RemoteFile(f"{base}/{mid}/{name}", name, digest))
        out[f"skillcorner-{mid}"] = MatchSource(
            f"skillcorner-{mid}", "skillcorner", tuple(rf), "skillcorner"
        )
    return out


SOURCES.update(_skillcorner_sources())


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(source: MatchSource, root: Path = RAW_DIR) -> dict[str, str]:
    """Verify all files of a source exist with the pinned checksum."""
    out: dict[str, str] = {}
    for rf in source.files:
        path = source.local_dir(root) / rf.filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `footymind download {source.match_id}` first."
            )
        digest = sha256_file(path)
        if digest != rf.sha256:
            raise ChecksumError(f"{path}: sha256 {digest} != pinned {rf.sha256}")
        out[rf.filename] = digest
    return out


def download(source: MatchSource, root: Path = RAW_DIR, force: bool = False) -> dict[str, str]:
    target = source.local_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    for rf in source.files:
        path = target / rf.filename
        if path.exists() and not force and sha256_file(path) == rf.sha256:
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(rf.url, timeout=120) as resp, tmp.open("wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
        digest = sha256_file(tmp)
        if digest != rf.sha256:
            tmp.unlink()
            raise ChecksumError(f"{rf.url}: sha256 {digest} != pinned {rf.sha256}")
        tmp.replace(path)
    return verify(source, root)
