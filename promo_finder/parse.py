"""Parse Kuku-style timestamped screenplays into numbered beats.

Every block in a screenplay ("**CAMERA:**", "**ACTION:**", "**SOUND FX:**",
"**JULIANA (Cold, Commanding):**") becomes one Beat with an id like E01-023.
The AI later refers to beats only by id; all timestamps come from here.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")
HEADER_RE = re.compile(r"^\*\*\s*([^*]+?)\s*:\s*\*\*\s*$")
STARTEND_RE = re.compile(
    r"Start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*End:\s*\(?\s*([\d:.]+)\s*\)?", re.I
)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)", re.I)
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TEXT", "TITLE", "VFX"}


def to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return float("nan")


def fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    id: str
    episode: int
    idx: int
    scene: int
    kind: str          # CAMERA | ACTION | DIALOGUE | SOUND FX | OTHER
    speaker: str
    emotion: str
    text: str
    start: float
    end: float

    def render(self, max_camera_chars: int = 140) -> str:
        t = f"[{self.id} {fmt_ts(self.start)}-{fmt_ts(self.end)}]"
        if self.kind == "DIALOGUE":
            emo = f" ({self.emotion})" if self.emotion else ""
            return f"{t} {self.speaker}{emo}: {self.text}"
        text = self.text
        if self.kind == "CAMERA" and len(text) > max_camera_chars:
            text = text[: max_camera_chars - 1] + "…"
        return f"{t} {self.kind}: {text}"


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def beat_index(self) -> dict[str, Beat]:
        return {b.id: b for b in self.beats}


def _episode_number(name: str, text: str) -> int | None:
    m = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"episode[-_ ]0*(\d+)", name, re.I)
    return int(m.group(1)) if m else None


def parse_episode_text(text: str, name: str = "", fallback_number: int = 0) -> Episode:
    num = _episode_number(name, text) or fallback_number
    logline = (re.search(r"EPISODE LOGLINE:\s*(.+)", text) or [None, ""])[1].strip()
    continuity = (re.search(r"CONTINUITY NOTES:\s*(.+)", text) or [None, ""])[1].strip()
    ep = Episode(number=num, title=f"Episode {num}", logline=logline, continuity=continuity)

    lines = text.splitlines()
    scene = 0
    i = 0
    idx = 0
    while i < len(lines):
        line = lines[i].strip()
        sm = SCENE_RE.match(line)
        if sm:
            scene = int(sm.group(1))
            i += 1
            continue
        hm = HEADER_RE.match(line)
        if not hm:
            i += 1
            continue
        header = hm.group(1).strip()
        # find Start/End line (normally the next line)
        j = i + 1
        start = end = None
        while j < len(lines) and j <= i + 3:
            se = STARTEND_RE.search(lines[j])
            if se:
                start, end = to_seconds(se.group(1)), to_seconds(se.group(2))
                break
            j += 1
        if start is None:
            i += 1
            continue
        # body = following non-empty lines until blank line / next header
        k = j + 1
        body = []
        while k < len(lines):
            l2 = lines[k].strip()
            if not l2:
                if body:
                    break
                k += 1
                continue
            if HEADER_RE.match(l2) or l2.startswith("--- SCRIPT") or SCENE_RE.match(l2):
                break
            body.append(l2)
            k += 1
        text_body = " ".join(body).strip()
        # header -> kind / speaker / emotion
        em = re.match(r"^(.*?)\s*\((.*)\)\s*$", header)
        base, emotion = (em.group(1).strip(), em.group(2).strip()) if em else (header, "")
        base_u = base.upper()
        if base_u in NON_DIALOGUE:
            kind = "SOUND FX" if base_u in ("SOUND FX", "SFX") else base_u
            if kind not in ("CAMERA", "ACTION", "SOUND FX"):
                kind = "OTHER"
            speaker = ""
        else:
            kind, speaker = "DIALOGUE", base_u
            text_body = text_body.strip()
        idx += 1
        ep.beats.append(
            Beat(
                id=f"E{num:02d}-{idx:03d}", episode=num, idx=idx, scene=scene,
                kind=kind, speaker=speaker, emotion=emotion, text=text_body,
                start=float(start), end=float(end if end and end >= start else start),
            )
        )
        i = k
    return ep


def load_show(path: str | Path | bytes, name: str = "") -> list[Episode]:
    """Load a show from a folder of .txt files, a .zip (path or bytes) or one .txt."""
    texts: list[tuple[str, str]] = []
    if isinstance(path, (bytes, bytearray)):
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(path)) as z:
                for n in z.namelist():
                    if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                        texts.append((n, z.read(n).decode("utf-8", "replace")))
        else:
            texts.append((name, path.decode("utf-8", "replace")))
    else:
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                texts.append((f.name, f.read_text("utf-8", "replace")))
        elif p.suffix.lower() == ".zip":
            return load_show(p.read_bytes(), p.name)
        else:
            texts.append((p.name, p.read_text("utf-8", "replace")))
    eps = [parse_episode_text(t, n, i + 1) for i, (n, t) in enumerate(sorted(texts))]
    eps = [e for e in eps if e.beats]
    eps.sort(key=lambda e: e.number)
    return eps


def beats_table(episodes: list[Episode]):
    import pandas as pd
    return pd.DataFrame([asdict(b) for e in episodes for b in e.beats])
