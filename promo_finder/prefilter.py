"""Cheap, rule-based tension scoring (no AI, ~0 cost).

Purpose: find the 'hot spots' in each episode so the AI only reads those,
and give every candidate a transparent tension score that is independent
of the AI. Word lists were built by surveying the emotion tags and sound
cues across the input pack; tune them in one place below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import Beat, Episode, fmt_ts

# --- lexicons -------------------------------------------------------------
HIGH_EMOTION = {
    "shouting", "screaming", "furious", "enraged", "outraged", "hysterical",
    "venomous", "threatening", "menacing", "sinister", "chilling", "cruel",
    "terrified", "panicked", "frantic", "shocked", "stunned", "horrified",
    "sobbing", "weeping", "desperate", "pleading", "heartbroken", "broken",
    "gasping", "choked", "accusatory", "accusing", "spiteful", "malicious",
    "vindictive", "gloating", "taunting", "mocking", "scornful", "contemptuous",
    "booming", "piercing", "alarmed", "incredulous", "defiant", "fierce",
    "cold", "smug", "triumphant", "seething", "livid", "icy", "deadly",
}
SHOCK_WORDS = re.compile(
    r"\b(slap\w*|strik\w*|struck|punch\w*|shov\w*|push\w*|grab\w*|seiz\w*|choke\w*|"
    r"stab\w*|knife|dagger|blade|sword|blood\w*|bleed\w*|wound\w*|poison\w*|arsenic|"
    r"collaps\w*|faint\w*|kneel\w*|knees|tear\w* (?:off|open|apart)|rip\w*|shatter\w*|"
    r"smash\w*|crash\w*|explo\w*|fire|burn\w*|flames|whip\w*|chain\w*|shackle\w*|cell|"
    r"arrest\w*|guards?|execut\w*|kill\w*|murder\w*|dead|death|die[ds]?|corpse|"
    r"pregnan\w*|heir|secret\w*|reveal\w*|truth|lie[sd]?|liar|betray\w*|traitor|"
    r"scandal\w*|disown\w*|divorce\w*|annul\w*|banish\w*|exile\w*|abandon\w*|"
    r"gasp\w*|scream\w*|shriek\w*|horrif\w*|shock\w*|stun\w*|freez\w*|frozen|"
    r"duke|duchess|crown|king|queen|throne|seal|ring|kiss\w*)\b",
    re.I,
)
DIALOGUE_PUNCH = re.compile(
    r"(how dare|you will pay|who are you|get out|on your knees|you're nothing|"
    r"i know what you|i saw|you lied|it was you|you killed|arrest|seize (?:her|him)|"
    r"you will regret|mark my words|you're mine|she's mine|he is the duke|your grace|"
    r"never again|kneel|enough!|liar|traitor|guards!)",
    re.I,
)
CLIFF_MARKERS = re.compile(
    r"(cut to black|smash cut|cliffhanger|unresolved|ominous|freez\w*|frozen|"
    r"hangs? in the air|abrupt\w*|stops? dead|holds? (?:her|his) breath|"
    r"eyes widen\w*|jaw drops|drained of (?:all )?colou?r|horrified|stunned|"
    r"silhouette (?:appears|looms)|door (?:bursts|slams|flies) open|a voice|"
    r"tension-filled|swells? to)",
    re.I,
)
RESOLVE_MARKERS = re.compile(
    r"(resolving chord|resolves|embrac\w*|smiles? (?:warmly|softly)|relief|"
    r"relieved|laugh\w* together|walk\w* together|side by side|peaceful|"
    r"gently|tender\w*|comfort\w*|fades? (?:out|to white)|sunrise|happily)",
    re.I,
)
STINGER_SFX = re.compile(
    r"(crack|slap|thud|bang|crash|shatter|explosion|detonation|scream|gasp|"
    r"sting|drone|rumble|clang|whip|punch|collapse|cough\w*|sputter\w*)",
    re.I,
)


@dataclass
class Candidate:
    episode: int
    start_beat: str
    end_beat: str
    start: float
    end: float
    hook_score: float
    cliff_score: float
    tension: float
    score: float
    is_episode_ending: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"E{self.episode:02d} {fmt_ts(self.start)}-{fmt_ts(self.end)}"


def beat_scores(b: Beat) -> dict[str, float]:
    """Return hook / cliff / tension / resolve scores for one beat (0..~5)."""
    text = b.text or ""
    emos = {w.strip().lower() for w in re.split(r"[,/]", b.emotion or "") if w.strip()}
    emo_hits = len(emos & HIGH_EMOTION)
    shock = len(SHOCK_WORDS.findall(text))
    punch = len(DIALOGUE_PUNCH.findall(text))
    cliff = len(CLIFF_MARKERS.findall(text))
    resolve = len(RESOLVE_MARKERS.findall(text))
    sting = len(STINGER_SFX.findall(text)) if b.kind == "SOUND FX" else 0
    weight = {"DIALOGUE": 1.0, "ACTION": 1.0, "SOUND FX": 0.9, "CAMERA": 0.5}.get(b.kind, 0.5)

    tension = weight * (0.8 * min(shock, 3) + 1.0 * emo_hits + 1.2 * punch + 0.8 * sting)
    if b.kind == "DIALOGUE":
        if "!" in text:
            tension += 0.4
        words = len(text.split())
        if words <= 8 and (emo_hits or punch):
            tension += 0.5  # short, sharp line = good hook material
    hook = tension + (0.6 if b.kind == "DIALOGUE" and "?" not in text else 0.0)
    cliff_s = weight * (1.5 * cliff + 0.5 * min(shock, 2)) + (0.6 if b.kind == "DIALOGUE" and "?" in text else 0.0)
    cliff_s += 0.5 * emo_hits
    return {"hook": hook, "cliff": cliff_s - 1.5 * resolve, "tension": tension, "resolve": resolve}


def score_episode(ep: Episode) -> dict[str, dict[str, float]]:
    return {b.id: beat_scores(b) for b in ep.beats}


def _window_value(scores, beats, i, j, lo, hi):
    """Hook = best hook in first 3 s; cliff = best cliff in last 3 s; tension = mean."""
    start, end = beats[i].start, beats[j].end
    first = [scores[b.id]["hook"] for b in beats[i:j + 1] if b.start <= start + 3.0]
    last = [scores[b.id]["cliff"] for b in beats[i:j + 1] if b.end >= end - 3.0]
    resolve_end = sum(scores[b.id]["resolve"] for b in beats[i:j + 1] if b.end >= end - 3.0)
    hook = max(first, default=0.0)
    cliff = max(last, default=0.0) - 1.0 * resolve_end
    inside = [scores[b.id]["tension"] for b in beats[i:j + 1]]
    tension = sum(inside) / max(end - start, 1.0) * 10  # tension per 10 s
    return hook, cliff, tension


def _iou(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def find_candidates(ep: Episode, k: int = 4, min_len: float = 20.0, max_len: float = 60.0) -> list[Candidate]:
    """Top-k non-overlapping 20-60 s windows by hook + cliffhanger + tension.

    Episode endings are always added: microdrama episodes are written to end
    on a cliffhanger, so the last 20-45 s is a natural promo candidate.
    """
    beats = sorted(ep.beats, key=lambda b: (b.start, b.idx))
    if not beats:
        return []
    scores = score_episode(ep)
    cands: list[Candidate] = []
    starts = [i for i, b in enumerate(beats) if b.kind != "CAMERA" or scores[b.id]["hook"] > 0]
    for i in starts:
        for j in range(i, len(beats)):
            dur = beats[j].end - beats[i].start
            if dur < min_len:
                continue
            if dur > max_len:
                break
            hook, cliff, tension = _window_value(scores, beats, i, j, min_len, max_len)
            score = 1.0 * hook + 1.0 * cliff + 0.6 * tension
            cands.append(Candidate(ep.number, beats[i].id, beats[j].id, beats[i].start,
                                   beats[j].end, round(hook, 2), round(cliff, 2),
                                   round(tension, 2), round(score, 2)))
    cands.sort(key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    for c in cands:
        if all(_iou(c, d) < 0.3 for d in chosen):
            chosen.append(c)
        if len(chosen) >= k:
            break
    # always consider the episode ending
    end_j = len(beats) - 1
    best_end = None
    for i in range(end_j, -1, -1):
        dur = beats[end_j].end - beats[i].start
        if dur > 45:
            break
        if dur >= min_len:
            hook, cliff, tension = _window_value(scores, beats, i, end_j, min_len, max_len)
            s = hook + cliff + 0.6 * tension
            if best_end is None or s > best_end.score:
                best_end = Candidate(ep.number, beats[i].id, beats[end_j].id, beats[i].start,
                                     beats[end_j].end, round(hook, 2), round(cliff, 2),
                                     round(tension, 2), round(s, 2), True)
    if best_end and all(_iou(best_end, d) < 0.6 for d in chosen):
        chosen.append(best_end)
    elif best_end:
        for d in chosen:
            if _iou(best_end, d) >= 0.6:
                d.is_episode_ending = True
    chosen.sort(key=lambda c: c.start)
    return chosen


def tension_curve(ep: Episode, step: float = 1.0) -> list[tuple[float, float]]:
    """Per-second tension (for the app's tension map)."""
    scores = score_episode(ep)
    n = int(ep.duration // step) + 1
    curve = [0.0] * n
    for b in ep.beats:
        t = scores[b.id]["tension"]
        if t <= 0:
            continue
        span = max(b.end - b.start, step)
        for s in range(int(b.start // step), min(n, int(b.end // step) + 1)):
            curve[s] += t / (span / step)
    # light smoothing
    sm = [sum(curve[max(0, i - 2): i + 3]) / len(curve[max(0, i - 2): i + 3]) for i in range(n)]
    return [(i * step, round(v, 3)) for i, v in enumerate(sm)]
