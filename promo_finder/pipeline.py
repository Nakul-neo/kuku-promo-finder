"""End-to-end promo finder.

Stages
  0. parse        screenplay -> numbered beats (code)
  1. pre-filter   rule-based tension scoring -> hot windows per episode (code, free)
  2. propose      AI reads only hot windows, proposes promos by beat id (AI)
  3. stitch       AI joins moments across episodes into arc promos (AI)
  4. validate     code computes timestamps, enforces 20-60 s, hook in first 3 s,
                  no resolved endings, dedupes; fixes or drops bad picks (code)
  5. cold viewer  a fresh AI with zero show context 'watches' each promo (AI)
  6. rank         transparent weighted score + funnel logic + posting plan (code)
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import prompts as P
from .llm import LLM, LLMError
from .parse import Episode, Beat, load_show, fmt_ts, beats_table
from .prefilter import find_candidates, score_episode, Candidate

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0

WEIGHTS = {
    "stop": 0.28,       # cold viewer: first 3 s stop the scroll
    "need_next": 0.28,  # cold viewer: needs to know what happens next
    "clarity": 0.16,    # cold viewer understood it without context
    "tension": 0.10,    # rule-based tension density (independent of AI)
    "funnel": 0.18,     # earlier episode = shorter path from promo to payoff
}
PENALTIES = {"spoiler_1": 0.03, "spoiler_2": 0.12, "ends_resolved": 0.08,
             "hook_late": 0.06, "length_fixed": 0.02, "extended_back": 0.04, "editor_rejected_before": 0.25}


@dataclass
class Segment:
    episode: int
    start_beat: str
    end_beat: str
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class Promo:
    id: str
    kind: str                     # single | stitched
    segments: list[Segment]
    hook_beat: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    trope: str = ""
    spoiler_level: int = 0
    context_needed: str = ""
    source: str = "ai"            # ai | rules
    flags: list[str] = field(default_factory=list)
    tension: float = 0.0
    cold: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def where(self) -> str:
        return " + ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in self.segments)


# ---------------------------------------------------------------- helpers
def _items(data, key: str) -> list:
    """Models sometimes return a bare JSON list instead of {key: [...]}; accept both."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        v = data.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        for v in data.values():  # any other single list value
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _ep_of(beat_id: str) -> int | None:
    m = re.match(r"E(\d+)-\d+", beat_id or "")
    return int(m.group(1)) if m else None


def render_window(ep: Episode, start: float, end: float, pad: float = 8.0) -> list[Beat]:
    return [b for b in ep.beats if b.end >= start - pad and b.start <= end + pad]


def render_promo_for_viewer(promo: Promo, idx: dict[str, Beat], episodes: dict[int, Episode]) -> str:
    """Promo as a viewer would experience it: relative seconds, no show context."""
    out, offset = [], 0.0
    for k, seg in enumerate(promo.segments):
        if k:
            out.append("[CUT]")
        ep = episodes[seg.episode]
        s_idx, e_idx = idx[seg.start_beat].idx, idx[seg.end_beat].idx
        for b in ep.beats:
            if s_idx <= b.idx <= e_idx:
                t = offset + max(0.0, b.start - seg.start)
                if b.kind == "DIALOGUE":
                    emo = f" ({b.emotion})" if b.emotion else ""
                    out.append(f"{t:5.1f}s  {b.speaker.title()}{emo}: {b.text}")
                elif b.kind == "CAMERA":
                    out.append(f"{t:5.1f}s  [shot] {b.text[:110]}")
                else:
                    out.append(f"{t:5.1f}s  [{b.kind.lower()}] {b.text}")
        offset += seg.duration
    return "\n".join(out)


def load_feedback(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def lessons_text(feedback: list[dict], limit: int = 20) -> str:
    if not feedback:
        return ""
    lines = []
    for f in feedback[-limit:]:
        verdict = "APPROVED" if f.get("decision") == "approve" else "REJECTED"
        reason = f.get("reason", "")
        note = f" - {f['note']}" if f.get("note") else ""
        lines.append(f"- {verdict} ({reason}{note}): hook was \"{f.get('hook', '')[:100]}\"")
    return ("EDITOR FEEDBACK FROM PAST RUNS (learn from it):\n" + "\n".join(lines) + "\n\n")


# ---------------------------------------------------------------- pipeline
class PromoFinder:
    def __init__(self, llm: LLM, show_name: str = "Untitled show", feedback_path="feedback/feedback.jsonl",
                 batch_episodes: int = 25, cold_batch: int = 25, n_stitched: int = 5,
                 per_episode_candidates: int = 3, time_budget_min: float = 15, log=None, progress=None):
        self.llm = llm
        self.show = show_name
        self.feedback_path = feedback_path
        self.batch_episodes = batch_episodes
        self.cold_batch = cold_batch
        self.n_stitched = n_stitched
        self.k = per_episode_candidates
        self.log = log or (lambda m: print(m, flush=True))
        self.progress = progress or (lambda frac, msg: None)
        self.ai = llm.provider != "baseline"
        self.time_budget_min = time_budget_min
        self.degraded: list[str] = []

    # 1 ---------------------------------------------------------------
    def prefilter(self, episodes: list[Episode]) -> dict[int, list[Candidate]]:
        return {e.number: find_candidates(e, k=self.k, min_len=MIN_LEN, max_len=MAX_LEN) for e in episodes}

    # 2 ---------------------------------------------------------------
    def _rules_promos(self, episodes, cands, n: int = 2, why_prefix: str = "") -> list[Promo]:
        out = []
        for e in episodes:
            idx = e.beat_index()
            for c in sorted(cands[e.number], key=lambda c: c.score, reverse=True)[:n]:
                out.append(Promo(
                    id="", kind="single", source="rules",
                    segments=[Segment(e.number, c.start_beat, c.end_beat)],
                    hook_beat=c.start_beat, hook=idx[c.start_beat].text[:140],
                    cliffhanger=idx[c.end_beat].text[:140],
                    why=f"{why_prefix}rule score {c.score} (hook {c.hook_score}, cliff {c.cliff_score})",
                    trope="(rules)"))
        return out

    def _episode_block(self, e: Episode, cands) -> str:
        lines = [f"=== EPISODE {e.number} ===", f"Logline: {e.logline}"]
        sc = score_episode(e)
        for wi, c in enumerate(cands[e.number], 1):
            tag = " (episode ending)" if c.is_episode_ending else ""
            lines.append(f"-- hot spot {wi}{tag}: {fmt_ts(c.start)}-{fmt_ts(c.end)}")
            # token saver: keep all dialogue/action/sfx, but only camera lines that carry tension
            lines += [b.render(90) for b in render_window(e, c.start, c.end, pad=5.0)
                      if b.kind != "CAMERA" or sc[b.id]["tension"] > 0]
        return "\n".join(lines)

    def propose(self, episodes, cands, feedback) -> list[Promo]:
        if not self.ai:
            return self._rules_promos(episodes, cands)
        promos: list[Promo] = []
        lessons = lessons_text(feedback)
        batches = [episodes[i:i + self.batch_episodes] for i in range(0, len(episodes), self.batch_episodes)]
        for bi, batch in enumerate(batches):
            label = f"{batch[0].number:02d}-{batch[-1].number:02d}"
            prompt = P.PROPOSE_USER.format(show=self.show, lessons=lessons,
                                           episodes="\n\n".join(self._episode_block(e, cands) for e in batch))
            self.progress(0.1 + 0.4 * bi / len(batches), f"AI reading episodes {batch[0].number}-{batch[-1].number}")
            try:
                data = self.llm.json(P.PROPOSE_SYSTEM, prompt, tag=f"propose_{label}")
            except LLMError as err:
                # never return nothing: fall back to the rule-based picks for these episodes, clearly labelled
                self.log(f"[propose] AI failed for episodes {label} ({str(err)[:120]}); using rule-based picks")
                self.degraded.append(f"episodes {label}: AI unavailable, rule-based picks used")
                promos += self._rules_promos(batch, cands, why_prefix="AI unavailable - ")
                continue
            got = [p for p in (self._promo_from_raw(r, "single") for r in _items(data, "promos")) if p]
            self.log(f"[propose] episodes {label}: {len(got)} promos")
            promos += got
        return promos

    def _promo_from_raw(self, raw: dict, kind: str) -> Promo | None:
        segs = []
        for s in raw.get("segments", []) or []:
            sb, eb = str(s.get("start_beat", "")).strip(), str(s.get("end_beat", "")).strip()
            ep = _ep_of(sb)
            if ep is None:
                continue
            segs.append(Segment(ep, sb, eb))
        if not segs:
            return None
        try:
            spoiler = int(raw.get("spoiler_level", 0) or 0)
        except (TypeError, ValueError):
            spoiler = 0
        spoiler = max(0, min(spoiler, 2))  # models sometimes go off the 0-2 scale
        return Promo(id="", kind=kind, segments=segs, hook_beat=str(raw.get("hook_beat", "")),
                     hook=str(raw.get("hook", "")), cliffhanger=str(raw.get("cliffhanger", "")),
                     why=str(raw.get("why", "")), trope=str(raw.get("trope", "")),
                     spoiler_level=spoiler, context_needed=str(raw.get("context_needed", "") or ""))

    # 3 ---------------------------------------------------------------
    def stitch(self, episodes, singles: list[Promo], idx) -> list[Promo]:
        if not self.ai or self.n_stitched <= 0 or not singles:
            return []
        epmap = {e.number: e for e in episodes}
        synopsis = "\n".join(f"Ep{e.number}: {e.logline}" for e in episodes)
        menu = []
        for p in singles:
            s = p.segments[0]
            ep = epmap[s.episode]
            a, b = idx[s.start_beat].idx, idx[s.end_beat].idx
            key = [x.render() for x in ep.beats if a <= x.idx <= b and x.kind != "CAMERA"]
            menu.append(f"-- moment {p.id} (Ep{s.episode}, {p.trope}): {p.hook} / ends: {p.cliffhanger}\n" + "\n".join(key))
        prompt = P.ARC_USER.format(show=self.show, synopsis=synopsis, menu="\n\n".join(menu), n=self.n_stitched)
        self.progress(0.55, "AI stitching cross-episode promos")
        try:
            data = self.llm.json(P.ARC_SYSTEM, prompt, tag="stitch")
        except LLMError as err:
            self.log(f"[stitch] skipped ({str(err)[:120]})")
            self.degraded.append("stitched promos: AI unavailable, skipped")
            return []
        out = []
        for raw in _items(data, "promos"):
            p = self._promo_from_raw(raw, "stitched")
            if p and len(p.segments) >= 2:
                out.append(p)
        return out

    # 4 ---------------------------------------------------------------
    def validate(self, promos: list[Promo], idx: dict[str, Beat], epmap: dict[int, Episode],
                 scores: dict[str, dict], feedback: list[dict], prefix: str = "P") -> tuple[list[Promo], list[dict]]:
        kept, dropped = [], []
        rejected = [f for f in feedback if f.get("decision") == "reject"]
        for p in promos:
            reason = self._fix(p, idx, epmap, scores)
            if reason:
                dropped.append({"hook": p.hook, "segments": [asdict(s) for s in p.segments], "reason": reason})
                continue
            for f in rejected:
                if _overlap_promo(p, f.get("segments", [])) > 0.6:
                    p.flags.append("editor_rejected_before")
                    break
            kept.append(p)
        # dedupe: drop near-identical single-episode promos (keep the first proposed)
        final: list[Promo] = []
        for p in kept:
            dup = next((q for q in final if _overlap_promo(p, [asdict(s) for s in q.segments]) > 0.6), None)
            if dup:
                dropped.append({"hook": p.hook, "segments": [asdict(s) for s in p.segments],
                                "reason": f"duplicate of {dup.id or 'earlier promo'}"})
                continue
            final.append(p)
        for i, p in enumerate(final, 1):
            p.id = f"{prefix}{i:03d}"
        return final, dropped

    def _fix(self, p: Promo, idx, epmap, scores) -> str | None:
        # ids exist and belong to one episode per segment
        for s in p.segments:
            if s.start_beat not in idx or s.end_beat not in idx:
                return f"unknown beat id ({s.start_beat}/{s.end_beat}) - model invented an id"
            if _ep_of(s.end_beat) != s.episode:
                return "segment spans two episodes"
            if idx[s.end_beat].idx < idx[s.start_beat].idx:
                s.start_beat, s.end_beat = s.end_beat, s.start_beat
                p.flags.append("segment_reversed")
            s.start, s.end = idx[s.start_beat].start, idx[s.end_beat].end
        # hook must land in the first 3 s: snap the cold open to the hook beat if needed
        first = p.segments[0]
        hb = idx.get(p.hook_beat)
        if hb and hb.episode == first.episode and idx[first.start_beat].idx <= hb.idx <= idx[first.end_beat].idx:
            if hb.start - first.start > HOOK_WINDOW:
                if p.duration - (hb.start - first.start) >= MIN_LEN:
                    first.start_beat, first.start = hb.id, hb.start
                    p.flags.append("snapped_to_hook")
                else:
                    p.flags.append("hook_late")
        # length: fix to 20-60 s without inventing anything
        if p.duration < MIN_LEN or p.duration > MAX_LEN:
            reason = self._fix_length(p, idx, epmap, scores)
            if reason:
                return reason
            p.flags.append("length_fixed")
        # resolved ending check (last 3 s)
        last = p.segments[-1]
        ep = epmap[last.episode]
        tail = [b for b in ep.beats if idx[last.start_beat].idx <= b.idx <= idx[last.end_beat].idx
                and b.end >= last.end - HOOK_WINDOW]
        if sum(scores[b.id]["resolve"] for b in tail) > 0:
            p.flags.append("ends_resolved")
        # independent tension density
        inside = [b for s in p.segments for b in epmap[s.episode].beats
                  if idx[s.start_beat].idx <= b.idx <= idx[s.end_beat].idx]
        p.tension = round(sum(scores[b.id]["tension"] for b in inside) / max(p.duration, 1) * 10, 2)
        return None

    def _fix_length(self, p: Promo, idx, epmap, scores) -> str | None:
        """Too long: shorten the longest segment (cut on its most cliffhanger-y beat), repeat;
        drop a segment only if it would shrink below 4 s. Too short: extend the last segment
        forward; if the episode ends there (episode-ending cliffhanger), extend the first
        segment backward into the build-up, starting on the strongest hook beat."""
        guard = 0
        while p.duration > MAX_LEN and guard < 10:
            guard += 1
            over = p.duration - MAX_LEN
            seg = max(p.segments, key=lambda s: s.duration)
            ep = epmap[seg.episode]
            a, b = idx[seg.start_beat].idx, idx[seg.end_beat].idx
            target = seg.duration - over
            if target < 4.0 and len(p.segments) > 1:
                p.segments.remove(seg)
                continue
            lo = max(target - 15.0, 4.0 if len(p.segments) > 1 else MIN_LEN)
            opts = [x for x in ep.beats if a <= x.idx <= b and lo <= x.end - seg.start <= target]
            if not opts:
                return f"length {p.duration:.1f}s cannot be fixed to 20-60s"
            best = max(opts, key=lambda x: (scores[x.id]["cliff"], x.end))
            seg.end_beat, seg.end = best.id, best.end
        if p.duration > MAX_LEN:
            return f"length {p.duration:.1f}s cannot be fixed to 20-60s"
        if p.duration < MIN_LEN:
            last = p.segments[-1]
            ep = epmap[last.episode]
            others = p.duration - last.duration
            lo, hi = MIN_LEN - others, MAX_LEN - others
            s_idx = idx[last.start_beat].idx
            fwd = [x for x in ep.beats if x.idx >= s_idx and lo <= x.end - last.start <= hi]
            if fwd:
                best = max(fwd, key=lambda x: (scores[x.id]["cliff"], -abs(x.end - last.start - 35)))
                last.end_beat, last.end = best.id, best.end
                return None
            first = p.segments[0]
            ep = epmap[first.episode]
            e_idx = idx[first.end_beat].idx
            need_lo = MIN_LEN - p.duration
            need_hi = MAX_LEN - p.duration
            back = [x for x in ep.beats if x.idx <= e_idx and need_lo <= first.start - x.start <= need_hi
                    and x.kind != "CAMERA"]
            if not back:
                return f"length {p.duration:.1f}s cannot be fixed to 20-60s"
            best = max(back, key=lambda x: (scores[x.id]["hook"], -abs(first.start - x.start - need_lo - 10)))
            shift = first.start - best.start
            first.start_beat, first.start = best.id, best.start
            p.hook_beat = best.id
            p.hook = f"{p.hook} [start moved {shift:.0f}s earlier into the build-up to reach 20 s]"
            p.flags.append("extended_back")
        return None

    # 5 ---------------------------------------------------------------
    @staticmethod
    def _rule_cold(p: Promo, epmap) -> dict:
        """Rule-based stand-in for the cold-viewer test (used without AI, or if the AI call fails)."""
        s0 = p.segments[0]
        ep = epmap[s0.episode]
        sc = score_episode(ep)
        first = [sc[b.id]["hook"] for b in ep.beats if s0.start <= b.start <= s0.start + 3]
        return {"first_3s_stop": min(10, round(3 + 1.5 * max(first, default=0))),
                "need_next": min(10, round(4 + 0.8 * max(sc[p.segments[-1].end_beat]["cliff"], 0))),
                "would_tap": None, "understood": "(rule estimate - no AI viewer check)", "confusing": [],
                "why_stop": "", "feeling": "", "source": "rules"}

    def cold_viewer(self, promos: list[Promo], idx, epmap) -> None:
        if not self.ai:
            for p in promos:
                p.cold = self._rule_cold(p, epmap)
            return
        cold_model = __import__("os").getenv("COLD_VIEWER_MODEL") or None
        batches = [promos[i:i + self.cold_batch] for i in range(0, len(promos), self.cold_batch)]
        for bi, batch in enumerate(batches):
            clips = "\n\n".join(f"=== CLIP {p.id} ({p.duration:.0f}s) ===\n{render_promo_for_viewer(p, idx, epmap)}"
                                for p in batch)
            self.progress(0.6 + 0.3 * bi / len(batches), f"Cold-viewer test {bi + 1}/{len(batches)}")
            try:
                data = self.llm.json(P.COLD_SYSTEM, P.COLD_USER.format(clips=clips),
                                     tag=f"cold_{bi + 1:02d}", model=cold_model)
                by_id = {str(v.get("clip_id")): v for v in _items(data, "verdicts")}
            except LLMError as err:
                self.log(f"[cold] AI viewer failed for {len(batch)} promos ({str(err)[:120]}); using rule estimate")
                self.degraded.append(f"cold-viewer batch {bi + 1}: AI unavailable, rule estimate used")
                by_id = {}
            for p in batch:
                if p.id in by_id:
                    p.cold = by_id[p.id] | {"source": "ai"}
                else:
                    p.cold = self._rule_cold(p, epmap)
                    p.flags.append("no_ai_viewer_check")
            self.log(f"[cold] batch {bi + 1}/{len(batches)}: {sum(1 for p in batch if p.id in by_id)} AI verdicts")

    # 6 ---------------------------------------------------------------
    def rank(self, promos: list[Promo], n_episodes: int) -> list[Promo]:
        t_vals = sorted(p.tension for p in promos) or [0]
        for p in promos:
            c = p.cold or {}
            stop = _num(c.get("first_3s_stop"), 5) / 10
            need = _num(c.get("need_next"), 5) / 10
            conf = c.get("confusing") or []
            conf = [x for x in conf if str(x).strip() and str(x).strip().lower() not in ("none", "nothing", "n/a")]
            clarity = 1 - 0.2 * min(len(conf), 3) - (0.2 if p.context_needed.strip() else 0)
            tension = _pct_rank(p.tension, t_vals)
            first_ep = min(p.episodes)
            funnel = 1.0 if first_ep <= 3 else max(0.35, 1 - 0.65 * math.log(first_ep / 3) / math.log(max(n_episodes, 4) / 3))
            parts = {"stop": stop, "need_next": need, "clarity": clarity, "tension": tension, "funnel": funnel}
            base = sum(WEIGHTS[k] * v for k, v in parts.items())
            pen = 0.0
            if p.spoiler_level >= 2:
                pen += PENALTIES["spoiler_2"]
            elif p.spoiler_level == 1:
                pen += PENALTIES["spoiler_1"]
            for f in p.flags:
                pen += PENALTIES.get(f, 0.0)
            bonus = 0.04 if c.get("would_tap") is True else 0.0
            p.scores = {k: round(v, 3) for k, v in parts.items()} | {"penalty": round(pen, 3), "bonus": bonus}
            p.score = round(base - pen + bonus, 4)
        promos.sort(key=lambda p: p.score, reverse=True)
        for i, p in enumerate(promos, 1):
            p.rank = i
        return promos

    def posting_plan(self, ranked: list[Promo], days: int = 7) -> list[dict]:
        """Week-1 order: open with the premise, never repeat an episode or trope back to back,
        max one big-spoiler promo, and include at least one stitched promo if it ranks well."""
        plan, used = [], set()
        pool = [p for p in ranked if "editor_rejected_before" not in p.flags]
        premise = next((p for p in pool if min(p.episodes) <= 3), None)
        if premise:
            plan.append((premise, "Day 1 = the premise: viewers who tap land at the start of the story"))
            used.add(premise.id)
        spoilers = sum(1 for p, _ in plan if p.spoiler_level >= 2)
        while len(plan) < days:
            prev = plan[-1][0] if plan else None
            pick = None
            for p in pool:
                if p.id in used:
                    continue
                if prev and (set(p.episodes) & set(prev.episodes) or p.trope.lower() == prev.trope.lower()):
                    continue
                if p.spoiler_level >= 2 and spoilers >= 1:
                    continue
                # don't post the same footage twice in a week (stitched promos reuse single-episode moments)
                if any(_overlap_promo(p, [asdict(s) for s in q.segments]) > 0.5 for q, _ in plan):
                    continue
                pick = p
                break
            if not pick:
                break
            used.add(pick.id)
            spoilers += pick.spoiler_level >= 2
            plan.append((pick, "highest remaining score that keeps the feed varied"))
        return [{"day": i + 1, "promo_id": p.id, "where": p.where(), "hook": p.hook, "trope": p.trope,
                 "score": p.score, "reason": r} for i, (p, r) in enumerate(plan)]

    # ---------------------------------------------------------------
    def run(self, source, out_dir: str | Path, source_name: str = "",
            episode_range: tuple[int, int] | None = None) -> dict:
        t0 = time.time()
        self.degraded = []
        # hard time budget: after this, any remaining AI step falls back to rules instead of waiting
        self.llm.deadline = t0 + self.time_budget_min * 60 if self.ai else None
        out = Path(out_dir)
        (out / "raw_llm").mkdir(parents=True, exist_ok=True)
        self.progress(0.02, "Parsing screenplay")
        episodes = load_show(source, source_name) if not isinstance(source, list) else source
        if episode_range:
            episodes = [e for e in episodes if episode_range[0] <= e.number <= episode_range[1]]
        if not episodes:
            raise ValueError("No episodes with timestamped beats found in the input")
        epmap = {e.number: e for e in episodes}
        idx = {b.id: b for e in episodes for b in e.beats}
        scores = {k: v for e in episodes for k, v in score_episode(e).items()}
        beats_table(episodes).to_csv(out / "beats.csv", index=False)
        self.log(f"[parse] {len(episodes)} episodes, {len(idx)} beats")

        cands = self.prefilter(episodes)
        _dump(out / "prefilter_candidates.json",
              {e: [asdict(c) | {"label": c.label()} for c in cs] for e, cs in cands.items()})
        self.log(f"[prefilter] {sum(len(v) for v in cands.values())} hot windows")
        feedback = load_feedback(self.feedback_path)

        singles = self.propose(episodes, cands, feedback)
        self.log(f"[propose] {len(singles)} single-episode proposals")
        singles, dropped1 = self.validate(singles, idx, epmap, scores, feedback, prefix="P")
        stitched = self.stitch(episodes, singles, idx)
        self.log(f"[stitch] {len(stitched)} stitched proposals")
        stitched, dropped2 = self.validate(stitched, idx, epmap, scores, feedback, prefix="S")
        allp, dropped = singles + stitched, dropped1 + dropped2
        self.log(f"[validate] kept {len(allp)}, dropped {len(dropped)}")

        self.cold_viewer(allp, idx, epmap)
        ranked = self.rank(allp, len(episodes))
        plan = self.posting_plan(ranked)

        for p in ranked:
            p.__dict__["script"] = render_promo_for_viewer(p, idx, epmap)
        result = {
            "show": self.show,
            "model": self.llm.describe(),
            "prompt_version": P.PROMPT_VERSION,
            "episodes": len(episodes),
            "beats": len(idx),
            "promos": [_promo_dict(p) for p in ranked],
            "posting_plan": plan,
            "dropped": dropped,
            "degraded": self.degraded,
            "models_used": sorted({c["model"] for c in self.llm.calls if c.get("model")}),
            "usage": self.llm.usage_summary() | {"wall_time_s": round(time.time() - t0, 1)},
            "weights": WEIGHTS, "penalties": PENALTIES,
        }
        _dump(out / "promos.json", result)
        write_csv(result, out / "promos.csv")
        (out / "promos_report.md").write_text(write_report(result), encoding="utf-8")
        # raw model I/O for this run (unedited prompts + responses)
        import shutil
        for c in self.llm.calls:
            if c.get("cache_file") and Path(c["cache_file"]).exists():
                shutil.copy(c["cache_file"], out / "raw_llm" / f"{c['tag']}.json")
        self.progress(1.0, "Done")
        if self.degraded:
            self.log("[done] NOTE - parts ran without AI: " + "; ".join(self.degraded))
        self.log(f"[done] {len(ranked)} promos -> {out}")
        return result


# ---------------------------------------------------------------- utils
def _num(x, default):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _pct_rank(v, sorted_vals):
    if len(sorted_vals) <= 1:
        return 0.5
    below = sum(1 for x in sorted_vals if x < v)
    return below / (len(sorted_vals) - 1)


def _overlap_promo(p: Promo, segs: list[dict]) -> float:
    """Time overlap between a promo and another promo's segments (fraction of the shorter)."""
    tot = 0.0
    for s in p.segments:
        for o in segs:
            if int(o.get("episode", -1)) != s.episode:
                continue
            tot += max(0.0, min(s.end, float(o["end"])) - max(s.start, float(o["start"])))
    other = sum(float(o["end"]) - float(o["start"]) for o in segs) or 1.0
    return tot / max(min(p.duration, other), 1.0)


def _promo_dict(p: Promo) -> dict:
    d = asdict(p)
    d["duration"] = p.duration
    d["episodes"] = p.episodes
    d["where"] = p.where()
    d["segments"] = [asdict(s) | {"start_ts": fmt_ts(s.start), "end_ts": fmt_ts(s.end),
                                  "duration": s.duration} for s in p.segments]
    d["script"] = p.__dict__.get("script", "")
    return d


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(result: dict, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "type", "score", "duration_s", "segments (episode start-end)", "hook",
                    "cliffhanger", "why", "trope", "spoiler_level", "cold_stop_1to10", "cold_need_next_1to10",
                    "cold_would_tap", "cold_understood", "flags"])
        for p in result["promos"]:
            c = p.get("cold") or {}
            w.writerow([p["rank"], p["id"], p["kind"], p["score"], p["duration"], p["where"], p["hook"],
                        p["cliffhanger"], p["why"], p["trope"], p["spoiler_level"], c.get("first_3s_stop"),
                        c.get("need_next"), c.get("would_tap"), c.get("understood"), ";".join(p["flags"])])


def write_report(result: dict, top: int = 25) -> str:
    L = [f"# Promo report: {result['show']}",
         f"Model: `{result['model']}` · prompts {result['prompt_version']} · {result['episodes']} episodes · "
         f"{len(result['promos'])} promos kept · {len(result['dropped'])} dropped by validator", ""]
    if result.get("models_used"):
        L.append(f"Models that answered: {', '.join(result['models_used'])}")
    if result.get("degraded"):
        L.append("**Ran partly without AI:** " + "; ".join(result["degraded"]))
    u = result["usage"]
    L.append(f"AI calls: {u['calls_total']} ({u['calls_live']} live) · tokens in/out: "
             f"{u['input_tokens']:,}/{u['output_tokens']:,} · wall time {u['wall_time_s']}s\n")
    L.append("## Week-1 posting plan\n")
    L.append("| Day | Promo | Where | Hook | Why this slot |\n|---|---|---|---|---|")
    for d in result["posting_plan"]:
        L.append(f"| {d['day']} | {d['promo_id']} | {d['where']} | {d['hook']} | {d['reason']} |")
    L.append(f"\n## Top {top} promos\n")
    for p in result["promos"][:top]:
        c = p.get("cold") or {}
        L += [f"### #{p['rank']} {p['id']} · score {p['score']:.3f} · {p['duration']:.0f}s · {p['kind']} · {p['trope']}",
              "| # | Episode | Start | End | Length |\n|---|---|---|---|---|"]
        for i, s in enumerate(p["segments"], 1):
            L.append(f"| {i} | {s['episode']} | {s['start_ts']} | {s['end_ts']} | {s['duration']:.1f}s |")
        L += ["", f"- **Hook:** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}", f"- **Why it works:** {p['why']}",
              f"- **Cold viewer:** stop {c.get('first_3s_stop')}/10 · need-next {c.get('need_next')}/10 · "
              f"tap: {c.get('would_tap')} · \"{c.get('understood', '')}\""]
        if c.get("confusing"):
            L.append(f"- **Confused by:** {'; '.join(map(str, c['confusing']))}")
        if p["flags"]:
            L.append(f"- **Validator flags:** {', '.join(p['flags'])}")
        L.append(f"- Score parts: {p['scores']}\n")
    if result["dropped"]:
        L.append("## Dropped by validator\n")
        for d in result["dropped"]:
            L.append(f"- {d['reason']} - \"{d['hook'][:90]}\"")
    return "\n".join(L)
