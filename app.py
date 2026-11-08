"""Streamlit app: upload a screenplay -> ranked promos -> editor review.

Run locally:  streamlit run app.py
"""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.llm import LLM, LLMError, load_dotenv
from promo_finder.parse import load_show, fmt_ts
from promo_finder.pipeline import PromoFinder, load_feedback
from promo_finder.prefilter import tension_curve

load_dotenv()
# Streamlit Cloud: settings live in the app's Secrets box
try:
    for _k in ("GEMINI_API_KEY", "LLM_MODEL", "LLM_FALLBACK_MODELS", "LLM_THINKING", "COLD_VIEWER_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001  (no secrets file locally)
    pass
FEEDBACK = Path("feedback/feedback.jsonl")

def svg_chart(xs, ys, kind="line", height=180, color="#ff4b4b", x_label="", y_label=""):
    """Dependency-free chart (inline SVG). Streamlit's built-in charts need Altair,
    which fails on some Python versions on Streamlit Cloud, so we draw our own."""
    if not ys:
        st.caption("(no data)")
        return
    w, h, pad = 760, height, 28
    ymax = max(max(ys), 1e-9)
    xmin, xmax = min(xs), max(xs)
    xr = (xmax - xmin) or 1
    sx = lambda x: pad + (x - xmin) / xr * (w - 2 * pad)
    sy = lambda y: h - pad - y / ymax * (h - 2 * pad)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:transparent;font:11px sans-serif;fill:#999">',
             f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#888" stroke-width="1"/>']
    if kind == "line":
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    else:
        bw = max((w - 2 * pad) / max(len(xs), 1) * 0.7, 2)
        for x, y in zip(xs, ys):
            parts.append(f'<rect x="{sx(x) - bw / 2:.1f}" y="{sy(y):.1f}" width="{bw:.1f}" '
                         f'height="{h - pad - sy(y):.1f}" fill="{color}"/>')
    parts.append(f'<text x="{pad}" y="{h - 8}">{xmin:g}</text><text x="{w - pad}" y="{h - 8}" '
                 f'text-anchor="end">{xmax:g} {x_label}</text><text x="{pad}" y="14">max {ymax:g} {y_label}</text></svg>')
    st.markdown("".join(parts), unsafe_allow_html=True)

REJECT_REASONS = ["weak hook (first 3s)", "no real cliffhanger", "needs context / confusing",
                  "too spoilery", "bad cut point", "duplicate of another promo", "visual-only moment (script oversells)", "other"]

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.caption("Reads a microdrama screenplay and finds 20-60s promo clips that open on a hook, cut on a cliffhanger "
           "and make sense to a stranger. The AI picks script lines by id; timestamps always come from the script.")


ARCH_SVG = r"""<svg viewBox="0 0 700 395" width="100%">
 <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#666"/></marker></defs>
 <g font-size="11">
 <rect x="10" y="10" width="150" height="44" rx="6" fill="#fff" stroke="#999"/><text x="85" y="28" text-anchor="middle" font-weight="700">INPUT</text><text x="85" y="44" text-anchor="middle">50 episode scripts (.txt / .zip)</text>
 <line x1="160" y1="32" x2="196" y2="32" stroke="#666" marker-end="url(#a)"/>
 <rect x="200" y="10" width="150" height="44" rx="6" fill="#ececef" stroke="#aaa"/><text x="275" y="28" text-anchor="middle" font-weight="700">① PARSE</text><text x="275" y="44" text-anchor="middle">number every line + its time</text>
 <line x1="350" y1="32" x2="386" y2="32" stroke="#666" marker-end="url(#a)"/>
 <rect x="390" y="10" width="160" height="44" rx="6" fill="#ececef" stroke="#aaa"/><text x="470" y="28" text-anchor="middle" font-weight="700">② PRE-FILTER (rules)</text><text x="470" y="44" text-anchor="middle">highlight ~3 hot spots / episode</text>
 <line x1="470" y1="54" x2="470" y2="86" stroke="#666" marker-end="url(#a)"/>
 <rect x="390" y="90" width="160" height="48" rx="6" fill="#dde8ff" stroke="#7d9be0"/><text x="470" y="108" text-anchor="middle" font-weight="700">③ PROPOSE (Gemini call 1)</text><text x="470" y="125" text-anchor="middle">picks promos by line number</text>
 <line x1="390" y1="114" x2="356" y2="114" stroke="#666" marker-end="url(#a)"/>
 <rect x="200" y="90" width="150" height="48" rx="6" fill="#dde8ff" stroke="#7d9be0"/><text x="275" y="108" text-anchor="middle" font-weight="700">④ STITCH (Gemini call 2)</text><text x="275" y="125" text-anchor="middle">joins moments across eps</text>
 <line x1="275" y1="138" x2="275" y2="170" stroke="#666" marker-end="url(#a)"/>
 <rect x="170" y="174" width="210" height="56" rx="6" fill="#ececef" stroke="#aaa"/><text x="275" y="192" text-anchor="middle" font-weight="700">⑤ VALIDATE (Python rules)</text><text x="275" y="208" text-anchor="middle">real times · 20–60 s · hook first</text><text x="275" y="222" text-anchor="middle">fix or drop bad picks</text>
 <line x1="380" y1="202" x2="416" y2="202" stroke="#666" marker-end="url(#a)"/>
 <rect x="420" y="174" width="170" height="56" rx="6" fill="#dde8ff" stroke="#7d9be0"/><text x="505" y="192" text-anchor="middle" font-weight="700">⑥ COLD VIEWER (call 3)</text><text x="505" y="208" text-anchor="middle">fresh AI, knows nothing,</text><text x="505" y="222" text-anchor="middle">judges each clip</text>
 <line x1="505" y1="230" x2="505" y2="262" stroke="#666" marker-end="url(#a)"/>
 <rect x="420" y="266" width="170" height="48" rx="6" fill="#ececef" stroke="#aaa"/><text x="505" y="284" text-anchor="middle" font-weight="700">⑦ RANK + PLAN (Python)</text><text x="505" y="300" text-anchor="middle">score formula · 7-day plan</text>
 <line x1="420" y1="290" x2="386" y2="290" stroke="#666" marker-end="url(#a)"/>
 <rect x="200" y="266" width="180" height="48" rx="6" fill="#dff3df" stroke="#7fbf7f"/><text x="290" y="284" text-anchor="middle" font-weight="700">⑧ WEBSITE + EDITOR</text><text x="290" y="300" text-anchor="middle">approve / reject + reason</text>
 <path d="M200,290 C120,290 110,150 190,125" fill="none" stroke="#2b7a2b" stroke-dasharray="4 3" marker-end="url(#a)"/>
 <text x="70" y="215" fill="#2b7a2b" font-size="10">editor's reasons are</text><text x="70" y="228" fill="#2b7a2b" font-size="10">added to the next</text><text x="70" y="241" fill="#2b7a2b" font-size="10">run's Gemini prompt</text>
 <rect x="10" y="335" width="680" height="50" rx="6" fill="#fff8ec" stroke="#e0b060"/>
 <text x="20" y="354" font-weight="700">Safety net around every Gemini call:</text>
 <text x="20" y="368">saved answers (no paying twice) · backup models if one is busy or out of quota · max 60 s wait</text><text x="20" y="381">time budget · if the AI fails → the rule-based picks are used instead (clearly labelled)</text>
 </g>
</svg>"""


def show_architecture():
    import streamlit.components.v1 as components
    with st.expander("🧭 How it works: architecture", expanded=False):
        components.html(
            '<div style="background:#fff;border-radius:8px;padding:10px;font-family:sans-serif">'
            + ARCH_SVG +
            '<div style="font-size:12px;margin-top:6px">'
            '<b style="background:#ececef;padding:1px 5px">GREY</b> Python code &nbsp; '
            '<b style="background:#dde8ff;padding:1px 5px">BLUE</b> Gemini (AI) &nbsp; '
            '<b style="background:#dff3df;padding:1px 5px">GREEN</b> human editor</div></div>',
            height=560, scrolling=False)
        st.markdown(
            "- **AI judges, code enforces.** Gemini decides what is dramatic; Python handles exact times, the 20-60 s rule, ranking.\n"
            "- **Gemini never writes timestamps.** It names script line IDs (e.g. E03-047); code looks up the real times.\n"
            "- **A separate 'cold viewer' AI** sees only the clip (no title, no plot) and says if a stranger would stop and tap.\n"
            "- **Ranking:** 28% stop-scroll + 28% need-next + 16% clarity + 10% rule tension + 18% early episode, minus penalties.\n"
            "- **Safety net:** saved answers, backup models, max 60 s wait, time budget; if AI fails, rule-based picks are used.")


show_architecture()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # noqa: BLE001
        secret_key = ""
    provider = st.selectbox("Provider", ["gemini", "openai-compatible", "baseline (rules only, no key)"],
                            index=0 if (os.getenv("GEMINI_API_KEY") or secret_key) else 2)
    prov = {"gemini": "gemini", "openai-compatible": "openai"}.get(provider, "baseline")
    model = st.text_input("Model (blank = auto)", os.getenv("LLM_MODEL", ""))
    base_url = st.text_input("Base URL (openai-compatible only)", os.getenv("LLM_BASE_URL", "")) if prov == "openai" else None
    key = st.text_input("API key (blank = use .env / secrets)", type="password")
    n_stitched = st.slider("Cross-episode promos to request", 0, 8, 5)
    budget = st.slider("Time budget (minutes)", 2, 20, 10,
                       help="If the AI is slow or out of quota, the rest of the run uses rule-based picks instead of waiting.")
    st.divider()
    st.header("Editor feedback")
    fb = load_feedback(FEEDBACK)
    st.write(f"{sum(f['decision'] == 'approve' for f in fb)} approved · {sum(f['decision'] == 'reject' for f in fb)} rejected")
    st.caption("Saved decisions are fed into the next run's prompt, and promos an editor rejected before are penalised.")
    if FEEDBACK.exists():
        st.download_button("Download feedback.jsonl", FEEDBACK.read_bytes(), "feedback.jsonl")

# ------------------------------------------------------------------ input
tab_run, tab_open = st.tabs(["Run on a new show", "Open a saved run"])
with tab_run:
    up = st.file_uploader("Screenplay: a .zip of episode .txt files, or several .txt files",
                          type=["zip", "txt"], accept_multiple_files=True)
    local_dirs = sorted(str(p) for p in Path("data").glob("*") if p.is_dir()) if Path("data").exists() else []
    local = st.selectbox("…or a show folder on this machine", ["(none)"] + local_dirs)
    show_name = st.text_input("Show name", "")
    c_a, c_b = st.columns(2)
    ep_from = c_a.number_input("From episode", 1, 999, 1)
    ep_to = c_b.number_input("To episode", 1, 999, 999, help="Tip for a quick demo: episodes 1-10 run in ~1-2 minutes.")
    go = st.button("Find promos", type="primary")

with tab_open:
    runs = sorted(Path("outputs").glob("*/*/promos.json"), reverse=True) if Path("outputs").exists() else []
    pick = st.selectbox("Saved runs", ["(none)"] + [str(r) for r in runs])
    up_json = st.file_uploader("…or upload a promos.json", type=["json"], key="pj")
    if st.button("Open run"):
        if up_json:
            st.session_state.result = json.loads(up_json.read())
            st.session_state.run_dir = None
        elif pick != "(none)":
            st.session_state.result = json.loads(Path(pick).read_text(encoding="utf-8"))
            st.session_state.run_dir = str(Path(pick).parent)
        st.session_state.episodes = None


def _load_input():
    if up:
        if len(up) == 1 and up[0].name.lower().endswith(".zip"):
            return load_show(up[0].read(), up[0].name), Path(up[0].name).stem
        tmp = Path(tempfile.mkdtemp())
        for f in up:
            (tmp / f.name).write_bytes(f.read())
        return load_show(tmp), "Uploaded show"
    if local != "(none)":
        return load_show(local), Path(local).name
    return None, None


if go:
    episodes, default_name = _load_input()
    if not episodes:
        st.error("No episodes with timestamped beats found. Upload a .zip or .txt files.")
        st.stop()
    name = show_name or re.sub(r"[-_]scripts.*$", "", default_name)
    try:
        llm = LLM(provider=prov, model=model or None, api_key=key or secret_key or None, base_url=base_url or None)
    except LLMError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, "Starting")
    logbox = st.empty()
    logs = []

    def log(m):
        logs.append(m)
        logbox.code("\n".join(logs[-8:]))

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    out = Path("outputs") / slug / f"{datetime.now():%Y%m%d-%H%M%S}_{llm.provider}"
    finder = PromoFinder(llm, show_name=name, feedback_path=FEEDBACK, n_stitched=n_stitched, log=log,
                         time_budget_min=budget, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
    episodes = [e for e in episodes if ep_from <= e.number <= ep_to]
    if not episodes:
        st.error("No episodes in that range.")
        st.stop()
    with st.spinner(f"Running with {llm.describe()} on {len(episodes)} episodes…"):
        st.session_state.result = finder.run(episodes, out)
    st.session_state.run_dir = str(out)
    st.session_state.episodes = episodes

res = st.session_state.get("result")
if not res:
    st.info("Upload a screenplay and press **Find promos**, or open a saved run.")
    st.stop()

# ------------------------------------------------------------------ results
promos = res["promos"]
u = res.get("usage", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episodes", res["episodes"])
c2.metric("Promos kept", len(promos))
c3.metric("Dropped by validator", len(res.get("dropped", [])))
c4.metric("AI calls", u.get("calls_total", 0))
c5.metric("Tokens in / out", f"{u.get('input_tokens', 0) / 1000:.0f}k / {u.get('output_tokens', 0) / 1000:.0f}k")
st.caption(f"Models: `{', '.join(res.get('models_used') or [res['model']])}` · prompts {res.get('prompt_version')} · "
           f"wall time {u.get('wall_time_s')}s")
if res.get("degraded"):
    st.warning("**Parts of this run used the rule-based fallback because the AI was unavailable** "
               "(free-tier quota, overload or time budget). Those promos are marked *rules*.\n\n- "
               + "\n- ".join(res["degraded"]))

t_rank, t_plan, t_map, t_quality, t_dl = st.tabs(
    ["Ranked promos + review", "Week-1 posting plan", "Tension map", "Quality checks", "Downloads"])

reviewed = {(f.get("show"), f.get("promo_where")): f["decision"] for f in load_feedback(FEEDBACK)}

with t_rank:
    f1, f2, f3 = st.columns(3)
    kinds = f1.multiselect("Type", ["single", "stitched"], default=["single", "stitched"])
    ep_range = f2.slider("Episodes", 1, max(res["episodes"], 1), (1, max(res["episodes"], 1)))
    top_n = f3.number_input("Show top", 5, len(promos) or 5, min(20, len(promos) or 5))
    shown = [p for p in promos if p["kind"] in kinds and ep_range[0] <= min(p["episodes"]) <= ep_range[1]][: int(top_n)]
    for p in shown:
        c = p.get("cold") or {}
        state = reviewed.get((res["show"], p["where"]))
        badge = {"approve": "✅", "reject": "❌"}.get(state, "")
        src = " · rules" if p.get("source") == "rules" else ""
        with st.expander(f"{badge} #{p['rank']} · {p['id']}{src} · {p['score']:.3f} · {p['duration']:.0f}s · {p['where']} · {p['hook'][:80]}",
                         expanded=p["rank"] <= 3):
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**Hook:** {p['hook']}  \n**Cliffhanger:** {p['cliffhanger']}  \n**Why it works:** {p['why']}  \n"
                            f"**Trope:** {p['trope']} · **Spoiler level:** {p['spoiler_level']}/2")
                st.dataframe(pd.DataFrame([{"#": i + 1, "Episode": s["episode"], "Start": s["start_ts"], "End": s["end_ts"],
                                            "Length (s)": s["duration"]} for i, s in enumerate(p["segments"])]),
                             hide_index=True, use_container_width=True)
                if p["flags"]:
                    st.warning("Validator: " + ", ".join(p["flags"]))
            with b:
                st.markdown("**Cold-viewer test** (AI with zero show context)")
                st.markdown(f"Stops scroll: **{c.get('first_3s_stop', '–')}/10** · Needs next: **{c.get('need_next', '–')}/10** · "
                            f"Would tap: **{c.get('would_tap', '–')}**")
                st.caption(f"\"{c.get('understood', '')}\"")
                if c.get("confusing"):
                    st.caption("Confused by: " + "; ".join(map(str, c["confusing"])))
                st.caption("Score parts: " + ", ".join(f"{k} {v}" for k, v in p["scores"].items()))
            with st.popover("Read the promo script"):
                st.code(p.get("script", ""), language=None)
            r1, r2, r3, r4 = st.columns([1, 2, 3, 1])
            dec = r1.radio("Decision", ["approve", "reject"], key=f"d_{p['id']}", horizontal=True, label_visibility="collapsed")
            reason = r2.selectbox("Reason", ["good promo"] + REJECT_REASONS if dec == "approve" else REJECT_REASONS,
                                  key=f"r_{p['id']}", label_visibility="collapsed")
            note = r3.text_input("Note", key=f"n_{p['id']}", placeholder="optional note", label_visibility="collapsed")
            if r4.button("Save", key=f"s_{p['id']}"):
                FEEDBACK.parent.mkdir(exist_ok=True)
                with FEEDBACK.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "show": res["show"],
                                         "promo_id": p["id"], "promo_where": p["where"], "decision": dec,
                                         "reason": reason, "note": note, "hook": p["hook"],
                                         "segments": [{"episode": s["episode"], "start": s["start"], "end": s["end"]}
                                                      for s in p["segments"]]}) + "\n")
                st.toast(f"Saved {dec} for {p['id']}")

with t_plan:
    st.markdown("Order for the first week. Day 1 is the show's premise (people who tap start at the beginning); "
                "no episode or trope repeats back to back; at most one big-spoiler promo.")
    st.dataframe(pd.DataFrame(res["posting_plan"]), hide_index=True, use_container_width=True)

with t_map:
    eps = st.session_state.get("episodes")
    if not eps and st.session_state.get("run_dir"):
        st.info("Tension map needs the screenplay. Run the show (or select its folder) to see it.")
    if eps:
        en = st.selectbox("Episode", [e.number for e in eps])
        ep = next(e for e in eps if e.number == en)
        tc = tension_curve(ep)
        svg_chart([t for t, _ in tc], [v for _, v in tc], "line", x_label="s", y_label="tension")
        hits = [p for p in promos if en in p["episodes"]]
        st.caption(f"Logline: {ep.logline}")
        for p in hits:
            for s in p["segments"]:
                if s["episode"] == en:
                    st.write(f"#{p['rank']} {p['id']}: {s['start_ts']} → {s['end_ts']} · {p['hook'][:90]}")

with t_quality:
    df = pd.DataFrame([{"rank": p["rank"], "id": p["id"], "episode": min(p["episodes"]), "kind": p["kind"],
                        "score": p["score"], "stop": (p.get("cold") or {}).get("first_3s_stop"),
                        "need_next": (p.get("cold") or {}).get("need_next"),
                        "would_tap": (p.get("cold") or {}).get("would_tap"),
                        "flags": ", ".join(p["flags"]), "duration": p["duration"]} for p in promos])
    q1, q2, q3 = st.columns(3)
    q1.metric("Promos with a validator flag", f"{(df['flags'] != '').mean():.0%}" if len(df) else "–")
    q2.metric("Cold viewer would tap", f"{(df['would_tap'] == True).mean():.0%}" if len(df) else "–")  # noqa: E712
    q3.metric("Median length", f"{df['duration'].median():.0f}s" if len(df) else "–")
    st.markdown("**Promos per episode** (are we over-picking some episodes?)")
    per_ep = df.groupby("episode").size() if len(df) else pd.Series(dtype=int)
    svg_chart(list(per_ep.index), list(per_ep.values), "bar", height=140, x_label="episode", y_label="promos")
    st.markdown("**Dropped by the validator**")
    st.dataframe(pd.DataFrame(res.get("dropped", [])), use_container_width=True)
    st.markdown("**All promos**")
    st.dataframe(df, hide_index=True, use_container_width=True)

with t_dl:
    st.download_button("promos.json", json.dumps(res, indent=2), "promos.json")
    rd = st.session_state.get("run_dir")
    if rd and Path(rd, "promos.csv").exists():
        st.download_button("promos.csv", Path(rd, "promos.csv").read_bytes(), "promos.csv")
        st.download_button("promos_report.md", Path(rd, "promos_report.md").read_bytes(), "promos_report.md")
        st.caption(f"Full run folder (incl. raw AI prompts/responses): `{rd}`")
