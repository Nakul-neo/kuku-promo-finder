# Montage: microdrama screenplay → ranked 20–60 s promos

*Finds the scroll-stopping moments in every episode.*

Reads a timestamped microdrama screenplay (one `.txt` per episode) and returns promo clips that
**open on a hook within 3 s**, **cut on a cliffhanger** and **make sense to a stranger**, each with
the episode, start and end timestamps of every segment, the hook, the cliffhanger, why it works, and a rank.

## How it works

```
screenplay .txt ─► 0 PARSE (code)        every line/action/shot becomes a numbered beat, e.g. E03-047
                ─► 1 PRE-FILTER (code)   rule-based tension score → ~4 hot windows per episode (free, 0.5 s/show)
                ─► 2 PROPOSE (AI)        reads only the hot windows; picks promos by BEAT ID, never timestamps
                ─► 3 STITCH (AI)         joins 2–3 moments across episodes into arc promos
                ─► 4 VALIDATE (code)     timestamps from the script; 20–60 s; hook in first 3 s;
                                         drops invented ids; flags resolved endings; dedupes
                ─► 5 COLD VIEWER (AI)    a separate call with ZERO show context "watches" each promo:
                                         stop-scroll 1–10, need-next 1–10, would tap?, what was confusing
                ─► 6 RANK (code)         transparent weighted score + funnel logic + week-1 posting plan
                ─► 7 EDITOR REVIEW (UI)  approve / reject with a reason → fed into the next run
```

Design choices:
- **AI never writes timestamps.** It refers to beat ids; code looks the times up. Made-up timestamps are impossible, and invented ids are caught and dropped.
- **Rules first, AI second.** The pre-filter cuts what the AI reads, and gives an AI-independent tension score used in ranking.
- **The judge is not the picker.** The cold-viewer test runs in a fresh call with no logline, no show title, no episode numbers. It can use a different model (`COLD_VIEWER_MODEL`).
- **Funnel-aware ranking.** Promos from early episodes rank higher (a viewer who taps lands near the moment they saw), big spoilers are penalised, and the posting plan opens with the premise and never repeats an episode or trope back to back.
- **Swap the model with env vars only.** `LLM_PROVIDER` = `gemini` | `openai` (any OpenAI-compatible API: Groq, OpenRouter, an internal gateway) | `baseline` (rules only, no key).

## Reliability (free-tier aware, always finishes)

- **~6 AI calls per show** (2 × 25-episode propose calls, 1 stitch call, ~3 cold-viewer calls), not one per episode.
- **Low "thinking" mode** by default (`LLM_THINKING=low`); if a model rejects the setting, it is dropped automatically.
- **Model fallback chain** (`LLM_FALLBACK_MODELS`): a model that is out of daily quota, overloaded, retired or timing out is swapped for the next one.
- **No long waits**: never sleeps more than 60 s; a run has a time budget (`--budget-min`, default 15); a circuit breaker stops calling the AI after 2 calls in a row fail completely.
- **Graceful fallback**: any step the AI could not do uses the rule-based picks / estimates instead, clearly labelled `rules` and listed under `degraded` in the output.
- **Saved answers**: every AI answer is cached by prompt, so re-running never pays twice.

Tested against simulated failures (daily quota, 503 overload, rejected thinking setting, timeouts, budget exhaustion): every scenario finishes with valid 20-60 s promos.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY (free at aistudio.google.com)

# command line
python run.py --input "path/to/Show folder"           # or a .zip of episode .txt files
LLM_PROVIDER=baseline python run.py --input ...        # rules only, no key
python run.py --input ... --episodes 1-10              # quick run on a range
LLM_MODEL=gemini-2.5-pro python run.py --input ...     # stronger model, nothing else changes

# app (upload a new show, review promos)
streamlit run app.py

# quality check against hand-picked promos
python evaluate.py --run outputs/<show>/<run>/promos.json --gold eval/gold.csv --baseline outputs/<show>/<rules run>/promos.json
```

Windows PowerShell: set variables with `$env:LLM_PROVIDER="baseline"` before `python run.py ...`.

## Outputs (per run, in `outputs/<show>/<timestamp>_<provider>/`)

| File | What |
|---|---|
| `promos_report.md` | Human-readable: week-1 posting plan, top 25 promos with segments, hook, cliffhanger, why, cold-viewer verdict |
| `promos.csv` / `promos.json` | Every promo kept, ranked, with score breakdown and validator flags |
| `prefilter_candidates.json` | Hot windows from the rule-based scorer |
| `beats.csv` | Parsed screenplay |
| `raw_llm/*.json` | Unedited prompt + raw response for every AI call |

## Confidentiality
The input pack is Kuku's content. `data/` and `*.zip` are git-ignored and never committed;
upload the screenplay in the app at run time.
