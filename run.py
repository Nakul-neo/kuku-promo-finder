"""Find promos in a microdrama screenplay.

Usage:
  python run.py --input "data/My Show"            # folder of episode .txt files
  python run.py --input show.zip --show "My Show"
  LLM_PROVIDER=baseline python run.py --input ... # rules only, no API key needed
  LLM_MODEL=gemini-2.5-pro python run.py ...      # swap model, nothing else changes
"""
import argparse
import sys
import re
from datetime import datetime
from pathlib import Path

from promo_finder.llm import LLM, load_dotenv
from promo_finder.pipeline import PromoFinder


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .txt episodes, a .zip, or one .txt")
    ap.add_argument("--show", default=None, help="show name (defaults to folder/zip name)")
    ap.add_argument("--out", default="outputs", help="output root")
    ap.add_argument("--provider", default=None, help="gemini | openai | baseline")
    ap.add_argument("--model", default=None)
    ap.add_argument("--feedback", default="feedback/feedback.jsonl")
    ap.add_argument("--stitched", type=int, default=5, help="number of cross-episode promos to request")
    ap.add_argument("--episodes", default=None, help="range, e.g. 1-10 (default: all)")
    ap.add_argument("--budget-min", type=float, default=15, help="time budget; after it, rules fill in")
    args = ap.parse_args()

    load_dotenv()
    llm = LLM(provider=args.provider, model=args.model)
    show = args.show or re.sub(r"[-_]scripts.*$", "", Path(args.input).stem)
    slug = re.sub(r"[^a-z0-9]+", "-", show.lower()).strip("-")[:40]
    out = Path(args.out) / slug / f"{datetime.now():%Y%m%d-%H%M%S}_{llm.provider}"
    print(f"Show: {show}\nModel: {llm.describe()}\nOutput: {out}")
    finder = PromoFinder(llm, show_name=show, feedback_path=args.feedback, n_stitched=args.stitched,
                         time_budget_min=args.budget_min)
    rng = tuple(int(x) for x in args.episodes.split("-")) if args.episodes else None
    res = finder.run(args.input, out, episode_range=rng)
    print("\nTop 10:")
    for p in res["promos"][:10]:
        print(f"  #{p['rank']:>2} {p['id']} {p['score']:.3f}  {p['where']}  | {p['hook'][:70]}")
    print(f"\nUsage: {res['usage']}")
    if res.get("degraded"):
        print("Ran partly without AI: " + "; ".join(res["degraded"]))


if __name__ == "__main__":
    main()
