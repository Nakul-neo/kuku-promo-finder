"""Compare a run against hand-picked 'gold' promos (and optionally against another run).

gold.csv columns: episode,start,end,note   (times as mm:ss or seconds)

  python evaluate.py --run outputs/<show>/<run>/promos.json --gold eval/gold.csv
  python evaluate.py --run <ai run>/promos.json --gold eval/gold.csv --baseline <rules run>/promos.json

A gold promo counts as FOUND if a predicted promo has a segment in the same episode
that covers >= 50% of the gold clip (or the gold clip covers >= 50% of it).
HOOK MATCH = the predicted start is within 3 s of the gold start.
"""
import argparse
import csv
import json

from promo_finder.parse import to_seconds


def load_gold(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s, e = r["start"].strip(), r["end"].strip()
            rows.append({"episode": int(r["episode"]), "start": to_seconds(s) if ":" in s else float(s),
                         "end": to_seconds(e) if ":" in e else float(e), "note": r.get("note", "")})
    return rows


def match(gold, promo):
    best = (0.0, False)
    for s in promo["segments"]:
        if s["episode"] != gold["episode"]:
            continue
        inter = max(0.0, min(s["end"], gold["end"]) - max(s["start"], gold["start"]))
        cover = max(inter / max(gold["end"] - gold["start"], 1), inter / max(s["end"] - s["start"], 1))
        hook = abs(s["start"] - gold["start"]) <= 3.0
        best = max(best, (cover, hook))
    return best


def evaluate(promos, gold, ks=(5, 10, 20)):
    out = {}
    for k in list(ks) + [len(promos)]:
        top = promos[:k]
        found = [g for g in gold if any(match(g, p)[0] >= 0.5 for p in top)]
        hooks = [g for g in gold if any(match(g, p)[0] >= 0.5 and match(g, p)[1] for p in top)]
        prec = sum(1 for p in top if any(match(g, p)[0] >= 0.5 for g in gold)) / max(len(top), 1)
        out[f"@{k}"] = {"recall": round(len(found) / max(len(gold), 1), 2),
                        "hook_match": round(len(hooks) / max(len(gold), 1), 2),
                        "precision_vs_gold": round(prec, 2)}
    misses = [g for g in gold if not any(match(g, p)[0] >= 0.5 for p in promos)]
    return out, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    gold = load_gold(a.gold)
    for label, path in [("run", a.run), ("baseline", a.baseline)]:
        if not path:
            continue
        res = json.load(open(path, encoding="utf-8"))
        metrics, misses = evaluate(res["promos"], gold)
        print(f"\n== {label}: {res['model']} ({len(res['promos'])} promos) vs {len(gold)} gold promos")
        for k, v in metrics.items():
            print(f"  {k:>5}: recall {v['recall']:.0%} · hook match {v['hook_match']:.0%} · top-k precision vs gold {v['precision_vs_gold']:.0%}")
        for g in misses:
            print(f"  MISSED Ep{g['episode']} {g['start']:.1f}-{g['end']:.1f}s  {g['note']}")


if __name__ == "__main__":
    main()
