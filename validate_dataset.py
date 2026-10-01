import json, re, csv, collections, sys, pathlib

JSONL = sys.argv[1] if len(sys.argv) > 1 else "instruction_dataset.jsonl"
CORPUS = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "domain_corpus")
AUDIT = sys.argv[3] if len(sys.argv) > 3 else "dataset_audit.csv"

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (" -> " + detail if detail else ""))
    ok &= bool(cond)

n = len(rows)
check("40-50+ pairs", n >= 40, f"{n} pairs")
check("only instruction/response keys", all(set(r) == {"instruction", "response"} for r in rows))
check("no duplicate instructions", len({r['instruction'] for r in rows}) == n)

wc = [len(r["response"].split()) for r in rows]
check("every response >= 30 words", min(wc) >= 30, f"min={min(wc)} avg={sum(wc)/n:.1f} max={max(wc)}")

sent = [len(re.findall(r"[.!?](?:\s|$)", r["response"])) for r in rows]
check("every response <= 3 sentences", max(sent) <= 3, f"max sentences={max(sent)}")

check("no 'According to' prefix / PAGE markers / emails",
      not any(re.search(r"^According to|--- PAGE|@\w+\.\w+", r["response"]) for r in rows))

# template check: first 3 words of each instruction
starts = collections.Counter(" ".join(r["instruction"].split()[:3]).lower() for r in rows)
top, cnt = starts.most_common(1)[0]
check("no instruction template > 20%", cnt / n <= 0.20, f"top opener '{top}' = {cnt}/{n} = {cnt/n:.1%}")

# verbatim-lift check: longest run of consecutive words shared with any corpus file
def norm(s): return re.findall(r"[a-z0-9_\-\./<>@]+", s.lower())
corp = {p.name: norm(p.read_text(errors='ignore')) for p in CORPUS.glob("*.txt")}
idx = {}
for name, toks in corp.items():
    s = set()
    for i in range(len(toks) - 7):
        s.add(tuple(toks[i:i+8]))
    idx[name] = s
worst = []
for r in rows:
    t = norm(r["response"])
    grams = [tuple(t[i:i+8]) for i in range(len(t) - 7)]
    hit = sum(any(g in s for s in idx.values()) for g in grams)
    worst.append(hit / max(1, len(grams)))
lifted = [w for w in worst if w > 0.5]
check("no response is mostly lifted (>50% 8-grams found verbatim in corpus)",
      len(lifted) == 0, f"max overlap={max(worst):.0%}, avg={sum(worst)/n:.0%}")

# source + type spread
src = collections.Counter(); typ = collections.Counter()
for row in csv.DictReader(open(AUDIT, encoding="utf-8")):
    src[row["source"]] += 1; typ[row["qtype"]] += 1
print("\nPer-source:", dict(src)); print("Per-type  :", dict(typ))
check("all 5 source documents covered", len(src) == 5)
check("factual + procedural + comparative all present", {"factual", "procedural", "comparative"} <= set(typ))
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
