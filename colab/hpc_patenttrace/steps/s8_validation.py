"""Section 8: technical validation of the summary layers.

Run from the pipeline notebook with run_step("s8_validation"); the parts below are executed in the namespace of the notebook.
"""

# ── 8.1  How much of each section the token budgets cover (CPU only) ─────────
# The facet budgets (background 1,500, summary 1,500, detailed description 2,000 tokens) are a design choice;
# this cell reports how much of the source text they cover on 1,000 records drawn at random. It makes no claim
# that the budgets are optimal, which would need the facets to be generated again with other budgets.
from transformers import AutoTokenizer as _ATP
_tok_prof = _ATP.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True) \
            if globals().get("USE_LLAMA_TOK_FOR_PROFILE") else \
            _ATP.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)

_SRC = m1 if "m1" in globals() else valid_df
_SAMPLE = _SRC.sample(min(1000, len(_SRC)), random_state=RANDOM_STATE)
_SUMC = "summary_filled" if "summary_filled" in _SAMPLE.columns else "summary"
BUDGETS = [("background",           "background",           1500),
           ("summary",              _SUMC,                  1500),
           ("detailed_description", "detailed_description", 2000)]

rows_p = []
for label, col, budget in BUDGETS:
    if col not in _SAMPLE.columns:
        print(f"skipped {label} (column {col} does not exist)"); continue
    L = np.array([len(_tok_prof.encode(str(t)[:400000], add_special_tokens=False))
                  for t in tqdm(_SAMPLE[col], desc=f"tokenize {label}")])
    rows_p.append({"section": label, "budget_tokens": budget,
                   "median_tokens": int(np.median(L)), "mean_tokens": round(float(L.mean()), 1),
                   "p90_tokens": int(np.percentile(L, 90)), "max_tokens": int(L.max()),
                   "pct_truncated": round(float((L > budget).mean() * 100), 1),
                   "mean_coverage_pct": round(float(np.minimum(L, budget).sum() / max(L.sum(), 1) * 100), 1)})

prof = pd.DataFrame(rows_p)
prof.to_csv(os.path.join(SAVE_DIR, "truncation_profile.csv"), index=False)
print(f"\n  {'section':24}{'budget':>7}{'median':>10}{'share cut':>12}{'coverage':>12}")
for r in rows_p:
    print(f"  {r['section']:24}{r['budget_tokens']:>7}{r['median_tokens']:>10}"
          f"{r['pct_truncated']:>11.1f}%{r['mean_coverage_pct']:>11.1f}%")
print(f"\n(sample of {len(_SAMPLE)} records)")
print("saved -> truncation_profile.csv")


# ── 8.2  Overlap of the facet terms with the full-text terms (Jaccard) ──────
jaccard_all = []
for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Jaccard"):
    s1 = stage1_df[stage1_df["patent_id"] == row["patent_id"]]
    if len(s1) == 0:
        continue
    s1 = s1.iloc[0]
    mk_s1  = extract_tech_terms(row["title"], s1["bg_summary"],
                                s1["inv_summary"], s1["tech_summary"])
    mk_raw = extract_tech_terms(row["title"], str(row["background"]),
                                str(row["summary"]), str(row["detailed_description"]))
    t_s1  = set(t.strip().lower() for t in mk_s1.split(",")  if t.strip())
    t_raw = set(t.strip().lower() for t in mk_raw.split(",") if t.strip())
    if t_s1 | t_raw:
        jaccard_all.append(len(t_s1 & t_raw) / len(t_s1 | t_raw))

jac = np.array(jaccard_all)
print(f"\nn = {len(jac)}")
print(f"mean Jaccard   : {jac.mean()*100:.1f}%")
print(f"median Jaccard : {np.median(jac)*100:.1f}%")
print(f"full overlap   : {(jac == 1.0).sum()} | zero overlap: {(jac == 0.0).sum()}")


# ── 8.3  Retention of the must_keep terms in two summary layers ─────────────
def calc_retention_s1(df, summary_col):
    """Retention of the terms extracted from the facets."""
    hit, total = 0, 0
    for _, row in df.iterrows():
        s1 = stage1_df[stage1_df["patent_id"] == row["patent_id"]]
        if len(s1) == 0:
            continue
        s1 = s1.iloc[0]
        mk = extract_tech_terms(row["title"], s1["bg_summary"],
                                s1["inv_summary"], s1["tech_summary"])
        terms = [t.strip().lower() for t in mk.split(",") if t.strip()]
        hit   += sum(1 for t in terms if t in str(row[summary_col]).lower())
        total += len(terms)
    return hit / total * 100 if total else 0

def calc_retention_raw(df, summary_col):
    """Retention computed with the "terms extracted from the original full text" (a stricter reference)."""
    hit, total = 0, 0
    for _, row in df.iterrows():
        mk = extract_tech_terms(row["title"], str(row["background"]),
                                str(row["summary"]), str(row["detailed_description"]))
        terms = [t.strip().lower() for t in mk.split(",") if t.strip()]
        hit   += sum(1 for t in terms if t in str(row[summary_col]).lower())
        total += len(terms)
    return hit / total * 100 if total else 0

SYSTEMS = [("direct-brief", "baseline_summary"), ("synthesized", "final_summary")]
with timed("8.3  Term retention (summary layers)", n_items=2 * len(valid_df)):
    ret_s1  = {n: calc_retention_s1 (valid_df, c) for n, c in SYSTEMS}
    ret_raw = {n: calc_retention_raw(valid_df, c) for n, c in SYSTEMS}
lens = {n: valid_df[c].str.split().str.len() for n, c in SYSTEMS}

hdr = "".join(f"{n:>13}" for n, _ in SYSTEMS)
print(f"{'':34}{hdr}")
print(f"{'retention (facet terms)':34}" +
      "".join(f"{ret_s1[n]:>12.1f}%" for n, _ in SYSTEMS))
print(f"{'retention (full-text terms)':34}" +
      "".join(f"{ret_raw[n]:>12.1f}%" for n, _ in SYSTEMS))
print(f"{'mean length (words)':34}" +
      "".join(f"{lens[n].mean():>13.0f}" for n, _ in SYSTEMS))


# ── 8.4  Compression ratio, ROUGE against the abstract, and a pairing check ───
# ROUGE with the abstract as the reference measures how similar a summary is to the abstract. The layers are
# meant to carry content that the abstract does not have, so ROUGE is not evidence of correctness here; its
# values are reported as a property of the layers. ROUGE does serve one check: a summary attached to the
# wrong record, or a degenerate generation, scores no better against its own abstract than against the
# abstract of another record. Each summary is therefore scored against its own abstract and against the
# abstract shifted by one position; records that fail are listed in pairing_integrity_flags.csv.
from rouge_score import rouge_scorer
import numpy as np, pandas as pd, os

_SUM_COL = "summary_filled" if "summary_filled" in valid_df.columns else "summary"
src_words = (valid_df["background"].astype(str) + " " +
             valid_df[_SUM_COL].astype(str) + " " +
             valid_df["detailed_description"].astype(str)).str.split().str.len()

Q_SYSTEMS = [(n, c) for n, c in (("direct-brief", "baseline_summary"),
                                 ("direct-full", "baseline_matched_summary"),
                                 ("synthesized",       "final_summary"))
             if c in valid_df.columns and valid_df[c].astype(str).str.len().gt(0).any()]
print("Systems included:", [n for n, _ in Q_SYSTEMS])

LOW_R1 = 0.10          # anomaly threshold declared in advance
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
abstracts = valid_df["abstract"].astype(str).tolist()
shuffled  = abstracts[1:] + abstracts[:1]        # shift by one position: a deterministic, reproducible control permutation

rows_q, flagged = [], []
for name, col in Q_SYSTEMS:
    hyps = valid_df[col].astype(str).tolist()
    sw   = valid_df[col].astype(str).str.split().str.len()
    comp = sw / src_words.clip(lower=1)
    with timed(f"8.4  ROUGE with shifted control ({name})", n_items=len(valid_df)):
        own1, own2, ownL, shf1 = [], [], [], []
        for i, h in enumerate(hyps):
            s = scorer.score(abstracts[i], h)
            own1.append(s["rouge1"].fmeasure); own2.append(s["rouge2"].fmeasure)
            ownL.append(s["rougeL"].fmeasure)
            shf1.append(scorer.score(shuffled[i], h)["rouge1"].fmeasure)
    own1, shf1 = np.array(own1), np.array(shf1)
    n_low  = int((own1 < LOW_R1).sum())
    n_fail = int((own1 <= shf1).sum())          # not better against its own abstract than against a random one
    print(f"  {name:10} compression {comp.mean()*100:5.2f}% | mean summary {sw.mean():5.0f} words | "
          f"ROUGE-1 {own1.mean():.4f} (shifted control {shf1.mean():.4f}) | "
          f"R-2 {np.mean(own2):.4f} | R-L {np.mean(ownL):.4f}")
    print(f"             pairing integrity: ROUGE-1 < {LOW_R1} in {n_low} records"
          f" ({n_low/len(own1)*100:.2f}%) | not better than the shifted control: {n_fail} records"
          f" ({n_fail/len(own1)*100:.2f}%)")
    rows_q.append({"system": name, "compression_ratio": round(float(comp.mean()), 4),
                   "mean_summary_words": round(float(sw.mean()), 1),
                   "rouge1_f": round(float(own1.mean()), 4),
                   "rouge2_f": round(float(np.mean(own2)), 4),
                   "rougeL_f": round(float(np.mean(ownL)), 4),
                   "rouge1_permuted": round(float(shf1.mean()), 4),
                   "rouge1_lift": round(float(own1.mean() - shf1.mean()), 4),
                   "n_low_rouge1": n_low, "n_not_better_than_permuted": n_fail})
    _bad = valid_df.loc[(own1 < LOW_R1) | (own1 <= shf1), "patent_id"]
    flagged.extend({"system": name, "patent_id": p,
                    "rouge1_own": round(float(own1[j]), 4),
                    "rouge1_permuted": round(float(shf1[j]), 4)}
                   for j, p in zip(np.where((own1 < LOW_R1) | (own1 <= shf1))[0], _bad))

pd.DataFrame(rows_q).to_csv(os.path.join(SAVE_DIR, "summary_metrics.csv"), index=False)
pd.DataFrame(flagged).to_csv(os.path.join(SAVE_DIR, "pairing_integrity_flags.csv"), index=False)
print(f"\nThe lift (own − shifted control) is the evidence of correct pairing; {len(flagged)} records are flagged for inspection")
print("saved -> summary_metrics.csv, pairing_integrity_flags.csv")


# ── 8.5  Content of the summaries beyond the abstract (CPU only) ─────────────
# Are the summaries only paraphrases of the abstract?
#   candidate = a long technical word that occurs in the specification but not in the title or abstract
#   hit       = a candidate that occurs in the summary
# The abstract itself scores 0 by this definition. The check needs no model and can be repeated in minutes.
import numpy as np, pandas as pd, os, re
for _n in ("HPC_WHITELIST", "TITLE_STOPWORDS", "GENERIC_SUFFIXES"):
    if _n not in globals():
        raise RuntimeError("Run 5.3 first (definition of the term extraction).")

_LONG_AV = re.compile(r"\b[a-zA-Z][a-zA-Z-]{11,}\b")
def _av_terms(text):
    out = set()
    for w in _LONG_AV.findall(str(text)):
        lw = w.lower()
        if lw in HPC_WHITELIST or (lw not in TITLE_STOPWORDS and not GENERIC_SUFFIXES.search(lw)):
            out.add(lw)
    return out

_SUMC = "summary_filled" if "summary_filled" in valid_df.columns else "summary"
AV_SYSTEMS = [(n, c) for n, c in (("direct-brief", "baseline_summary"),
                                  ("direct-full", "baseline_matched_summary"),
                                  ("synthesized",       "final_summary"))
              if c in valid_df.columns and valid_df[c].astype(str).str.len().gt(0).any()]
print("Systems compared:", [n for n, _ in AV_SYSTEMS])

rec = []
with timed("8.5  Content beyond the abstract", n_items=len(valid_df)):
    for _, r in tqdm(valid_df.iterrows(), total=len(valid_df), desc="added-value"):
        abs_terms  = _av_terms(f'{r["title"]} {r["abstract"]}')
        spec_terms = _av_terms(f'{r["background"]} {r[_SUMC]} '
                               f'{str(r["detailed_description"])[:60000]}')
        cand = spec_terms - abs_terms
        if not cand:
            continue
        row = {"patent_id": r["patent_id"], "n_candidate": len(cand)}
        for name, col in AV_SYSTEMS:
            s_ = str(r[col]).lower()
            n_hit = sum(1 for t in cand if t in s_)
            row[f"{name}_added"] = n_hit
            row[f"{name}_rate"]  = n_hit / len(cand)
        rec.append(row)

add_df = pd.DataFrame(rec)
add_df.to_csv(os.path.join(SAVE_DIR, "added_value_beyond_abstract.csv"), index=False)
print(f"\nrecords that can be evaluated {len(add_df):,} / {len(valid_df):,}"
      f" | mean candidates per record {add_df['n_candidate'].mean():.1f}")
print(f"  {'system':16}{'added terms':>14}{'coverage':>10}{'at least one':>14}")
for name, _ in AV_SYSTEMS:
    a = add_df[f"{name}_added"]
    print(f"  {name:16}{a.mean():>14.2f}{add_df[f'{name}_rate'].mean()*100:>9.1f}%"
          f"{(a >= 1).mean()*100:>13.1f}%")
print(f"  {'abstract itself':16}{0.00:>14.2f}{0.0:>9.1f}%{0.0:>13.1f}%   ← doing nothing")

# Paired bootstrap: report first the budget-matched pair (synthesized vs direct-full); otherwise report synthesized vs direct-brief and note the mixture
_hi = "synthesized"
_lo = "direct-full" if any(n == "direct-full" for n, _ in AV_SYSTEMS) else "direct-brief"
_note = ("input budgets matched; the difference can be attributed to the process structure"
         if _lo == "direct-full" else
         "input budgets not matched (direct-brief reads about 2,000 tokens, synthesized about 5,000); the difference mixes the two effects")
rng = np.random.RandomState(RANDOM_STATE)
d  = (add_df[f"{_hi}_added"] - add_df[f"{_lo}_added"]).values
bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(1000)])
print(f"\n{_hi} − {_lo}: {d.mean():+.2f} words per record"
      f" | 95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
print(f"({_note})")
print("saved -> added_value_beyond_abstract.csv")


# ── 8.6  Coverage beyond the input window: does deep content reach the summary? ──
# Visible range of the three layers (tokens of the synthesis model):
#   direct-brief  background[0,800)   summary[0,400)   detailed[0,800)    about 2,000
#   direct-full   background[0,1500)  summary[0,1500)  detailed[0,2000)   about 5,000
#   synthesized   background[0,1500)  summary[0,1500)  detailed[0,2000)   about 5,000
# The two windows use different visible baselines, so that terms that the specification repeats in its
# visible part do not count:
#   deep zone [800,2000): minus the visible range of direct-brief. direct-brief cannot read it; the other two can.
#                    direct-full − direct-brief = reading-depth effect
#                    synthesized − direct-full       = process-structure effect (both read it; which one brings more of it into the output)
#                    The latter is a descriptive difference, not a hypothesis test.
#   control window [2000,4000): minus the union of the visible ranges of the three. No system reads it, so the survival rate must be near 0.
from transformers import AutoTokenizer as _AT
_tok_cov = _AT.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)

_LONG_COV = re.compile(r"\b[a-zA-Z][a-zA-Z-]{11,}\b")

def _cov_terms(segment):
    terms = set()
    for w in _LONG_COV.findall(segment):
        lw = w.lower()
        if lw in HPC_WHITELIST or (lw not in TITLE_STOPWORDS
                                   and not GENERIC_SUFFIXES.search(lw)):
            terms.add(lw)
    return terms

def _tok_slice(text, lo, hi, max_chars):
    toks = _tok_cov.encode(str(text)[:max_chars], add_special_tokens=False)
    if len(toks) <= lo:
        return None
    return _tok_cov.decode(toks[lo:hi], skip_special_tokens=True)

SYSTEMS = [(n, c) for n, c in (("direct-brief", "baseline_summary"),
                               ("direct-full", "baseline_matched_summary"),
                               ("synthesized",       "final_summary"))
           if c in valid_df.columns and valid_df[c].astype(str).str.len().gt(0).any()]
print("Systems compared:", [n for n, _ in SYSTEMS])

from collections import Counter as _Counter
cov, n_eval = {n: _Counter() for n, _ in SYSTEMS}, _Counter()
with timed("8.6  Coverage beyond the input window (summary layers)", n_items=len(valid_df)):
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="coverage"):
        sum_in = str(row.get("summary_filled", row.get("summary", "")))
        head   = f'{row["title"]} {row["abstract"]} '
        vis_short = (head
                     + (_tok_slice(row["background"], 0, 800, 12000) or "") + " "
                     + (_tok_slice(sum_in,            0, 400, 12000) or "") + " "
                     + (_tok_slice(row["detailed_description"], 0, 800, 12000) or "")).lower()
        vis_union = (head
                     + (_tok_slice(row["background"], 0, 1500, 20000) or "") + " "
                     + (_tok_slice(sum_in,            0, 1500, 20000) or "") + " "
                     + (_tok_slice(row["detailed_description"], 0, 2000, 30000) or "")).lower()
        for wkey, lo, hi, vis in (("deep", 800, 2000, vis_short),
                                  ("control", 2000, 4000, vis_union)):
            seg = _tok_slice(row["detailed_description"], lo, hi, 30000)
            if not seg:
                continue
            terms = {t for t in _cov_terms(seg) if t not in vis}
            if not terms:
                continue
            n_eval[wkey] += 1
            for name, col in SYSTEMS:
                s = str(row[col]).lower()
                cov[name][f"{wkey}_hit"]   += sum(1 for t in terms if t in s)
                cov[name][f"{wkey}_total"] += len(terms)

print(f"\nRecords that can be evaluated: deep zone [800,2000) = {n_eval['deep']}; control [2000,4000) = {n_eval['control']}")
print(f"{'system':12}{'deep zone':>12}{'control':>10}")
rows_c, rate = [], {}
for name, _ in SYSTEMS:
    c = cov[name]
    deep = c["deep_hit"] / c["deep_total"] * 100 if c["deep_total"] else float("nan")
    ctrl = c["control_hit"] / c["control_total"] * 100 if c["control_total"] else float("nan")
    rate[name] = deep
    print(f"{name:12}{deep:>11.2f}%{ctrl:>9.2f}%")
    rows_c.append({"system": name, "deep_survival_pct": round(deep, 2),
                   "control_pct": round(ctrl, 2),
                   "deep_n_terms": int(c["deep_total"]), "control_n_terms": int(c["control_total"])})
pd.DataFrame(rows_c).to_csv(os.path.join(SAVE_DIR, "coverage_beyond_truncation.csv"), index=False)

# ── Decomposition of the two effects (only valid with direct-full) ─────────────
if {"direct-brief", "direct-full", "synthesized"} <= set(rate):
    print(f"\nreading-depth effect      direct-full − direct-brief = {rate['direct-full'] - rate['direct-brief']:+.2f} percentage points")
    print(f"process-structure effect  synthesized       − direct-full = {rate['synthesized'] - rate['direct-full']:+.2f} percentage points"
          f"   ← descriptive difference, not a hypothesis test")
else:
    print("\n(5.11 was not run, so the two effects cannot be separated: the present difference contains both reading depth and process structure)")
print("saved -> coverage_beyond_truncation.csv")


# ── 8.7  Where the terms that only one layer contains come from (CPU only) ────
# Is the non-overlapping content of the synthesized and the direct-full layer a structural difference or
# generation noise? Both layers use greedy decoding (5.2), so the same input and prompt give the same output;
# a difference can only come from the three recorded design differences: input span, prompt path (three
# section prompts and two calls against one prompt) and the term list, which only the synthesized layer gets.
# Four descriptive predictions follow from these differences:
#   Prediction 1  terms only in synthesized lean toward the deep zone of the detailed description (after token 800)
#   Prediction 2  consensus terms lean toward the front part (title/abstract/background/first 800 tokens of the detailed description)
#   Prediction 3  terms only in synthesized fall in the domain whitelist more often than terms only in the control (the effect of the term guidance)
#   Prediction 4  if it is only noise, none of the three above shows a difference
#   Additional output: the five zones cover only the first 8,000 tokens of the detailed description and do not include the claims; a term that the zones do not find is compared once more against the "full lower-case text"
#   (the six text fields, including the claims): a term still found in the full text counts as elsewhere_in_source; only a term really not in the full text is not_in_source.
#   not_in_source is an upper bound of fabrication (paraphrases and inflections also fall here), not an error rate.
for _fn in ("_tok_cov", "_cov_terms", "_tok_slice"):
    if _fn not in globals():
        raise RuntimeError("Run 8.6 first (this cell uses its tokenizer and term helpers).")

_PAIR_HI = "final_summary"
_PAIR_LO = ("baseline_matched_summary"
            if "baseline_matched_summary" in valid_df.columns
            and valid_df["baseline_matched_summary"].astype(str).str.len().gt(0).any()
            else "baseline_summary")
_LO_NAME = "direct-full" if _PAIR_LO == "baseline_matched_summary" else "direct-brief"
print(f"Pair: synthesized ({_PAIR_HI}) vs {_LO_NAME} ({_PAIR_LO})")
if _LO_NAME == "direct-brief":
    print("Note: 5.11 was not run; the result contains both effects, reading depth and process structure")

BUCKETS = ["front", "background", "dd_0_800", "dd_800_2000", "dd_2000_plus", "elsewhere_in_source", "not_in_source"]
_nis_rows = []   # per record: share of the synthesized-only terms that are not in the full text (edge cases of 8.8)
GROUPS  = ["consensus", "synthesized_only", f"{_LO_NAME}_only"]
from collections import Counter as _C
tally = {g: _C() for g in GROUPS}
wl    = {g: _C() for g in GROUPS}
_SUMC = "summary_filled" if "summary_filled" in valid_df.columns else "summary"

with timed("8.7  Structure of the non-overlapping content", n_items=len(valid_df)):
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="overlap"):
        sum_in = str(row.get(_SUMC, ""))
        zones = [
            ("front",        f'{row["title"]} {row["abstract"]}'.lower()),
            ("background",  ((_tok_slice(row["background"], 0, 1500, 20000) or "") + " " +
                             (_tok_slice(sum_in,            0, 1500, 20000) or "")).lower()),
            ("dd_0_800",     (_tok_slice(row["detailed_description"], 0, 800, 12000) or "").lower()),
            ("dd_800_2000",  (_tok_slice(row["detailed_description"], 800, 2000, 30000) or "").lower()),
            ("dd_2000_plus", (_tok_slice(row["detailed_description"], 2000, 8000, 60000) or "").lower()),
        ]
        _full = (f'{row["title"]} {row["abstract"]} {row["background"]} {sum_in} '
                 f'{row["detailed_description"]} {row.get("claims_text", "")}').lower()
        _row_n = _row_nis = 0
        t_hi = _cov_terms(str(row[_PAIR_HI]))
        t_lo = _cov_terms(str(row[_PAIR_LO]))
        for g, terms in (("consensus", t_hi & t_lo), ("synthesized_only", t_hi - t_lo),
                         (f"{_LO_NAME}_only", t_lo - t_hi)):
            for t in terms:
                b = next((z for z, txt in zones if t in txt),
                         "elsewhere_in_source" if t in _full else "not_in_source")
                tally[g][b] += 1
                wl[g]["hit" if t in HPC_WHITELIST else "miss"] += 1
                if g == "synthesized_only":
                    _row_n += 1; _row_nis += (b == "not_in_source")
        _nis_rows.append({"patent_id": row["patent_id"], "n_synth_only_terms": _row_n,
                          "n_not_in_full_text": _row_nis})

print(f"\nSource position distribution of each group (%)")
print(f"  {'group':18}" + "".join(f"{b:>15}" for b in BUCKETS) + f"{'n_terms':>10}")
rows_o = []
for g in GROUPS:
    tot = sum(tally[g].values()) or 1
    print(f"  {g:18}" + "".join(f"{tally[g][b]/tot*100:>14.1f}%" for b in BUCKETS) + f"{tot:>10,}")
    r = {"group": g, "n_terms": tot,
         "whitelist_pct": round(wl[g]["hit"] / max(wl[g]["hit"] + wl[g]["miss"], 1) * 100, 1)}
    r.update({b: round(tally[g][b] / tot * 100, 2) for b in BUCKETS})
    rows_o.append(r)
pd.DataFrame(rows_o).to_csv(os.path.join(SAVE_DIR, "overlap_structure.csv"), index=False)

_deep = {g: tally[g]["dd_800_2000"] + tally[g]["dd_2000_plus"] for g in GROUPS}
_tot  = {g: sum(tally[g].values()) or 1 for g in GROUPS}
_lo_g = f"{_LO_NAME}_only"
print(f"\nFour descriptive comparisons")
print(f"  1  synthesized-only terms in the deep zone : {_deep['synthesized_only']/_tot['synthesized_only']*100:.1f}%"
      f"  vs {_lo_g} {_deep[_lo_g]/_tot[_lo_g]*100:.1f}%"
      f"  → {'agrees' if _deep['synthesized_only']/_tot['synthesized_only'] > _deep[_lo_g]/_tot[_lo_g] else 'does not agree'}")
_front = {g: tally[g]["front"] + tally[g]["background"] + tally[g]["dd_0_800"] for g in GROUPS}
print(f"  2  consensus terms in the front part       : {_front['consensus']/_tot['consensus']*100:.1f}%"
      f"  vs synthesized only {_front['synthesized_only']/_tot['synthesized_only']*100:.1f}%"
      f"  → {'agrees' if _front['consensus']/_tot['consensus'] > _front['synthesized_only']/_tot['synthesized_only'] else 'does not agree'}")
print(f"  3  synthesized-only terms in the whitelist : {rows_o[1]['whitelist_pct']:.1f}%"
      f"  vs {_lo_g} {rows_o[2]['whitelist_pct']:.1f}%"
      f"  → {'agrees' if rows_o[1]['whitelist_pct'] > rows_o[2]['whitelist_pct'] else 'does not agree'}")
print(f"  4  if noise, 1–3 must all show no difference (for comparison; not counted separately)")
print(f"\nShare of long terms not in the full text (upper bound of fabrication; compared against the full text with the claims): "
      + " | ".join(f"{g} {tally[g]['not_in_source']/_tot[g]*100:.1f}%" for g in GROUPS))
print("  ← upper bound, not an error rate; the direct evidence is the human audit (8.8 sample)")
_nis_df = pd.DataFrame(_nis_rows)
_nis_df["not_in_source_share"] = _nis_df["n_not_in_full_text"] / _nis_df["n_synth_only_terms"].replace(0, np.nan)
_nis_df.to_csv(os.path.join(SAVE_DIR, "not_in_full_text_per_record.csv"), index=False)
valid_df = (valid_df.drop(columns=[c for c in ("not_in_source_share",) if c in valid_df.columns])
                    .merge(_nis_df[["patent_id", "not_in_source_share"]], on="patent_id", how="left"))
print("saved -> not_in_full_text_per_record.csv (upper bound per record, for the edge cases of 8.8)")
print("saved -> overlap_structure.csv")


# ── 8.8  Human-audit sample: 50 records stratified by class section and 12 edge cases (CPU only) ──
# The sampling rule, the seed and the definition of the edge cases are all in this cell, so the sample is
# reproducible. The annotation protocol is released with the data (validation/human_audit/). The sample of
# the released dataset was drawn on 2026-09-12 and annotated afterwards; when the three sample files exist
# (for example copied by 1.7 from the reused run), no new sample is drawn.
_AUDIT_FILES = [os.path.join(SAVE_DIR, f) for f in ("audit_primary_50.csv", "audit_edge_12.csv", "audit_sentences.csv")]
if all(os.path.exists(p) for p in _AUDIT_FILES):
    print("The audit sample exists; no new sample is drawn:", [os.path.basename(p) for p in _AUDIT_FILES])
else:
    import re as _re
    _src = valid_df.copy()
    _src["ipc_sec"] = _src["main_ipc_4"].astype(str).str[0]
    _src["ipc_sec"] = _src["ipc_sec"].where(_src["ipc_sec"].isin(["G", "H"]), "other")

    # Primary sample: stratified in proportion by CPC section (G, H, other), filled up to 50 in total
    _share = _src["ipc_sec"].value_counts(normalize=True)
    _n = {k: int(round(50 * v)) for k, v in _share.items()}
    _n[_share.idxmax()] += 50 - sum(_n.values())
    _prim = pd.concat([_src[_src["ipc_sec"] == k].sample(n=v, random_state=RANDOM_STATE)
                       for k, v in _n.items() if v > 0])
    _prim = _prim.assign(stratum="primary:" + _prim["ipc_sec"])

    # Edge cases: fixed rules; records already in the primary sample are excluded
    _edge = []
    def _take(mask, k, name):
        _e = _src[mask & ~_src["patent_id"].isin(_prim["patent_id"])
                  & ~_src["patent_id"].isin(pd.concat(_edge)["patent_id"] if _edge else [])].head(k)
        _edge.append(_e.assign(stratum=name))
    _take(_src["patent_id"].isin(["US11321283B2", "US11573920B2", "US11321264B2"]), 3, "edge:low_cosine_to_facets")
    _flag_p = os.path.join(SAVE_DIR, "pairing_integrity_flags.csv")
    if os.path.exists(_flag_p):
        _flag = pd.read_csv(_flag_p)
        _take(_src["patent_id"].isin(_flag.loc[_flag["system"] == "synthesized", "patent_id"]), 3, "edge:pairing_flag")
    _take(_src["detailed_description"].astype(str).str.split().str.len().rank(ascending=False) <= 2, 2, "edge:longest_description")
    _take(_src["background"].astype(str).str.len() < MIN_LEN, 2, "edge:short_background_substituted_what")
    if "not_in_source_share" in _src.columns:      # written back per record by 8.7
        _take(_src["not_in_source_share"].rank(ascending=False, method="first") <= 2, 2, "edge:highest_not_in_full_text")
    _edge = pd.concat(_edge).drop_duplicates("patent_id").head(12)

    _cols = ["patent_id", "stratum", "title", "abstract", "final_summary"]
    _prim_out = _prim[_cols].assign(patent_url="https://patents.google.com/patent/" + _prim["patent_id"].astype(str))
    _edge_out = _edge[_cols].assign(patent_url="https://patents.google.com/patent/" + _edge["patent_id"].astype(str))
    _prim_out.to_csv(os.path.join(SAVE_DIR, "audit_primary_50.csv"), index=False)
    _edge_out.to_csv(os.path.join(SAVE_DIR, "audit_edge_12.csv"), index=False)

    _sent = []
    for _, r in pd.concat([_prim_out, _edge_out]).iterrows():
        for j, sent in enumerate(_re.split(r"(?<=[.!?])\s+", str(r["final_summary"]).strip())):
            if sent:
                _sent.append({"patent_id": r["patent_id"], "stratum": r["stratum"], "sentence_id": j + 1,
                              "sentence": sent, "label": "", "evidence_span": "", "error_type": "", "annotator": ""})
    pd.DataFrame(_sent).to_csv(os.path.join(SAVE_DIR, "audit_sentences.csv"), index=False)
    print(f"primary sample {len(_prim_out)} ({ {k: int(v) for k, v in _prim['ipc_sec'].value_counts().items()} }) | edge {len(_edge_out)}"
          f" ({ {k: int(v) for k, v in _edge_out['stratum'].value_counts().items()} }) | sentences {len(_sent)}")
    print("saved -> audit_primary_50.csv, audit_edge_12.csv, audit_sentences.csv")


# ── 8.9  Usage example: CPC classification from each representation (family-grouped split) ──
# The example shows how the released embeddings plug into a standard classifier; it is not a claim that one
# representation is better than another.
# Main task: multi-label four-character CPC codes (micro-F1). The secondary codes name technical subjects that
# the specification discloses besides the main one, often not in the abstract.
# Secondary tasks: the main four-character code (macro-F1) and the main three-character class (accuracy).
# The classifier is fixed in advance: logistic regression on unit-length embeddings, a linear probe of the
# representation rather than of the classifier.
import ast, numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

# MIN_LABEL_N (codes with fewer records are dropped) and N_BOOT (bootstrap draws) are set in 1.3.

# Source of the multi-label labels: ipc_description (the list of all class codes of a record).
# A table in memory with this column is used first; otherwise the two columns are read from the candidate file.
_IPCSRC = None
for _c in ("m1", "corpus_df", "released_df", "valid_df", "df", "df_raw"):
    _o = globals().get(_c)
    if isinstance(_o, pd.DataFrame) and "ipc_description" in _o.columns:
        _IPCSRC = _o[["patent_id", "ipc_description"]]
        print(f"ipc_description source: {_c} in memory")
        break
if _IPCSRC is None:
    if "ipc_description" not in pd.read_csv(CLEAN_PATH, nrows=0).columns:
        raise RuntimeError(f"The candidate file has no ipc_description column: {CLEAN_PATH}")
    with timed("8.9  Read ipc_description"):
        _IPCSRC = pd.read_csv(CLEAN_PATH, usecols=["patent_id", "ipc_description"], dtype=str)
    print(f"ipc_description source: candidate file ({len(_IPCSRC):,} records)")
_IPCSRC = (_IPCSRC.dropna(subset=["patent_id"])
                  .astype({"patent_id": str})
                  .drop_duplicates("patent_id"))

def _ipc4(s):
    try:
        lst = ast.literal_eval(str(s)) if str(s).strip().startswith("[") else []
    except Exception:
        lst = []
    return {d["ipc"][:4] for d in lst if isinstance(d, dict) and d.get("ipc")}

lab = (valid_df[["patent_id", "main_ipc_4"]].astype({"patent_id": str})
       .merge(_IPCSRC, on="patent_id", how="left"))
# When the join key does not match, the multi-label task silently degrades into a single-label task with "only main_ipc_4",
# and the comparison of this section loses its meaning. Thus the coverage is printed, and the run stops when nothing matches.
_miss = int(lab["ipc_description"].isna().sum())
if _miss == len(lab):
    raise RuntimeError("The patent_id of valid_df does not match the ipc_description source at all; "
                       "SAVE_DIR and the candidate file belong to different corpus versions.")
print(f"ipc_description coverage: {len(lab) - _miss:,}/{len(lab):,} "
      f"({(1 - _miss / len(lab)) * 100:.1f}%)")
lab["all4"] = [s | ({m} if isinstance(m, str) and m else set())
               for s, m in zip(lab["ipc_description"].map(_ipc4), lab["main_ipc_4"])]
_cnt = pd.Series([c for s in lab["all4"] for c in s]).value_counts()
_keep = set(_cnt[_cnt >= MIN_LABEL_N].index)
lab["kept"] = [s & _keep for s in lab["all4"]]
mask = np.array([len(s) > 0 for s in lab["kept"]])
sub  = lab[mask].reset_index(drop=True)

mlb = MultiLabelBinarizer(classes=sorted(_keep)); Y = mlb.fit_transform(sub["kept"])
le  = LabelEncoder(); y1 = le.fit_transform(sub["main_ipc_4"].values)
# Three-character classes: about ten times fewer classes than the subclasses, a granularity close to that of the topics
sub = add_ipc3(sub)
le3 = LabelEncoder(); y3 = le3.fit_transform(sub["main_ipc_3"].values)
g   = build_family_groups(sub["patent_id"].values)
ix  = np.arange(len(sub))
tr, te, _, _ = family_train_test_split(ix, y1, g)

print(f"n={len(sub):,}/{len(valid_df):,} | multi-label classes {Y.shape[1]} (threshold >= {MIN_LABEL_N})"
      f" | mean labels per record {Y.sum(1).mean():.2f} | train {len(tr):,} | test {len(te):,} (family-grouped split)")
print(f"### Feasibility check: records with two or more labels {(Y.sum(1) >= 2).mean()*100:.1f}%"
      f"  ← this is the part that the abstract cannot easily answer; if it is too low, this task cannot show differences between representations")

REPS = {k: v for k, v in REPRESENTATIONS.items() if v is not None}
rows, P_ = [], {}
for rep, X in REPS.items():
    Xs = X[mask]
    with timed(f"8.9  {rep}  multi-label", n_items=len(tr), quiet=True) as t:
        m = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0), n_jobs=-1)
        _ = m.fit(Xs[tr], Y[tr]); Yp = m.predict(Xs[te])
    with timed(f"8.9  {rep}  single-label", n_items=len(tr), quiet=True) as t2:
        c = LogisticRegression(max_iter=2000, C=1.0); _ = c.fit(Xs[tr], y1[tr]); yp = c.predict(Xs[te])
    with timed(f"8.9  {rep}  ipc3", n_items=len(tr), quiet=True) as t3:
        c3 = LogisticRegression(max_iter=2000, C=1.0); _ = c3.fit(Xs[tr], y3[tr]); yp3 = c3.predict(Xs[te])
    P_[rep] = Yp
    ev = np.unique(y1[te])
    # Simple metric: the share of test patents with "at least one correct class code"
    _any_hit = float((np.logical_and(Y[te], Yp).sum(1) > 0).mean())
    rows.append({"representation": rep, "dim": int(Xs.shape[1]),
                 "any_label_hit": round(_any_hit, 4),
                 "ml_f1_micro": round(f1_score(Y[te], Yp, average="micro", zero_division=0), 4),
                 "ml_f1_macro": round(f1_score(Y[te], Yp, average="macro", zero_division=0), 4),
                 "single_f1_macro": round(f1_score(y1[te], yp, average="macro",
                                                   labels=ev, zero_division=0), 4),
                 "single_acc": round(accuracy_score(y1[te], yp), 4),
                 "ipc3_acc": round(accuracy_score(y3[te], yp3), 4),
                 "ipc3_f1_macro": round(f1_score(y3[te], yp3, average="macro",
                                                 labels=np.unique(y3[te]), zero_division=0), 4),
                 "n_ipc3_classes": int(len(np.unique(y3))),
                 "n_records": int(len(sub)), "n_labels": int(Y.shape[1]),
                 "n_test": int(len(te)), "fit_sec": round(t.seconds + t2.seconds, 1)})

# Floor: the single-label accuracy when the largest training class is always guessed, without the vectors
_floor  = float(pd.Series(y1[tr]).value_counts(normalize=True).iloc[0])
_floor3 = float(pd.Series(y3[tr]).value_counts(normalize=True).iloc[0])
print(f"\n  Classes: three-character {len(np.unique(y3))} | four-character {len(np.unique(y1))} | multi-label {Y.shape[1]}")
print(f"\n  {'representation':16}{'Acc (three)':>13}{'Acc (four)':>13}{'any correct':>12}"
      f"{'micro (multi)':>14}")
for r in rows:
    print(f"  {r['representation']:16}{r['ipc3_acc']*100:>12.1f}%{r['single_acc']*100:>12.1f}%"
          f"{r['any_label_hit']*100:>11.1f}%{r['ml_f1_micro']:>14.4f}")
print(f"  {'(largest class)':16}{_floor3*100:>12.1f}%{_floor*100:>12.1f}%{'—':>12}{'—':>14}"
      f"   <- floor without the vectors")
pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "ipc_downstream_comparison.csv"), index=False)

# ── Paired bootstrap (micro-F1), resampled indices shared over the test set ───
COMPARISONS = [
    # ── Against the inputs that already exist (the abstract, the cut full text) ───
    ("synthesized", "abstract",
     "better than the abstract alone (a field that the patent page already gives)?"),
    ("synthesized", "fulltext-truncated",
     "better than encoding the joined full text (cut at 512 tokens, out-of-distribution input)?"),
    # ── Descriptive differences between the released layers (CSV only) ───
    ("synthesized", "direct-full",  "release layers only: difference between layers"),
    ("synthesized", "direct-brief", "release layers only: difference between layers"),
    ("direct-full", "direct-brief", "release layers only: difference between layers"),
]
Yte = Y[te]; rng = np.random.RandomState(RANDOM_STATE); n = len(te)
bidx = [rng.randint(0, n, n) for _ in range(N_BOOT)]
boot = []
with timed("8.9  Paired bootstrap (micro-F1)", n_items=len(bidx)):
    for hi, lo, q in COMPARISONS:
        if hi not in P_ or lo not in P_:
            print(f"  [skipped {hi} − {lo}] (representation does not exist)"); continue
        pt = (f1_score(Yte, P_[hi], average="micro", zero_division=0)
              - f1_score(Yte, P_[lo], average="micro", zero_division=0))
        a = np.array([f1_score(Yte[b], P_[hi][b], average="micro", zero_division=0)
                      - f1_score(Yte[b], P_[lo][b], average="micro", zero_division=0) for b in bidx])
        l, h = np.percentile(a, 2.5), np.percentile(a, 97.5)
        v = "interval includes 0; no difference shown" if l <= 0 <= h else ("interval excludes 0; positive" if l > 0 else "interval excludes 0; negative")
        print(f"  [{hi:14} − {lo:14}] Δ {pt:+.4f} | 95% CI [{l:+.4f}, {h:+.4f}] → {v}   ({q})")
        boot.append({"comparison": f"{hi} - {lo}", "question": q, "metric": "micro_f1",
                     "point_estimate": round(float(pt), 4), "ci_low": round(float(l), 4),
                     "ci_high": round(float(h), 4), "n_boot": len(bidx), "verdict": v})
pd.DataFrame(boot).to_csv(os.path.join(SAVE_DIR, "ipc_downstream_bootstrap.csv"), index=False)
print("saved -> ipc_downstream_comparison.csv, ipc_downstream_bootstrap.csv")


# ── 8.10  Usage example: original assignee from each representation (family-grouped split) ──
# The label is assignee, the original applicant, not current_assignee: a transfer of the patent does not change
# its text, so a current-assignee label would ask for information that the text does not hold. Companies keep
# fixed drafting habits, so the task partly recognises vocabulary; it is an example, not validation evidence.
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# MIN_ASSIGNEE_N (minimum records per assignee class) is set in 1.3.

_src = (pd.read_csv(CLEAN_PATH, usecols=["patent_id", "assignee"], dtype=str)
          .fillna("").drop_duplicates("patent_id"))
lab = (valid_df[["patent_id"]].astype({"patent_id": str})
       .merge(_src, on="patent_id", how="left").fillna(""))
if len(lab) != len(valid_df):
    raise RuntimeError(f"The row count changed after the merge ({len(lab):,} vs {len(valid_df):,}): "
                       "make sure that patent_id is unique in the source file.")
_hit = int((lab["assignee"] != "").sum())
print(f"assignee coverage: {_hit:,}/{len(lab):,} ({_hit/len(lab)*100:.1f}%)")
if _hit == 0:
    raise RuntimeError("The patent_id of valid_df and the candidate file do not match at all.")

_cnt  = lab.loc[lab["assignee"] != "", "assignee"].value_counts()
_keep = set(_cnt[_cnt >= MIN_ASSIGNEE_N].index)
mask  = lab["assignee"].isin(_keep).values
sub   = lab[mask].reset_index(drop=True)

le = LabelEncoder(); y = le.fit_transform(sub["assignee"].values)
g  = build_family_groups(sub["patent_id"].values)
ix = np.arange(len(sub))
tr, te, y_tr, y_te = family_train_test_split(ix, y, g)

_maj = float(sub["assignee"].value_counts(normalize=True).iloc[0])
print(f"n={len(sub):,}/{len(valid_df):,} | classes {len(_keep)} (threshold >= {MIN_ASSIGNEE_N})"
      f" | train {len(tr):,} | test {len(te):,} (family-grouped split)")
print(f"### Feasibility check: share of the largest class {_maj*100:.1f}%"
      f"  <- an accuracy below this value means that the representation gives no information")

REPS = {k: v for k, v in REPRESENTATIONS.items() if v is not None}
rows = []
for rep, X in REPS.items():
    Xs = X[mask]
    with timed(f"8.10  {rep}  assignee prediction", n_items=len(tr), quiet=True) as _t:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        _ = clf.fit(Xs[tr], y_tr)
        yp = clf.predict(Xs[te])
    rows.append({"representation": rep, "dim": int(Xs.shape[1]),
                 "accuracy": round(accuracy_score(y_te, yp), 4),
                 "f1_macro": round(f1_score(y_te, yp, average="macro",
                                            labels=np.unique(y_te), zero_division=0), 4),
                 "n_records": int(len(sub)), "n_classes": int(len(_keep)),
                 "n_test": int(len(te)), "majority_share": round(_maj, 4),
                 "fit_sec": round(_t.seconds, 1)})

print(f"\n  {'representation':20}{'dim':>6}{'accuracy':>11}{'macro-F1':>11}")
for r in rows:
    print(f"  {r['representation']:20}{r['dim']:>6}{r['accuracy']:>11.4f}{r['f1_macro']:>11.4f}")
print("\n(One table is enough for the example; no interval estimates between representations)")

pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "developer_demo_comparison.csv"), index=False)
pd.DataFrame({"patent_id": sub.loc[te, "patent_id"].values,
              "assignee": le.inverse_transform(y_te),
              "predicted": le.inverse_transform(yp)}
             ).to_csv(os.path.join(SAVE_DIR, "developer_demo_predictions.csv"), index=False)
print("saved -> developer_demo_comparison.csv, developer_demo_predictions.csv")


# ── 8.11  GPT-2 perplexity of the three summary layers (distribution and outliers) ──
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

def calc_ppl(texts):
    ppls = []
    for text in tqdm(texts, desc="PPL"):
        text = str(text).strip()
        if not text:
            ppls.append(float("nan")); continue
        enc = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"].to(DEVICE)
        with torch.no_grad():
            loss = gpt2_model(ids, labels=ids).loss
        ppls.append(torch.exp(loss).item())
    return np.array(ppls)

PPL_SETS = [(n, c) for n, c in (("direct-brief", "baseline_summary"),
                                ("direct-full",  "baseline_matched_summary"),
                                ("synthesized",  "final_summary"))
            if c in valid_df.columns and valid_df[c].astype(str).str.len().gt(0).any()]
with timed("8.11  GPT-2 perplexity (summary layers)", n_items=len(PPL_SETS) * len(valid_df)):
    ppl = {n: calc_ppl(valid_df[c].tolist()) for n, c in PPL_SETS}

names = list(ppl)
print(f"\n{'':10}" + "".join(f"{n:>14}" for n in names))
for stat, fn in [("mean", np.nanmean), ("median", np.nanmedian),
                 ("p90", lambda a: np.nanpercentile(a, 90)),
                 ("min", np.nanmin), ("max", np.nanmax)]:
    print(f"{stat:10}" + "".join(f"{fn(ppl[n]):>14.2f}" for n in names))

for n in names:
    _anom = int((np.nan_to_num(ppl[n]) > 200).sum())
    print(f"{n} PPL > 200 (manual inspection of each record): {_anom} records")
_out = pd.DataFrame({"patent_id": valid_df["patent_id"]})
for n in names:
    _out[f"ppl_{n.replace('-', '_')}"] = ppl[n]
_out.to_csv(os.path.join(SAVE_DIR, "ppl_summaries.csv"), index=False)
print("saved -> ppl_summaries.csv")

free_gpu("gpt2_model", "gpt2_tok")


# ── 8.12  Cosine similarity of two summary layers to the abstract (SPECTER2) ──
# cos(summary embedding, abstract embedding); the summary side has no title.
from sklearn.metrics.pairwise import cosine_similarity   # 8.13 uses it
if "embed_batch" not in globals():
    raise RuntimeError("Run 6.12 first (SPECTER2 and embed_batch).")

with timed("8.12  SPECTER2 embedding (direct-brief, without title)", n_items=len(valid_df)):
    E_dir = embed_batch(valid_df["baseline_summary"].astype(str).tolist(),
                        desc="cos_to_abstract [direct-brief]")
with timed("8.12  SPECTER2 embedding (synthesized, without title)", n_items=len(valid_df)):
    E_two = embed_batch(valid_df["final_summary"].astype(str).tolist(),
                        desc="cos_to_abstract [synthesized]")
with timed("8.12  SPECTER2 embedding (abstract)", n_items=len(valid_df)):
    E_abs = embed_batch(valid_df["abstract"].astype(str).tolist(),
                        desc="cos_to_abstract [abstract]")

cos_to_abstract = {"direct-brief": np.einsum("ij,ij->i", E_dir, E_abs),
                   "synthesized":  np.einsum("ij,ij->i", E_two, E_abs)}
names = list(cos_to_abstract)
print(f"\n{'':10}" + "".join(f"{n:>13}" for n in names))
for stat, fn in [("mean", np.mean), ("median", np.median),
                 ("min", np.min), ("max", np.max)]:
    print(f"{stat:10}" + "".join(f"{fn(cos_to_abstract[n]):>13.4f}" for n in names))

pd.DataFrame({"patent_id": valid_df["patent_id"],
              "cos_to_abstract_direct": cos_to_abstract["direct-brief"],
              "cos_to_abstract_twostage": cos_to_abstract["synthesized"]}).to_csv(
    os.path.join(SAVE_DIR, "abstract_cosine.csv"), index=False)
print("saved -> abstract_cosine.csv")

free_gpu("specter_model", "specter_tokenizer")


# ── 8.13  Mean similarity within each topic, for each representation ────────
# The only measurement of section 8 that uses the topics; it runs on the topic_mask subset of 6.11.
def intra_cluster_sim_per_topic(X, labels):
    topic_sims = {}
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        if len(idx) < 2:
            continue
        sim = cosine_similarity(X[idx])
        n = len(idx)
        topic_sims[label] = np.mean([sim[i][j] for i in range(n) for j in range(i+1, n)])
    return topic_sims

labels_topic = valid_df.loc[topic_mask, "merged_topic"].values
REP_X = {k: v[topic_mask] for k, v in REPRESENTATIONS.items() if v is not None}
print(f"sample {topic_mask.sum():,} records | topics {len(set(labels_topic))}")
with timed("8.13  Within-topic similarity (representations)", n_items=3 * len(labels_topic)):
    sims = {k: intra_cluster_sim_per_topic(X, labels_topic) for k, X in REP_X.items()}

common = set.intersection(*(set(v) for v in sims.values()))
print("overall mean intra-topic similarity: " +
      " | ".join(f"{k} {np.mean([sims[k][t] for t in common]):.4f}" for k in REP_X))
print()
for t in sorted(common):
    print(f"  {t[:40]:40}" + "".join(f" {sims[k][t]:.4f}" for k in REP_X))
pd.DataFrame([{"merged_topic": t,
               **{k.split()[0].lower(): round(sims[k][t], 4) for k in REP_X}}
              for t in sorted(common)]).to_csv(
    os.path.join(SAVE_DIR, "intra_topic_similarity.csv"), index=False)
print("\nsaved -> intra_topic_similarity.csv")


# ── 8.14  BERTScore of two summary layers against the abstract ───────────────
from bert_score import score as bert_score

refs = valid_df["abstract"].tolist()
HYPS = [("direct-brief", valid_df["baseline_summary"].tolist()),
        ("synthesized",  valid_df["final_summary"].tolist())]

with timed("8.14  BERTScore (deberta-xlarge-mnli, summary layers)", n_items=2 * len(valid_df)):
    bs = {}
    for name, hyps in HYPS:
        P, R, F = bert_score(hyps, refs, lang="en", batch_size=64, device=DEVICE,
                             model_type="microsoft/deberta-xlarge-mnli", verbose=False)
        bs[name] = (P, R, F)

names = [n for n, _ in HYPS]
print(f"\n{'':12}" + "".join(f"{n:>14}" for n in names))
for row, j in [("Precision", 0), ("Recall", 1), ("F1", 2)]:
    print(f"{row:12}" + "".join(f"{bs[n][j].mean():>14.4f}" for n in names))
