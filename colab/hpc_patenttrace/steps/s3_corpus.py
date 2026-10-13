"""Section 3: corpus rules (completeness filter and inclusion rule).

Run from the pipeline notebook with run_step("s3_corpus"); the parts below are executed in the namespace of the notebook.
"""

# ── 3.1  Load the candidates (with a check against section 2) ─────────────────
with timed("3.1  Load the candidates"):
    df_raw = pd.read_csv(CLEAN_PATH, low_memory=False)

n_dup = df_raw["patent_id"].duplicated().sum()
if n_dup:
    print(f"WARNING: dropping {n_dup} duplicated patent_id rows")
    df_raw = df_raw.drop_duplicates(subset="patent_id").reset_index(drop=True)

# The candidate file must hold exactly the records that section 2 kept.
if "N_CLEAN" in globals() and len(df_raw) != N_CLEAN:
    raise RuntimeError(f"The candidate file has {len(df_raw):,} records, but section 2 kept {N_CLEAN:,}. "
                       "Run section 2 again with the same configuration.")
N_INPUT = len(df_raw)
print(f"Candidates: {N_INPUT:,} records x {len(df_raw.columns)} columns ({os.path.basename(CLEAN_PATH)})")
df = df_raw


# ── 3.2  Descriptive statistics of the candidates (before the corpus rules) ──

# Safety: make sure that patent_id is unique (all downstream merges assume that it is unique)
n_dup = df["patent_id"].duplicated().sum()
if n_dup:
    print(f"WARNING: dropping {n_dup} duplicated patent_id rows")
    df = df.drop_duplicates(subset="patent_id").reset_index(drop=True)

print(f"records : {len(df):,}")
print(f"columns : {len(df.columns)}")

print(f"\n{'='*68}")
print("per-column validity")
print(f"{'='*68}")
for col in df.columns:
    if df[col].dtype == object:
        valid = (df[col].fillna("").astype(str).str.strip() != "").sum()
    else:
        valid = df[col].notna().sum()
    print(f"  {col:42s} {valid:>7,}  ({valid/len(df)*100:5.1f}%)")

TEXT_STAT_COLS = [c for c in ["abstract","background","summary","detailed_description","claims_text"] if c in df.columns]
print(f"\n{'='*68}")
print("word-count statistics (non-empty rows)")
print(f"{'='*68}")
print(f"  {'column':26s} {'n':>7} {'mean':>8} {'median':>8} {'min':>6} {'max':>9}")
for col in TEXT_STAT_COLS:
    filled = df[col].fillna("").astype(str).str.strip()
    filled = filled[filled != ""]
    w = filled.str.split().str.len()
    print(f"  {col:26s} {len(w):>7,} {w.mean():>8.0f} {w.median():>8.0f} {w.min():>6,} {w.max():>9,}")

print(f"\n{'='*68}")
print("application_year distribution")
print(f"{'='*68}")
for yr, cnt in df["application_year"].value_counts().sort_index().items():
    print(f"  {yr}  {cnt:>5,}  {'█' * (cnt // 40)}")

print(f"\n{'='*68}")
print("main_ipc_4 top 10")
print(f"{'='*68}")
for ipc, cnt in df["main_ipc_4"].value_counts().head(10).items():
    print(f"  {ipc:8s} {cnt:>5,} ({cnt/len(df)*100:4.1f}%)  {'█' * (cnt // 40)}")

print(f"\n{'='*68}")
print("current_assignee top 10")
print(f"{'='*68}")
for co, cnt in df["current_assignee"].value_counts().head(10).items():
    print(f"  {co[:40]:42s} {cnt:>5,}")


# ── 3.3  Five-field completeness filter ──────────────────────────────────────
# A record stays when title, abstract, background, detailed_description and claims_text all have text.
# summary is exempt by design: a summary of the invention is not a required part of a granted
# specification, and requiring it would discard records that are otherwise complete. Because the parser
# writes "N/A" when a section heading does not match one of its standard headings, this filter measures
# the success of section extraction, not the completeness of the patents themselves.
REQUIRED_FULL = ["title", "abstract", "background",
                 "detailed_description", "claims_text"]
_complete = pd.Series(True, index=df_raw.index)
for _c in REQUIRED_FULL:
    _complete &= df_raw[_c].fillna("").astype(str).str.strip().str.len() > 0

# For 3.4: keep a snapshot of the minimum columns before the filter and a matrix of the missing fields (selection-bias comparison)
_BIAS_COLS = [c for c in ("patent_id", "main_ipc_4", "application_year",
                          "current_assignee") if c in df_raw.columns]
_bias_src = df_raw[_BIAS_COLS].copy()
_bias_src["_kept"] = _complete.values
_bias_miss = pd.DataFrame(
    {c: df_raw[c].fillna("").astype(str).str.strip().str.len().eq(0).values
     for c in REQUIRED_FULL})

print(f"before completeness filter  : {len(df_raw):>7,}")
print(f"  − one of five fields empty: {(~_complete).sum():>7,}")
df_raw = df_raw[_complete].reset_index(drop=True)
print(f"after completeness filter   : {len(df_raw):>7,}")

N_COMPLETE = len(df_raw)
df = df_raw


# ── 3.4  Who the completeness filter removes (CPU only, read-only) ──────────
# The filter removes more than half of the candidates, so the removed and the kept records are compared on
# the missing field, class section and subclass, application year and assignee. Distributions only: no
# statistical test and no inference. All tables are written to tables/.
import os
import numpy as np, pandas as pd

for _n in ("_bias_src", "_bias_miss"):
    if _n not in globals():
        raise RuntimeError("Run 3.3 first (this cell uses the snapshot that 3.3 keeps before the filter).")

_b      = _bias_src
_keep   = _b["_kept"].to_numpy(dtype=bool)
_n_all  = len(_b); _n_keep = int(_keep.sum()); _n_drop = _n_all - _n_keep
print(f"Completeness filter: {_n_all:,} → {_n_keep:,}"
      f" (removed {_n_drop:,} = {_n_drop / _n_all * 100:.1f}%)\n")

# ── Table A  Which column is missing: the structure of the removal reasons ───
_rows = []
for _c in REQUIRED_FULL:
    _m = _bias_miss[_c].to_numpy(dtype=bool)
    _rows.append({"required_field": _c,
                  "records_missing": int(_m.sum()),
                  "share_of_all_pct": round(_m.mean() * 100, 1),
                  "share_of_removed_pct": round(_m[~_keep].mean() * 100, 1)})
_tabA = pd.DataFrame(_rows).sort_values("records_missing", ascending=False)
print("Table A  Missing values of each required field")
print(_tabA.to_string(index=False))
_nmiss = _bias_miss.sum(axis=1).to_numpy()
_dist  = pd.Series(_nmiss[~_keep]).value_counts().sort_index()
print("\n  Removed records by the number of missing fields:",
      " | ".join(f"{k} field(s) {v:,} records" for k, v in _dist.items()))
_tabA.to_csv(os.path.join(TBL_DIR, "completeness_missing_by_field.csv"), index=False)

# ── Table B  Removal rate by patent classification section (first letter of the CPC) ───
def _rate_table(key, label, min_n=0):
    _t = (pd.DataFrame({label: key, "kept": _keep})
          .groupby(label, dropna=False)
          .agg(before=("kept", "size"), after=("kept", "sum")))
    _t = _t[_t["before"] >= min_n].copy()
    _t["removed"]  = _t["before"] - _t["after"]
    _t["removal_rate_pct"] = (_t["removed"] / _t["before"] * 100).round(1)
    _t["share_before_pct"] = (_t["before"] / _n_all * 100).round(1)
    _t["share_after_pct"] = (_t["after"] / max(_n_keep, 1) * 100).round(1)
    return _t.sort_values("before", ascending=False)

if "main_ipc_4" in _b.columns:
    _sec = _b["main_ipc_4"].fillna("").astype(str).str[:1].replace("", "(missing)")
    _tabB = _rate_table(_sec, "cpc_section")
    print("\nTable B  Removal rate by class section (first letter of the CPC code)")
    print(_tabB.to_string())
    _spread = _tabB.loc[_tabB["before"] >= 100, "removal_rate_pct"]
    print(f"\n  Range of the removal rate for sections with ≥100 records: {_spread.min():.1f}% – {_spread.max():.1f}%")
    _tabB.to_csv(os.path.join(TBL_DIR, "completeness_bias_by_ipc_section.csv"))

    _tabB4 = _rate_table(_b["main_ipc_4"].fillna("(missing)").astype(str),
                         "cpc_subclass", min_n=100)
    print("\n  Removal rate of the 12 largest four-character subclasses")
    print(_tabB4.head(12).to_string())
    _tabB4.to_csv(os.path.join(TBL_DIR, "completeness_bias_by_ipc4.csv"))

# ── Table C  Removal rate by application year ───────────────────────────────
if "application_year" in _b.columns:
    _yr = pd.to_numeric(_b["application_year"], errors="coerce")
    _tabC = _rate_table(_yr.fillna(-1).astype(int), "application_year").sort_index()
    print("\nTable C  Removal rate by application year")
    print(_tabC.to_string())
    _y = _tabC.loc[_tabC.index >= 2015, "removal_rate_pct"]
    if len(_y):
        print(f"\n  Range of the yearly removal rate from 2015 on: {_y.min():.1f}% – {_y.max():.1f}%")
    _tabC.to_csv(os.path.join(TBL_DIR, "completeness_bias_by_year.csv"))

# ── Table D  Comparison of assignee concentration ───────────────────────────
if "current_assignee" in _b.columns:
    _a = _b["current_assignee"].fillna("").astype(str).str.strip()
    _rows = []
    for _name, _mask in ((f"before filter ({_n_all:,})",  np.ones(_n_all, bool)),
                         (f"kept ({_n_keep:,})",  _keep),
                         (f"removed ({_n_drop:,})",  ~_keep)):
        _s = _a[_mask]; _s = _s[_s != ""]
        _vc = _s.value_counts()
        _rows.append({"subset": _name, "records": int(_mask.sum()),
                      "distinct_assignees": int(_vc.size),
                      "top10_share_pct": round(_vc.head(10).sum() / max(len(_s), 1) * 100, 1),
                      "single_record_assignees_pct": round((_vc == 1).mean() * 100, 1)})
    _tabD = pd.DataFrame(_rows)
    print("\nTable D  Comparison of assignee concentration")
    print(_tabD.to_string(index=False))
    _tabD.to_csv(os.path.join(TBL_DIR, "completeness_bias_assignee.csv"), index=False)

    _top = _a[_keep].value_counts().head(10).index
    _cmp = pd.DataFrame({
        "records_kept": _a[_keep].value_counts().reindex(_top).fillna(0).astype(int),
        "records_removed": _a[~_keep].value_counts().reindex(_top).fillna(0).astype(int)})
    _cmp["assignee_removal_rate_pct"] = (_cmp["records_removed"] /
                          (_cmp["records_kept"] + _cmp["records_removed"]) * 100).round(1)
    print("\n  Removal rate of each of the 10 largest assignees of the kept corpus")
    print(_cmp.to_string())
    _cmp.to_csv(os.path.join(TBL_DIR, "completeness_bias_top_assignees.csv"))

print("\nsaved -> tables/completeness_*.csv")
# ── Everpure Inc (formerly Pure Storage): its specifications have no standard section headings ───
_e = _bias_src[_bias_src["current_assignee"].astype(str).str.strip() == "Everpure Inc"].copy()
_e["cpc_section"] = _e["main_ipc_4"].fillna("").astype(str).str[:1].replace("", "(missing)")
print("Everpure Inc:", len(_e), "records")
if len(_e):
    print(pd.crosstab(_e["cpc_section"], _e["_kept"].map({True: "kept", False: "removed"}), margins=True))
print("\nFour-character class distribution of the kept records:")
print(_e.loc[_e["_kept"], "main_ipc_4"].value_counts().head(8).to_string())
print("\nFour-character class distribution of the removed records:")
print(_e.loc[~_e["_kept"], "main_ipc_4"].value_counts().head(8).to_string())

# Also check whether the year gradient comes from A/C contamination
_x = _bias_src.copy()
_x["section"] = _x["main_ipc_4"].fillna("").astype(str).str[:1]
_x["computing"] = _x["section"].isin(["G", "H"])
print("\nComputing vs non-computing classes, removal rate by year:")
print(_x.pivot_table(index=pd.to_numeric(_x["application_year"], errors="coerce"),
                     columns="computing", values="_kept",
                     aggfunc=lambda s: round((1 - s.mean()) * 100, 1)).to_string())

# ── Yearly removal rate of the computing sections (G, H) and of the other sections, written to a file ───
(_x.pivot_table(index=pd.to_numeric(_x["application_year"], errors="coerce"),
                columns="computing", values="_kept",
                aggfunc=lambda s: round((1 - s.mean()) * 100, 1))
   .rename(columns={True: "computing_GH_removed_pct", False: "other_sections_removed_pct"})
   .rename_axis("application_year")
   .to_csv(os.path.join(TBL_DIR, "completeness_bias_by_year_computing.csv")))
print("saved -> tables/completeness_bias_by_year_computing.csv")


# ── 3.5  Inclusion rule: the expanded phrase must occur ──────────────────────
# One rule defines the released corpus: the text must contain the expanded phrase of the abbreviation,
# "high performance comput*" (hyphen and space variants, endings compute, computer, computing, computation).
# The search itself used the abbreviation HPC, so a released record contains both the abbreviation (somewhere
# in the Google Patents record) and the expanded phrase (in the parsed text). The rule runs again in every run.
EXPANSION_RE = re.compile(r"high[\s\-]{0,2}performance[\s\-]{0,2}comput", re.I)
TEXT_FIELDS  = ["title", "abstract", "background", "summary",
                "detailed_description", "claims_text"]
_fields = [c for c in TEXT_FIELDS if c in df_raw.columns]

with timed("3.5  Inclusion rule (expanded phrase)", n_items=len(df_raw)):
    _joined = df_raw[_fields].fillna("").astype(str).agg(" ".join, axis=1)
    has_expansion = _joined.str.contains(EXPANSION_RE)

df = df_raw[has_expansion].reset_index(drop=True)
N_RELEASED, N_EXCLUDED = len(df), int((~has_expansion).sum())
print(f"records with five complete fields : {len(df_raw):>6,}")
print(f"  without the expanded phrase     : {N_EXCLUDED:>6,}")
print(f"released corpus                   : {N_RELEASED:>6,}")

# The complete records that the rule excludes are released as well (full text, normalised again in 9.3)
_excl = add_ipc3(df_raw[~has_expansion].reset_index(drop=True))
_excl.to_csv(os.path.join(SAVE_DIR, "excluded_no_expansion.csv"), index=False)
print(f"saved -> excluded_no_expansion.csv ({len(_excl):,} records x {len(_excl.columns)} columns)")

# Identifiers of the released corpus: the fingerprint that all later steps check against
pd.DataFrame({"patent_id": df["patent_id"]}).to_csv(os.path.join(SAVE_DIR, "released_corpus_ids.csv"), index=False)
print(f"saved -> released_corpus_ids.csv ({N_RELEASED:,} records)")

# With a reused run: the corpus of this run must be the same set of records
_rel_path = os.path.join(SAVE_DIR, "reused_released_corpus_ids.csv")
if os.path.exists(_rel_path):
    _old, _new = set(pd.read_csv(_rel_path, dtype=str)["patent_id"]), set(df["patent_id"])
    print(f"\nCheck against the reused run: {len(_new):,} here, {len(_old):,} there, "
          f"only here {len(_new - _old)}, only there {len(_old - _new)} -> "
          f"{'identical' if _new == _old else 'DIFFERENT'}")
    if _new != _old:
        raise RuntimeError("The released corpus differs from the reused run; its generated layers cannot be reused. "
                           "Set REUSE_DIR = None or use the input file of that run.")


# ── 3.6  Inclusion evidence of each record and exclusion rate by section (CPU only) ──
# Two pieces of evidence that the rule removes the other senses of the abbreviation and not random records:
# (1) evidence_flags.csv records why each record is included. It is released, not applied: users who want
#     only records that define the abbreviation as high-performance computing keep sense_basis == "definition".
# (2) Exclusion rate by class section. A rule that matched a surface string would exclude every section at
#     the same rate; a rule that matches the computing sense must exclude much more in sections C and A
#     (hydroxypropyl cellulose, hematopoietic progenitor cell) than in G and H.
import os
import numpy as np, pandas as pd

# Definition form: the document ties the abbreviation to the expanded phrase ("high-performance computing (HPC)" or the reverse)
DEFINITION_RE = re.compile(
    r"high[\s\-]{0,2}performance[\s\-]{0,2}comput\w*[^()\n]{0,60}\(\s*HPC\s*\)"
    r"|\bHPC\b\s*[\(,:]\s*(?:i\.e\.,?\s*|that is,?\s*|or\s*)?"
    r"high[\s\-]{0,2}performance[\s\-]{0,2}comput", re.I)
HPC_ABBR_RE = re.compile(r"\bHPC\b")

with timed("3.6  Inclusion evidence flags", n_items=len(df)):
    _txt = df[_fields].fillna("").astype(str).agg(" ".join, axis=1)
    _exp_n  = _txt.map(lambda t: len(EXPANSION_RE.findall(t)))
    _abb_n  = _txt.map(lambda t: len(HPC_ABBR_RE.findall(t)))
    _hasdef = _txt.map(lambda t: bool(DEFINITION_RE.search(t)))
    _basis  = np.where(_hasdef, "definition",
              np.where(_exp_n.values > 0, "phrase", "no_evidence"))

ev = pd.DataFrame({"patent_id": df["patent_id"].values,
                   "sense_basis": _basis,
                   "expansion_mentions": _exp_n.values,
                   "hpc_mentions": _abb_n.values})
ev.to_csv(os.path.join(SAVE_DIR, "evidence_flags.csv"), index=False)

print("Inclusion evidence (sense_basis)")
print("-" * 62)
for _b in ("definition", "phrase", "no_evidence"):
    _m = ev["sense_basis"] == _b
    if not _m.any():
        print(f"  {_b:12} {0:>6,}  ( 0.0%)"); continue
    _e = ev.loc[_m, "expansion_mentions"]
    print(f"  {_b:12} {int(_m.sum()):>6,}  ({_m.mean()*100:>4.1f}%)"
          f"  expanded-phrase mentions median {_e.median():>4.0f} | mean {_e.mean():>5.2f}"
          f" | three or more {(_e >= 3).mean()*100:>4.1f}% | exactly one {int((_e == 1).sum()):>5,}")
_one = ev[ev["expansion_mentions"] == 1]
if len(_one):
    _top = _one["sense_basis"].value_counts()
    print(f"\n  {len(_one):,} records in the whole corpus mention it only once; of these, "
          f"{_top.index[0]} has {int(_top.iloc[0]):,}")

# ── Differential exclusion rate: if the rule caught a surface string, the exclusion rates of all sections would be the same ───
_bef = df_raw["main_ipc_4"].astype(str).str[0].replace("", "?")
_aft = df["main_ipc_4"].astype(str).str[0].replace("", "?")
_b, _a = _bef.value_counts(), _aft.value_counts()
_rows = []
for _s in sorted(set(_b.index)):
    _nb, _na = int(_b.get(_s, 0)), int(_a.get(_s, 0))
    _rows.append({"ipc_section": _s, "before": _nb, "after": _na,
                  "removed": _nb - _na,
                  "exclusion_rate_pct": round((_nb - _na) / _nb * 100, 1) if _nb else None})
t_excl = pd.DataFrame(_rows).sort_values("before", ascending=False)
t_excl.to_csv(os.path.join(SAVE_DIR, "differential_exclusion.csv"), index=False)

print(f"\nDifferential exclusion rate ({len(df_raw):,} -> {len(df):,})")
print("-" * 62)
print(f"  {'section':<6}{'before':>8}{'after':>8}{'removed':>9}{'rate (%)':>9}")
for _r in t_excl.itertuples():
    print(f"  {_r.ipc_section:<8}{_r.before:>8,}{_r.after:>8,}{_r.removed:>9,}"
          f"{_r.exclusion_rate_pct:>8.1f}%")
_noncomp_b = int(_b[[s for s in _b.index if s not in ("G", "H")]].sum())
_noncomp_a = int(_a[[s for s in _a.index if s not in ("G", "H")]].sum())
print(f"\n  Share of the non-computing sections (other than G, H): {_noncomp_b/len(df_raw)*100:.1f}% "
      f"-> {_noncomp_a/len(df):.1%}")
print("saved -> evidence_flags.csv, differential_exclusion.csv")


# ── 3.7  Working table m1 for the generation steps ───────────────────────────
KEEP_COLS = ["patent_id","title","abstract","current_assignee",
             "main_ipc_4","main_ipc_description_4","application_year",
             "background","summary","detailed_description","claims_text"]
TEXT_COLS = ["title","abstract","background","summary","detailed_description"]

df = add_ipc3(df)                      # add the three-character class to the released corpus
m1 = df[[c for c in KEEP_COLS if c in df.columns]].copy()
m1 = add_ipc3(m1)
for col in TEXT_COLS:
    if col in m1.columns:
        m1[col] = m1[col].fillna("").astype(str).str.strip()

# summary fill: when `summary` is missing or too short, take the first
# 2,000 characters of `detailed_description` as a pseudo-summary.
# Many records have no summary section, so the values are converted to strings before len() is applied.
_s = m1["summary"].fillna("").astype(str) if "summary" in m1.columns else pd.Series("", index=m1.index)
_d = m1["detailed_description"].fillna("").astype(str)
m1["summary_filled"] = np.where(
    (_s.str.len() < MIN_LEN) & (_d.str.len() >= MIN_LEN),
    _d.str[:2000], _s)
m1["summary_is_substituted"] = (m1["summary_filled"] != _s)

# sections_usable: the number of usable sections among the three sections (0-3)
m1["sections_usable"] = (
    (m1["background"].str.len()           >= MIN_LEN).astype(int) +
    (m1["summary_filled"].str.len()       >= MIN_LEN).astype(int) +
    (m1["detailed_description"].str.len() >= MIN_LEN).astype(int)
)

print(f"cohort size          : {len(m1):,}")
print(f"summary before fill  : {(m1['summary'].str.len()>=MIN_LEN).sum():,}")
print(f"summary after  fill  : {(m1['summary_filled'].str.len()>=MIN_LEN).sum():,}")
print("\nsections_usable distribution (3 = all sections usable):")
print(m1["sections_usable"].value_counts().sort_index().to_string())


# ── 3.8  Litigated families in the released corpus (derived at run time) ──────
# The litigation (court/review records) and family (patent family) columns of the corpus are value-added fields that only this dataset has.
# The subset is derived from the candidate file in every run, like the released corpus in 3.5.
#
# Derivation rules
#   seed         litigation column not empty: the patent itself has a record of a district-court case, of an inter partes review
#                (inter partes review, IPR) or of a reexamination
#   family_root  the family root number in the family column ("US13/750696 (n=2)" gives "US13/750696");
#                a record without a family is a family of its own, written as self:<patent_id>
#   subset       all members of each family that contains a seed. A continuation and its parent application have
#                almost the same specification text and fall at the same position of the semantic landscape, so the whole family is included
#   case_group   the district-court case number parsed from the litigation text (for example 3:20-cv-03137).
#                One case number can cover more than one family (several patents of the same defendant are litigated together); family_root
#                cannot give this information. A record that has only an inter partes review/reexamination and no case number uses the proceeding type plus
#                the family root as its group key
CASE_NO_RE  = re.compile(r"\d+:\d{2}-cv-\d+", re.I)
PROCEEDINGS = (("district court", "DC"), ("IPR", "IPR"), ("reexamination", "REEX"))

with timed("3.8  Litigated-family subset", n_items=len(df)):
    _lit_src = pd.read_csv(CLEAN_PATH, usecols=["patent_id", "family", "litigation"],
                           dtype=str).fillna("")
    _lit_src = _lit_src[_lit_src["patent_id"].isin(set(df["patent_id"]))].copy()

    _root = _lit_src["family"].str.extract(r"^\s*(\S+)", expand=False).fillna("")
    _lit_src["family_root"]  = _root.where(_root != "", "self:" + _lit_src["patent_id"])
    _lit_src["case_numbers"] = _lit_src["litigation"].map(
        lambda t: ";".join(sorted(set(CASE_NO_RE.findall(t)))))

    _seed = _lit_src[_lit_src["litigation"].str.strip() != ""]
    _cg = {}
    for _r, _g in _seed.groupby("family_root"):
        _nums = sorted({n for s in _g["case_numbers"] if s for n in s.split(";")})
        if _nums:
            _cg[_r] = ";".join(_nums)
        else:
            _tags = [tag for kw, tag in PROCEEDINGS if any(kw in t for t in _g["litigation"])]
            _cg[_r] = f"{'/'.join(_tags) or 'UNKNOWN'}:{_r}"

    lit_family_df = _lit_src[_lit_src["family_root"].isin(set(_seed["family_root"]))].copy()
    lit_family_df["case_group"] = lit_family_df["family_root"].map(_cg)
    lit_family_df["is_seed"]    = lit_family_df["patent_id"].isin(set(_seed["patent_id"]))
    lit_family_df = (lit_family_df[["patent_id", "family_root", "case_numbers",
                                    "case_group", "is_seed", "litigation"]]
                     .sort_values(["case_group", "family_root", "patent_id"])
                     .reset_index(drop=True))

lit_family_df.to_csv(os.path.join(SAVE_DIR, "litigation_family.csv"), index=False)
print(f"seed (own record)         : {len(_seed):>4,}  ({len(_seed)/len(df)*100:.2f}% of the released corpus)")
print(f"subset with family members: {len(lit_family_df):>4,}  "
      f"({len(lit_family_df)/len(df)*100:.2f}% of the released corpus)")
print(f"families / case groups    : {lit_family_df['family_root'].nunique()} / "
      f"{lit_family_df['case_group'].nunique()}")
print("Note: the litigation column records the 'known' court and review events. An empty field does not mean that the patent was not litigated.")
print("\nsaved -> litigation_family.csv")


# ── 3.9  Descriptive tables of the released corpus (CPU only) ────────────────
# Seven tables, each saved to tables/ as a CSV.
_rel = df


def _save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(os.path.join(TBL_DIR, name), index=False)


def _wordcount(col: pd.Series) -> pd.Series:
    s = col.fillna("").astype(str).str.strip()
    return s[s != ""].str.split().str.len()


print(f"Released corpus: {len(_rel):,} records × {len(_rel.columns)} columns\n")

# ── Table 1  Corpus construction funnel ──────────────────────────────────────
_stages = [("candidates after the homonym screening", N_INPUT, ""),
           ("five-field completeness filter", N_COMPLETE,
            "title, abstract, background, detailed_description and claims_text all non-empty"),
           ("inclusion rule (expanded phrase)", len(_rel), "the text contains a variant of high performance comput*")]
t1 = pd.DataFrame(_stages, columns=["stage", "records", "criterion"])
t1["removed"] = [""] + [f"−{t1['records'][i-1] - t1['records'][i]:,}" for i in range(1, len(t1))]
print("Table 1  Corpus construction funnel"); print(t1.to_string(index=False)); _save(t1, "t1_funnel.csv")

# ── Table 2  Field completeness ──────────────────────────────────────────────
t2 = pd.DataFrame({
    "field": _rel.columns,
    "records_non_empty": [int((_rel[c].fillna("").astype(str).str.strip() != "").sum()) for c in _rel.columns],
})
t2["non_empty_pct"] = (t2["records_non_empty"] / len(_rel) * 100).round(1)
print(f"\nTable 2  Field completeness ({len(t2)} fields; complete fields: "
      f"{int((t2['non_empty_pct'] == 100).sum())})")
print(t2[t2["non_empty_pct"] < 100].to_string(index=False))
_save(t2, "t2_field_completeness.csv")

# ── Table 3  Text length ─────────────────────────────────────────────────────
_rows = []
for c in ["title", "abstract", "background", "summary",
          "detailed_description", "claims_text"]:
    if c not in _rel.columns:
        continue
    w = _wordcount(_rel[c])
    _rows.append([c, len(w), round(w.mean()), int(w.median()),
                  int(w.quantile(.10)), int(w.quantile(.90)), int(w.max())])
t3 = pd.DataFrame(_rows, columns=["field", "records_non_empty", "mean", "median", "P10", "P90", "max"])
print("\nTable 3  Text length (words)"); print(t3.to_string(index=False)); _save(t3, "t3_text_length.csv")

# ── Table 4  Application year distribution ───────────────────────────────────
_yr = pd.to_numeric(_rel["application_year"], errors="coerce")
t4 = _yr.value_counts().sort_index().rename_axis("application_year").reset_index(name="records")
t4["share_pct"] = (t4["records"] / len(_rel) * 100).round(1)
t4["note"] = np.where(t4["application_year"] >= 2023, "right-truncated (applications still under examination are not yet granted)", "")
print("\nTable 4  Application year distribution"); print(t4.to_string(index=False)); _save(t4, "t4_application_year.csv")

# ── Table 5  Patent classification distribution (CPC; the column name keeps ipc) ──────────────
_sec = _rel["main_ipc_4"].astype(str).str[0]
t5a = _sec.value_counts().rename_axis("cpc_section").reset_index(name="records")
t5a["share_pct"] = (t5a["records"] / len(_rel) * 100).round(1)
print("\nTable 5a  International Patent Classification section"); print(t5a.to_string(index=False))
_save(t5a, "t5a_ipc_section.csv")

_desc = (_rel.drop_duplicates("main_ipc_4")
         .set_index("main_ipc_4")["main_ipc_description_4"].astype(str))
t5b = _rel["main_ipc_4"].value_counts().head(15).rename_axis("cpc_subclass").reset_index(name="records")
t5b["share_pct"] = (t5b["records"] / len(_rel) * 100).round(1)
t5b["cpc_description"] = t5b["cpc_subclass"].map(_desc).str[:60]
print("\nTable 5b  The 15 largest four-character classes"); print(t5b.to_string(index=False))
_save(t5b, "t5b_ipc_top15.csv")

# ── Table 6  Assignee concentration ──────────────────────────────────────────
_ca = _rel["current_assignee"]
_vc = _ca.value_counts()
print(f"\nTable 6  Assignee concentration")
print(f"  distinct assignees        : {_ca.nunique():,}")
print(f"  share of the top 10       : {_vc.head(10).sum()/len(_rel)*100:.1f}%")
print(f"  share of the top 20       : {_vc.head(20).sum()/len(_rel)*100:.1f}%")
print(f"  companies with 1 record   : {int((_vc == 1).sum()):,}"
      f" ({(_vc == 1).sum()/_ca.nunique()*100:.1f}% of all companies)")
t6 = _vc.head(20).rename_axis("assignee").reset_index(name="records")
t6["share_pct"] = (t6["records"] / len(_rel) * 100).round(1)
t6.insert(0, "rank", range(1, len(t6) + 1))
print(t6.to_string(index=False)); _save(t6, "t6_top20_assignee.csv")

# ── Table 7  Citations, claims and value-added fields ────────────────────────
_rows = []
for c in ["citation_count", "claim_count"]:
    v = pd.to_numeric(_rel[c], errors="coerce")
    _rows.append([c, int(v.notna().sum()), round(v.mean(), 1), int(v.median()),
                  int(v.quantile(.90)), int(v.max())])
t7 = pd.DataFrame(_rows, columns=["field", "records_non_empty", "mean", "median", "P90", "max"])
print("\nTable 7  Citations and claims"); print(t7.to_string(index=False)); _save(t7, "t7_citation_claim.csv")

print(f"\nsaved -> {TBL_DIR}/ {len(os.listdir(TBL_DIR))} table files in total")


# ── 3.10  Figure 3.1: from the crawler output to the released corpus ─────────
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

_funnel = pd.DataFrame([
    ("crawler output (step 5)",             N_RAW,      ""),
    ("candidates",                          N_CLEAN,    "HPC has another sense"),
    ("five complete fields",                N_COMPLETE, "a required section was not extracted"),
    ("released corpus",                     N_RELEASED, "no \"high performance comput*\" in the text"),
], columns=["stage", "records", "removed_because"])
_funnel["removed"] = [0] + [int(_funnel["records"][i - 1] - _funnel["records"][i]) for i in range(1, len(_funnel))]
_funnel["kept_percent_of_input"] = (_funnel["records"] / N_RAW * 100).round(1)
_funnel.to_csv(os.path.join(TBL_DIR, "funnel_from_crawler_output.csv"), index=False)
_funnel

fig, ax = plt.subplots(figsize=(7.2, 2.6))
_y = np.arange(len(_funnel))[::-1]
_ = ax.barh(_y, _funnel["records"], color=PALETTE[0], height=0.62)
for yy, (_, r) in zip(_y, _funnel.iterrows()):
    _ = ax.text(r["records"] + N_RAW * 0.01, yy, f"{r['records']:,}", va="center", fontsize=8, color=INK)
    if r["removed"]:
        _ = ax.text(N_RAW * 1.17, yy, f"-{r['removed']:,}  {r['removed_because']}", va="center",
                    fontsize=7.5, color=INK_2)
_ = ax.set_yticks(_y, _funnel["stage"])
_ = ax.set_xlim(0, N_RAW * 1.15)
_ = ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
_ = ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
_ = ax.set_xlabel("records")
_ = ax.set_title("Figure 3.1  Records at each step of the corpus construction")
fig.tight_layout()
_ = save_fig(fig, "fig_3_1_corpus_funnel")
plt.show()
