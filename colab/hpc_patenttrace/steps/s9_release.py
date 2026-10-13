"""Section 9: release assets, cost table and release files.

Run from the pipeline notebook with run_step("s9_release"); the parts below are executed in the namespace of the notebook.
"""

# ── 9.1  Release assets: prompts, term lists and settings, written from the objects of this run ──
import json, os, re

ASSET_DIR = os.path.join(SAVE_DIR, "release_assets")
for _d in ("prompts", "terminology", "config"):
    os.makedirs(os.path.join(ASSET_DIR, _d), exist_ok=True)

_written, _skipped = [], []

def _put(rel, text, why="not defined (the section was not run)"):
    """Write a text to its own file; text None means that the object is not yet defined."""
    if text is None:
        _skipped.append((rel, why)); return
    p = os.path.join(ASSET_DIR, rel)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text if str(text).endswith("\n") else str(text) + "\n")
    _written.append((rel, len(str(text))))

_G = globals()
_sp = _G.get("SECTION_PROMPTS") or {}

# ── prompts: one file each; {placeholder} kept as it is ──────────────────
_put("prompts/stage1_background.txt",            _sp.get("BACKGROUND"))
_put("prompts/stage1_summary.txt",               _sp.get("SUMMARY"))
_put("prompts/stage1_detailed_description.txt",  _sp.get("DETAILED DESCRIPTION"))
_put("prompts/stage2_cot_reasoning.txt",         _G.get("COT_REASONING_PROMPT"))
_put("prompts/stage2_final_synthesis.txt",       _G.get("FINAL_SYNTHESIS_PROMPT_V2"))
_put("prompts/stage2_compression_optional.txt",  _G.get("COMPRESSION_PROMPT"))
_put("prompts/single_pass_baseline.txt",         _G.get("DIRECT_PROMPT"))
_put("prompts/cluster_labeling.txt",             _G.get("CLUSTER_LABEL_PROMPT"))

# ── terminology: the three term lists + the extractor ───────────────────
_wl = _G.get("HPC_WHITELIST")
_sw = _G.get("TITLE_STOPWORDS")
_gs = _G.get("GENERIC_SUFFIXES")
_put("terminology/hpc_whitelist.txt",
     None if _wl is None else
     ("# High-performance computing whitelist, tier 1 of the must-keep term strategy.\n"
      f"# {len(_wl)} terms, one per line, lower-case, matched case-insensitively.\n"
      + "\n".join(sorted(_wl))))
_put("terminology/title_stopwords.txt",
     None if _sw is None else
     ("# Words excluded when tier 2 harvests informative terms from the patent title.\n"
      f"# {len(_sw)} terms, one per line.\n" + "\n".join(sorted(_sw))))
_put("terminology/generic_suffixes.txt",
     None if _gs is None else
     ("# Regular expression (case-insensitive) rejecting morphologically generic words\n"
      "# in tiers 2 and 3. Anchored at the end of the token.\n" + _gs.pattern))

# Extractor: the source code of the function itself, so that it is the definition of 5.3
_why_fn = "not defined (the section was not run)"
try:
    import inspect
    _fn_src = inspect.getsource(_G["extract_tech_terms"])
except KeyError:
    _fn_src = None
except Exception as _e:                       # the function exists, but its source code is not available
    _fn_src, _why_fn = None, (
        f"The function is defined, but inspect cannot get its source code ({type(_e).__name__}). "
        "It is not rewritten from constants, so that the released extractor does not differ from the version that actually ran: "
        "rerun this cell in a normal notebook session.")
_put("terminology/extract_tech_terms.py",
     None if _fn_src is None else
     ('"""Must-keep term extractor (tier 1 whitelist > tier 2 title > tier 3 long words), cap 10.\n'
      'Standalone copy of the single definition in section 5.3 of the pipeline notebook.\n'
      'Run from this directory; the three lists are loaded from the files beside it.\n"""\n'
      "import re\n\n"
      "HPC_WHITELIST = {w.strip() for w in open('hpc_whitelist.txt') if w.strip() and not w.startswith('#')}\n"
      "TITLE_STOPWORDS = {w.strip() for w in open('title_stopwords.txt') if w.strip() and not w.startswith('#')}\n"
      "GENERIC_SUFFIXES = re.compile(\n"
      "    [l for l in open('generic_suffixes.txt') if not l.startswith('#')][0].strip(), re.I)\n\n"
      + _fn_src), why=_why_fn)

# ── config: generation ─────────────────────────────────────────────────
_matched = _G.get("MATCHED_BUDGETS") or {"background": 1500, "summary": 1500, "detail": 2000}
_generation = {
    "random_state": _G.get("RANDOM_STATE"),
    "min_section_chars": _G.get("MIN_LEN"),
    "models": {
        "stage1_facets":   {"id": S1_MODEL_ID, "precision": "4-bit NF4" if S1_4BIT else str(GEN_DTYPE).replace("torch.", ""),
                            "batch": _G.get("S1_GPU_BATCH")},
        "stage2_condense": {"id": S2_MODEL_NAME,
                            "precision": str(GEN_DTYPE).replace("torch.", ""), "batch": _G.get("S2_GPU_BATCH")},
        "single_pass":     {"id": S2_MODEL_NAME,
                            "precision": str(GEN_DTYPE).replace("torch.", ""), "batch": _G.get("BASE_BATCH")},
        "embedding":       {"id": _G.get("SPECTER2_MODEL", "allenai/specter2_base"),
                            "adapter": "allenai/specter2 (proximity)",
                            "dim": 768, "max_input_tokens": 512},
    },
    "decoding": {
        "do_sample": False, "repetition_penalty": 1.15,
        "max_new_tokens": {"stage1": 200, "stage2": 300, "single_pass": 300, "cluster_label": 100},
        "note": "greedy for every generated set, so no difference between sets is sampling noise",
    },
    "input_budgets_tokens": {
        "synthesized_facets": {"background": 1500, "summary": 1500,
                               "detailed_description": 2000, "total_cap_approx": 5000},
        "direct_brief":       {"background": 800, "summary": 400,
                               "detailed_description": 800, "total_cap_approx": 2000},
        "direct_full":        {"background": _matched["background"], "summary": _matched["summary"],
                               "detailed_description": _matched["detail"], "total_cap_approx": 5000,
                               "realised_prompt_tokens_median": _G.get("DIRECT_FULL_PROMPT_MEDIAN")},
    },
    "set_differences": {
        "direct_full_vs_direct_brief": "input budget only",
        "synthesized_vs_direct_full": "a separate facet step (Llama-3.1-8B-Instruct, 4-bit; inputs counted in Llama tokens), a reasoning call (up to 350 new tokens) before the final call, a different prompt template, and up to ten terms to keep in both calls",
        "held_constant_in_final_call": ["model", "precision", "decoding", "repetition_penalty",
                          "max_new_tokens", "target length 5-7 sentences"],
        "note": "the three sets are released as parallel data layers; no set is ranked above another",
    },
    "must_keep_terms": {"tiers": ["whitelist", "title", "long words (>=12 chars)"],
                        "cap_per_patent": 10, "applied_to": ["synthesized only"]},
    "summary_fallback": {"field": "summary",
                         "rule": "first 2000 characters of detailed_description when summary has fewer than 100 characters and detailed_description has at least 100 characters",
                         "affected_share": round(float(m1["summary_is_substituted"].mean()), 3)},
}
_put("config/generation.json", json.dumps(_generation, indent=2, ensure_ascii=False))

# ── config: clustering (read the live objects when they are available) ───
_cl = _G.get("clusterer")
_um = _G.get("umap_3d")
_put("config/clustering.json", json.dumps({
    "umap": {"n_components": getattr(_um, "n_components", 3),
             "n_neighbors": getattr(_um, "n_neighbors", 15),
             "min_dist": getattr(_um, "min_dist", 0.0),
             "metric": getattr(_um, "metric", "cosine"),
             "random_state": _G.get("RANDOM_STATE"),
             "low_memory": getattr(_um, "low_memory", False)},
    "hdbscan": {"min_cluster_size": getattr(_cl, "min_cluster_size", 30),
                "min_samples": getattr(_cl, "min_samples", 10),
                "metric": getattr(_cl, "metric", "euclidean"),
                "cluster_selection_method": getattr(_cl, "cluster_selection_method", "eom"),
                "prediction_data": bool(getattr(_cl, "prediction_data", True)),
                "gen_min_span_tree": bool(getattr(_cl, "gen_min_span_tree", True))},
    "selection_rule": ("the only setting in the released six-point sweep with DBCV > 0.3; "
                       "Silhouette and noise share are secondary criteria"),
    "reported_thresholds": {"silhouette": "> 0.3", "dbcv": "> 0.3", "noise_share": "< 0.20"},
    "sweep_file": "hdbscan_param_sweep.csv",
    "topic_merge": {"method": "average-linkage agglomerative clustering of the cluster centroids on cosine distance (section 6.9)",
                    "n_topics": _G.get("N_TARGET"), "min_topic_records": _G.get("MIN_CLASS_N"),
                    "small_topic_rule": "a topic with fewer than min_topic_records records joins the topic with the most similar centroid",
                    "label_rule": "the usable label of the largest member cluster that a larger topic has not taken",
                    "renamed_topics": len(_G.get("RENAME") or {}),
                    "worksheet": "merge_map_worksheet.csv",
                    "note": "rule-based; the worksheet lists the merge so that users can audit or replace it"},
}, indent=2, ensure_ascii=False))

# ── config: corpus rule ────────────────────────────────────────────────
_re_obj = _G.get("EXPANSION_RE")
_put("config/corpus_rule.json", json.dumps({
    "inclusion_regex": _re_obj.pattern if _re_obj is not None
                       else r"high[\s\-]{0,2}performance[\s\-]{0,2}comput",
    "regex_flags": "IGNORECASE",
    "scanned_fields": _G.get("TEXT_FIELDS", ["title", "abstract", "background", "summary",
                                             "detailed_description", "claims_text"]),
    "required_non_empty_fields": ["title", "abstract", "background",
                                  "detailed_description", "claims_text"],
    "exempt_field": {"summary": "not a required component of a granted specification"},
    "funnel": [{"stage": "input file (query term HPC; records using the abbreviation in unrelated senses removed)", "n": N_INPUT},
               {"stage": "five-field completeness", "n": N_COMPLETE},
               {"stage": "expansion-phrase rule", "n": N_RELEASED},
               {"stage": "excluded by the rule, released in full", "n": N_EXCLUDED}],
    "portability_note": ("the abbreviation / expansion-phrase / field-completeness funnel "
                         "transfers to any technical domain delimited by an abbreviation; "
                         "replace the regex and the required-field list"),
}, indent=2, ensure_ascii=False))

print(f"release_assets -> {ASSET_DIR}")
for rel, sz in _written:
    print(f"  {rel:46} {sz:>6} chars")
if _skipped:
    print("\n  Skipped:")
    for rel, why in _skipped:
        print(f"    {rel:44} {why}")
else:
    print(f"\n  All {len(_written)} files written")


# ── 9.2  Run-time tables: this run, and the cost table of the release ─────────
# runtime_log.csv holds the timed steps of this run. For the release table, a generation step that this run
# did not perform (its checkpoint was reused) takes the full generation of that step recorded in the log of the
# reused run (6,535 records, at least 60 s), or else the first full generation run on one A100-SXM4-80GB
# (2026-08-16); every other step keeps the time measured in this run.
rt = pd.read_csv(RUNTIME_PATH)
rt["_sec"] = rt["step"].astype(str).str.split().str[0]
rt["note"] = rt["note"].astype(object)

_FIRST_PASS = {"4.3": 17075, "5.6": 14243, "5.10": 13253, "5.11": 28524}   # seconds; 4:44:35, 3:57:23, 3:40:53, 7:55:24
_GEN_NAME   = {"4.3": "4.3  Facet generation (Llama-3.1-8B-Instruct)",
               "5.6": "5.6  Synthesized summary generation (Qwen2.5-14B-Instruct)",
               "5.10": "5.10  direct-brief generation (Qwen2.5-14B-Instruct, single prompt)",
               "5.11": "5.11  direct-full generation (Qwen2.5-14B-Instruct, facet budgets)"}
# Full generations in the log of the reused run. A log of the earlier notebook numbers these steps
# 3.3, 4.6, 7.1b and 7.1b′; it is recognised by its section 7.1b.
_measured = {}
_reused_rt_path = os.path.join(SAVE_DIR, "reused_runtime_log.csv")
if os.path.exists(_reused_rt_path):
    _old = pd.read_csv(_reused_rt_path)
    _old_sec = _old["step"].astype(str).str.split().str[0]
    if _old_sec.isin(["7.1b", "7.1b′"]).any():
        _old_sec = _old_sec.replace({"3.3": "4.3", "4.6": "5.6", "7.1b": "5.10", "7.1b′": "5.11"})
    _old_n = pd.to_numeric(_old["n_items"], errors="coerce").fillna(0)
    for _s in _FIRST_PASS:
        _full = _old[(_old_sec == _s) & (_old_n >= N_RELEASED) & (_old["seconds"] >= 60)]
        if len(_full):
            _measured[_s] = _full.iloc[-1]

_n_it = pd.to_numeric(rt["n_items"], errors="coerce").fillna(0)
rel = rt.copy()
if PROFILE == "full":
    for _s, _secs in _FIRST_PASS.items():
        _m = (rel["_sec"] == _s) & (_n_it == 0)          # the step ran, but generated nothing (reused checkpoint)
        if not _m.any():
            continue
        if _s in _measured:
            _r = _measured[_s]
            _vals = [float(_r["seconds"]), _hms(_r["seconds"]), int(float(_r["n_items"])), float(_r["ms_per_item"]),
                     str(_r["hardware"]), "full generation recorded in the log of the reused run; reused in this run"]
        else:
            _vals = [_secs, _hms(_secs), N_RELEASED, round(_secs / N_RELEASED * 1000, 1),
                     "GPU NVIDIA A100-SXM4-80GB 85GB", "first full generation run (2026-08-16); reused in this run"]
        rel.loc[_m, ["step", "seconds", "hms", "n_items", "ms_per_item", "hardware", "note"]] = [_GEN_NAME[_s]] + _vals
_by_sec = rel.groupby("_sec")["seconds"].sum()
_syn = float(_by_sec.get("4.3", 0) + _by_sec.get("5.6", 0))
_dir = float(_by_sec.get("5.10", 0) + _by_sec.get("5.11", 0))
_all = float(rel["seconds"].sum()); _oth = _all - _syn - _dir
print(f"Cost of the release: synthesized path {_hms(_syn + _oth)} ({(_syn + _oth) / N_RELEASED:.1f} s per record); "
      f"all three summary layers {_hms(_all)} ({_all / N_RELEASED:.1f} s per record); other steps {_hms(_oth)}")
rel = rel.drop(columns="_sec").reset_index(drop=True)
rel.to_csv(os.path.join(SAVE_DIR, "runtime_log_release.csv"), index=False)
print("saved -> runtime_log_release.csv (the working runtime_log.csv is not changed)\n")

rt = rt.drop(columns="_sec")
rt["hours"] = (rt["seconds"] / 3600).round(3)
print("Timed steps of this run:")
print(f"{'step':58} {'hh:mm:ss':>10} {'n':>9} {'ms/item':>11}")
print("-" * 92)
for _, r in rt.iterrows():
    n   = "" if pd.isna(r["n_items"])     else f"{int(r['n_items']):,}"
    mpi = "" if pd.isna(r["ms_per_item"]) else f"{r['ms_per_item']:,.1f}"
    print(f"{str(r['step'])[:58]:58} {r['hms']:>10} {n:>9} {mpi:>11}")
print("-" * 92)
print(f"{'total of the timed steps':58} {_hms(rt['seconds'].sum()):>10}")
for hw in rt["hardware"].unique():
    sub = rt[rt["hardware"] == hw]
    print(f"  {hw}: {sub['seconds'].sum()/3600:.2f} h ({len(sub)} steps)")

# The same release table in Markdown, for a README or a report
print("\n| step | wall-clock | items | ms/item | hardware |")
print("|---|---|---|---|---|")
for _, r in rel.iterrows():
    n   = "-" if pd.isna(r["n_items"])     else f"{int(r['n_items']):,}"
    mpi = "-" if pd.isna(r["ms_per_item"]) else f"{r['ms_per_item']:,.1f}"
    print(f"| {r['step']} | {r['hms']} | {n} | {mpi} | {r['hardware']} |")


# ── 9.3  Write the release files to SAVE_DIR/release/ (working files are not changed) ──
# Three layers:
#   Layer 1 source and rule-derived: hpc_patenttrace_corpus.csv, excluded_no_expansion.csv, released_corpus_ids.csv,
#                          evidence_flags.csv, ownership_events.csv, litigation_family.csv
#   Layer 2 model-generated and derived: facets.csv, summaries_synthesized.csv, summaries_direct_brief.csv,
#                          summaries_direct_full.csv, embeddings/, cluster_results.csv, topic_summary.csv, merge_map_worksheet.csv
#   Layer 3 workflow assets and records: release_assets/, runtime_log_release.csv, tables/, usage_*.csv, audit_*.csv
# Normalisation: list columns become JSON; the topic is split into topic_id and topic_label; the facets get
# what_facet_source, sections_usable and must_keep_terms. The family and litigation notes are written in English
# by section 2; the replacements below only convert files of earlier runs. Column definitions: the data dictionary.
import os, re, json, ast, shutil, glob
import numpy as np, pandas as pd

RELEASE_DIR = os.path.join(SAVE_DIR, "release")
for _d in ("embeddings", "tables"):
    os.makedirs(os.path.join(RELEASE_DIR, _d), exist_ok=True)
_manifest = []

def _save(frame, name):
    p = os.path.join(RELEASE_DIR, name)
    frame.to_csv(p, index=False)
    _manifest.append({"file": name, "rows": len(frame), "columns": len(frame.columns),
                      "MB": round(os.path.getsize(p) / 1e6, 2)})
    print(f"  {name:40} {len(frame):>6,} × {len(frame.columns):>3}  {os.path.getsize(p) / 1e6:8.2f} MB")

LIST_COLS = ["ipc_description", "citations", "applications_claiming_priority", "legal_events"]
_LIT_EN   = [("IPR\u8b49\u66f8", "IPR certificate"), ("\u518d\u5be9\u67e5", "reexamination"), ("\u5730\u9662", "district court "), ("\u4ef6(", " cases (")]

def _to_json(x):
    x = "" if pd.isna(x) else str(x)
    if not x.strip():
        return ""
    try:
        return json.dumps(ast.literal_eval(x), ensure_ascii=False)
    except Exception:
        return x                       # already JSON or cannot be parsed: kept as it is

def _normalise(frame):
    out = frame.copy()
    for c in LIST_COLS:
        if c in out.columns:
            out[c] = out[c].map(_to_json)
    if "family" in out.columns:
        out["family"] = out["family"].fillna("").astype(str).str.replace(r"\((\d+)\u4ef6\)", r"(n=\1)", regex=True)
    if "litigation" in out.columns:
        _l = out["litigation"].fillna("").astype(str)
        for k, v in _LIT_EN:
            _l = _l.str.replace(k, v, regex=False)
        out["litigation"] = _l
    return out

def _ipc3(frame):
    return add_ipc3(frame) if "add_ipc3" in globals() else frame.assign(main_ipc_3=frame["main_ipc_4"].astype(str).str[:3])

print("Layer 1 ─ source and rule-derived")
_ids = pd.read_csv(os.path.join(SAVE_DIR, "released_corpus_ids.csv"), dtype=str)["patent_id"]
_idset = set(_ids)
_raw = pd.read_csv(CLEAN_PATH, dtype=str, keep_default_na=False, low_memory=False).drop_duplicates("patent_id")
_main = _raw[_raw["patent_id"].isin(_idset)].reset_index(drop=True)
assert len(_main) == len(_idset) == N_RELEASED, f"the main file has {len(_main):,} records and the identifier list {len(_idset):,}; they do not agree"
_save(_normalise(_ipc3(_main)), "hpc_patenttrace_corpus.csv")

_REQ = ["title", "abstract", "background", "detailed_description", "claims_text"]
_complete = np.logical_and.reduce([_raw[c].str.strip().str.len().gt(0).values for c in _REQ])
_excl = _raw[_complete & ~_raw["patent_id"].isin(_idset)].reset_index(drop=True)
assert len(_excl) == N_EXCLUDED, f"excluded records {len(_excl):,} differ from 3.5 ({N_EXCLUDED:,})"
_save(_normalise(_ipc3(_excl)), "excluded_no_expansion.csv")
del _raw

_save(pd.DataFrame({"patent_id": _ids}), "released_corpus_ids.csv")
for _f in ("evidence_flags.csv", "ownership_events.csv"):
    _t = pd.read_csv(os.path.join(SAVE_DIR, _f), dtype=str, keep_default_na=False)
    assert set(_t["patent_id"]) == _idset, f"the identifiers of {_f} do not agree with the main file"
    _save(_t, _f)
_lf = os.path.join(SAVE_DIR, "litigation_family.csv")
if os.path.exists(_lf):
    _save(_normalise(_ipc3(pd.read_csv(_lf, dtype=str, keep_default_na=False))), "litigation_family.csv")

print("Layer 2 ─ model-generated and derived")
_s1 = pd.read_csv(os.path.join(SAVE_DIR, "stage1_summaries.csv"), dtype=str, keep_default_na=False)
_fp = pd.read_csv(os.path.join(SAVE_DIR, "patent_full_pipeline.csv"), dtype=str, keep_default_na=False,
                  usecols=lambda c: c in {"patent_id", "title", "summary_is_substituted", "sections_usable",
                                          "cot_reasoning", "final_summary"})
assert set(_s1["patent_id"]) == _idset and set(_fp["patent_id"]) == _idset, "the identifiers of the facet or synthesis file do not agree with the main file"
_fac = _s1[["patent_id", "bg_summary", "inv_summary", "tech_summary"]].merge(
    _fp[["patent_id", "title", "summary_is_substituted", "sections_usable"]], on="patent_id", how="left")
_fac["what_facet_source"] = np.where(_fac["summary_is_substituted"].str.lower().eq("true"),
                                     "detailed_description_first_2000_chars", "summary_section")
if "extract_tech_terms" in globals():
    _fac["must_keep_terms"] = [extract_tech_terms(t, b, i, h) for t, b, i, h in
                               zip(_fac["title"], _fac["bg_summary"], _fac["inv_summary"], _fac["tech_summary"])]
else:
    _fac["must_keep_terms"] = ""
    print("  (extract_tech_terms not defined: must_keep_terms left empty; rerun this cell in a session that ran 4.3)")
_save(_fac[["patent_id", "bg_summary", "inv_summary", "tech_summary",
            "what_facet_source", "sections_usable", "must_keep_terms"]], "facets.csv")
_save(_fp[["patent_id", "cot_reasoning", "final_summary"]], "summaries_synthesized.csv")
_db = pd.read_csv(os.path.join(SAVE_DIR, "baseline_direct.csv"), dtype=str, keep_default_na=False) \
        .rename(columns={"baseline_summary": "direct_brief_summary"})
_save(_db[["patent_id", "direct_brief_summary"]], "summaries_direct_brief.csv")
_dfu = pd.read_csv(os.path.join(SAVE_DIR, "baseline_direct_matched.csv"), dtype=str, keep_default_na=False) \
        .rename(columns={"baseline_matched_summary": "direct_full_summary"})
_save(_dfu[["patent_id", "direct_full_summary"]], "summaries_direct_full.csv")

_cl = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv"), dtype=str, keep_default_na=False)
_SENT = globals().get("NON_TOPIC_SENTINEL", "__noise__")
_RN = globals().get("RENAME", {}) or {}
_lab = _cl["merged_topic"].replace(_RN)
_sizes = _lab[_lab != _SENT].value_counts()                 # sorted by record count -> topic_id 0, 1, ...
_tid = {t: i for i, t in enumerate(_sizes.index)}
_cl["topic_id"] = _lab.map(_tid).fillna(-1).astype(int)
_cl["topic_label"] = _lab
_save(_cl, "cluster_results.csv")
_ts = pd.read_csv(os.path.join(SAVE_DIR, "topic_summary.csv"), dtype=str, keep_default_na=False)
_ts["topic_label"] = _ts["merged_topic"].replace(_RN)
_ts["topic_id"] = _ts["topic_label"].map(_tid).fillna(-1).astype(int)
_save(_ts[["topic_id", "topic_label"] + [c for c in _ts.columns if c not in ("topic_id", "topic_label", "merged_topic")]],
      "topic_summary.csv")
_mw = pd.read_csv(os.path.join(SAVE_DIR, "merge_map_worksheet.csv"), dtype=str, keep_default_na=False)
_mw["topic_label"] = _mw["auto_merged_topic"].replace(_RN)
_mw["topic_id"] = _mw["topic_label"].map(_tid).fillna(-1).astype(int)
_save(_mw, "merge_map_worksheet.csv")
for _p in sorted(glob.glob(os.path.join(SAVE_DIR, "embeddings", "*"))):
    _ = shutil.copy2(_p, os.path.join(RELEASE_DIR, "embeddings", os.path.basename(_p)))
    _manifest.append({"file": "embeddings/" + os.path.basename(_p), "rows": N_RELEASED if _p.endswith(".parquet") else "",
                      "columns": 769 if _p.endswith(".parquet") else "", "MB": round(os.path.getsize(_p) / 1e6, 2)})
print(f"  embeddings/ copied {len(glob.glob(os.path.join(RELEASE_DIR, 'embeddings', '*')))} files")

print("Layer 3 ─ workflow assets and records")
for _p in sorted(glob.glob(os.path.join(SAVE_DIR, "tables", "*.csv"))):
    _ = shutil.copy2(_p, os.path.join(RELEASE_DIR, "tables", os.path.basename(_p)))
print(f"  tables/ copied {len(glob.glob(os.path.join(RELEASE_DIR, 'tables', '*.csv')))} files")
for _pat in ("usage_*.csv", "audit_*.csv", "runtime_log_release.csv", "hdbscan_param_sweep.csv",
             "differential_exclusion.csv", "ipc_downstream_comparison.csv", "ipc_downstream_bootstrap.csv",
             "summary_metrics.csv", "pairing_integrity_flags.csv", "added_value_beyond_abstract.csv",
             "coverage_beyond_truncation.csv", "overlap_structure.csv", "not_in_full_text_per_record.csv",
             "ppl_summaries.csv", "truncation_profile.csv", "topic_classifier_benchmark.csv",
             "developer_demo_comparison.csv", "intra_topic_similarity.csv"):
    for _p in sorted(glob.glob(os.path.join(SAVE_DIR, _pat))):
        _ = shutil.copy2(_p, os.path.join(RELEASE_DIR, os.path.basename(_p)))
        _manifest.append({"file": os.path.basename(_p), "rows": "", "columns": "", "MB": round(os.path.getsize(_p) / 1e6, 2)})
if not os.path.exists(os.path.join(RELEASE_DIR, "runtime_log_release.csv")):
    print("  Note: runtime_log_release.csv does not exist; run 9.2 first, then this cell again")
_assets = os.path.join(SAVE_DIR, "release_assets")
if os.path.isdir(_assets):
    _ = shutil.copytree(_assets, os.path.join(RELEASE_DIR, "release_assets"), dirs_exist_ok=True)
    print("  release_assets/ copied (prompts, terminology, config)")
else:
    print("  Note: release_assets/ does not exist; run 9.1 first, then this cell again")

pd.DataFrame(_manifest).to_csv(os.path.join(RELEASE_DIR, "MANIFEST_release.csv"), index=False)
print(f"\nrelease -> {RELEASE_DIR}  ({len(_manifest)} items, list in MANIFEST_release.csv)")


# ── 9.4  Release names: every ipc in a file or column name becomes cpc ─────────
# The class codes are CPC codes (Cooperative Patent Classification paths of Google Patents). 9.3 wrote release/
# with the working names; this part renames ipc to cpc in the file names, the column names, the JSON keys of
# cpc_description and three columns that list field or step names (hpc_patenttrace/names.py). Each converted file
# is checked against the file it replaces: same rows, same identifiers, same class codes and the same values in
# every other column. release/NAME_MAPPING_ipc_to_cpc.csv lists every old and new name. The working files of
# SAVE_DIR keep their names. The same step can be run alone on the release/ folder of an earlier run:
#   python -m hpc_patenttrace.names <release folder>
from hpc_patenttrace.names import convert_release, MAPPING_FILE

NAME_CHECK = convert_release(RELEASE_DIR)
print(f"{len(NAME_CHECK)} files converted to the CPC names and checked against the originals; "
      f"old and new names in release/{MAPPING_FILE}")


# ── 9.5  Files written to SAVE_DIR ──────────────────────────────────────────
_rows = []
for _root, _dirs, _files in os.walk(SAVE_DIR):
    for _f in _files:
        _p = os.path.join(_root, _f)
        _rows.append({"folder": os.path.relpath(_root, SAVE_DIR), "file": _f, "MB": round(os.path.getsize(_p) / 1e6, 2)})
files_df = pd.DataFrame(_rows).sort_values(["folder", "file"]).reset_index(drop=True)
print(f"{SAVE_DIR}: {len(files_df):,} files, {files_df['MB'].sum()/1e3:.2f} GB")
files_df.groupby("folder").agg(files=("file", "size"), MB=("MB", "sum")).round(1)
files_df[files_df["folder"] == "."]
