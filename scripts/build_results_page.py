#!/usr/bin/env python3
"""Build the results pages of the HPC-PatentTrace release (for GitHub Pages).

Every number on the pages is read from release/ in the output folder of a full run of the step 6 pipeline
notebook (colab/step6_patentsum_pipeline.ipynb, folder OUTPUT_DIR/full). The script writes
  docs/index.html         English page
  docs/index_tw.html      Traditional Chinese page
  docs/assets/data.js     the values that the charts and the record viewer use (shared by both pages)
docs/assets/results.js and docs/assets/results.css are static files of the pages.

Usage:
  uv run python scripts/build_results_page.py --run path/to/output/full
  uv run python scripts/build_results_page.py --run path/to/output/full --out docs
"""
import argparse
import html
import json
import os

import pandas as pd

LAYERS = ["synthesized", "direct-brief", "direct-full"]
REPS = ["abstract", "fulltext-truncated", "direct-brief", "direct-full", "synthesized"]
SECTION_ORDER = ["G", "H", "A", "C", "B", "F", "E", "D"]
CHANGE_TYPES = ["true_assignment", "change_of_name", "merger", "no_assignment_record", "security_interest",
                "release_of_lien", "corrective", "partial_transfer", "other"]
# Example records: drawn at random (seed 42) from the released records of each class section
EXAMPLE_SECTIONS = [("G", 3), ("H", 2), ("other", 1)]
# Section numbers of the four generation steps in runtime_log_release.csv (sections 4 and 5 of the pipeline)
GENERATION_STEPS = [("4.3", "facets"), ("5.6", "synthesized"), ("5.10", "direct-brief"), ("5.11", "direct-full")]


class Release:
    """Reads the files of release/ in the output folder of the pipeline notebook."""

    def __init__(self, path):
        sub = os.path.join(path, "release")
        self.prefix = sub if os.path.isdir(sub) else path
        if not os.path.isfile(os.path.join(self.prefix, "MANIFEST_release.csv")):
            raise FileNotFoundError(f"{path} holds no release/MANIFEST_release.csv (section 9 of the pipeline writes it)")

    def csv(self, rel, **kw):
        return pd.read_csv(os.path.join(self.prefix, rel), **kw)


def collect(rel):
    """Read the release files and return (data for the charts, facts for the text)."""
    D, F = {}, {}

    # ── corpus construction ──
    funnel = rel.csv("tables/funnel_from_crawler_output.csv")      # crawler output, candidates, complete, released
    n_query, n_cand, n_complete, n_corpus = (int(v) for v in funnel["records"])
    ids = rel.csv("released_corpus_ids.csv")
    assert len(ids) == n_corpus, "funnel_from_crawler_output.csv and released_corpus_ids.csv disagree"
    F.update(n_query=n_query, n_cand=n_cand, n_homonym=n_query - n_cand, n_complete=n_complete,
             n_incomplete=n_cand - n_complete, n_corpus=n_corpus, n_excluded=n_complete - n_corpus)
    D["funnel"] = [n_query, n_cand, n_complete, n_corpus]

    comp = rel.csv("tables/completeness_bias_by_ipc_section.csv")  # section, before, after, removed, rate, ...
    comp = comp.set_index(comp.columns[0])
    excl = rel.csv("differential_exclusion.csv").set_index("ipc_section")
    D["removal"] = {"sections": SECTION_ORDER,
                    "completeness": [float(comp.loc[s].iloc[3]) for s in SECTION_ORDER],
                    "inclusion": [float(excl.loc[s, "exclusion_rate_pct"]) for s in SECTION_ORDER],
                    "candidates": [int(comp.loc[s].iloc[0]) for s in SECTION_ORDER],
                    "complete": [int(excl.loc[s, "before"]) for s in SECTION_ORDER]}
    for s in ("G", "H", "A", "C"):
        F[f"comp_{s}"] = float(comp.loc[s].iloc[3])
        F[f"excl_{s}"] = float(excl.loc[s, "exclusion_rate_pct"])

    miss = rel.csv("tables/completeness_missing_by_field.csv")      # field, missing, share of all, share of removed
    D["missing"] = [{"field": str(r.iloc[0]), "n": int(r.iloc[1]), "share": float(r.iloc[2]),
                     "share_removed": float(r.iloc[3])} for _, r in miss.iterrows()]
    _m = {d["field"]: d for d in D["missing"]}
    F.update(miss_dd=_m["detailed_description"]["share"], miss_bg=_m["background"]["share"],
             miss_cl=_m["claims_text"]["share"])

    # ── composition ──
    sec = rel.csv("tables/t5a_ipc_section.csv")                    # section, records, share
    D["sections"] = [{"s": str(r.iloc[0]), "n": int(r.iloc[1]), "share": float(r.iloc[2])} for _, r in sec.iterrows()]
    F["share_GH"] = sum(d["share"] for d in D["sections"] if d["s"] in ("G", "H"))
    yrs = rel.csv("tables/t4_application_year.csv")                # year, records, share, note (right truncation)
    D["years"] = [{"y": int(r.iloc[0]), "n": int(r.iloc[1]), "share": float(r.iloc[2]),
                   "trunc": bool(pd.notna(r.iloc[3]))} for _, r in yrs.iterrows()]
    F["trunc_years"] = [d["y"] for d in D["years"] if d["trunc"]]
    top = rel.csv("tables/t6_top20_assignee.csv")                  # rank, assignee, records, share
    D["assignees"] = [{"rank": int(r.iloc[0]), "name": str(r.iloc[1]), "n": int(r.iloc[2]), "share": float(r.iloc[3])}
                      for _, r in top.head(10).iterrows()]
    F["top1_name"], F["top1_n"] = D["assignees"][0]["name"], D["assignees"][0]["n"]
    F["top1_share"] = D["assignees"][0]["share"]
    F["top10_share"] = round(sum(d["n"] for d in D["assignees"]) / n_corpus * 100, 1)

    # ── summary layers ──
    sm = rel.csv("summary_metrics.csv").set_index("system")
    D["layers"] = {l: {"words": float(sm.loc[l, "mean_summary_words"]), "compression": float(sm.loc[l, "compression_ratio"]),
                       "rouge1": float(sm.loc[l, "rouge1_f"]), "rouge2": float(sm.loc[l, "rouge2_f"]),
                       "rougeL": float(sm.loc[l, "rougeL_f"]), "rouge1_shift": float(sm.loc[l, "rouge1_permuted"]),
                       "not_better": int(sm.loc[l, "n_not_better_than_permuted"]), "low": int(sm.loc[l, "n_low_rouge1"])}
                   for l in LAYERS}
    flags = rel.csv("pairing_integrity_flags.csv")
    F["n_flags"] = len(flags)
    F["n_flags_syn"] = int((flags["system"] == "synthesized").sum())
    F["rouge_own_min"] = min(v["rouge1"] for v in D["layers"].values())
    F["rouge_own_max"] = max(v["rouge1"] for v in D["layers"].values())
    F["rouge_shift_min"] = min(v["rouge1_shift"] for v in D["layers"].values())
    F["rouge_shift_max"] = max(v["rouge1_shift"] for v in D["layers"].values())

    ppl = rel.csv("ppl_summaries.csv")
    _pc = {"synthesized": "ppl_synthesized", "direct-brief": "ppl_direct_brief", "direct-full": "ppl_direct_full"}
    D["ppl"] = {l: [round(float(v), 1) for v in ppl[_pc[l]].dropna()] for l in LAYERS}
    F["ppl_mean"] = {l: float(ppl[_pc[l]].mean()) for l in LAYERS}
    F["ppl_max"] = float(ppl[list(_pc.values())].max().max())

    av = rel.csv("added_value_beyond_abstract.csv")
    F["av_candidates"] = float(av["n_candidate"].mean())
    D["added"] = {l: {"mean": float(av[f"{l}_added"].mean()), "any": float((av[f"{l}_added"] > 0).mean() * 100)}
                  for l in LAYERS}

    # Example records (EXAMPLE_SECTIONS)
    rec = rel.csv("hpc_patenttrace_corpus.csv", usecols=["patent_id", "title", "abstract", "patent_url", "main_ipc_4"])
    rec["section"] = rec["main_ipc_4"].fillna("").str[:1].where(lambda s: s.isin(["G", "H"]), "other")
    syn = rel.csv("summaries_synthesized.csv").set_index("patent_id")["final_summary"]
    brief = rel.csv("summaries_direct_brief.csv").set_index("patent_id")["direct_brief_summary"]
    full = rel.csv("summaries_direct_full.csv").set_index("patent_id")["direct_full_summary"]
    D["examples"] = []
    for section, k in EXAMPLE_SECTIONS:
        for _, r in rec[rec["section"] == section].sample(k, random_state=42).iterrows():
            D["examples"].append({"id": r["patent_id"], "title": r["title"], "url": r["patent_url"],
                                  "abstract": r["abstract"], "synthesized": syn[r["patent_id"]],
                                  "direct-brief": brief[r["patent_id"]], "direct-full": full[r["patent_id"]]})

    # ── topics ──
    clu = rel.csv("cluster_results.csv")
    assert len(clu) == n_corpus
    ts = rel.csv("topic_summary.csv")
    noise = ts[ts["topic_id"] == -1]
    ts = ts[ts["topic_id"] != -1].sort_values("patent_count", ascending=False)
    D["topics"] = [{"id": int(r["topic_id"]), "label": r["topic_label"], "n": int(r["patent_count"]),
                    "ipc": r["top_ipc"], "company": r["top_company"]} for _, r in ts.iterrows()]
    F.update(n_topics=len(ts), n_topic_records=int(ts["patent_count"].sum()), n_noise=int(noise["patent_count"].iloc[0]),
             n_clusters=int(clu.loc[clu["cluster_id"] != -1, "cluster_id"].nunique()))
    assert F["n_topic_records"] + F["n_noise"] == n_corpus
    F["topic_share"] = F["n_topic_records"] / n_corpus * 100
    F["noise_share"] = F["n_noise"] / n_corpus * 100
    sec_code = clu["main_ipc_4"].fillna("?").astype(str).str[0].map({"G": 0, "H": 1}).fillna(2).astype(int)
    D["map"] = {"x": clu["umap_3d_x"].round(2).tolist(), "y": clu["umap_3d_y"].round(2).tolist(),
                "z": clu["umap_3d_z"].round(2).tolist(), "topic": clu["topic_id"].astype(int).tolist(),
                "section": sec_code.tolist(),
                "title": [t if len(t) <= 90 else t[:87] + "..." for t in clu["title"].fillna("").astype(str)],
                "who": clu["assignee_display"].fillna("").astype(str).tolist(),
                "ipc": clu["main_ipc_4"].fillna("").astype(str).tolist()}

    met = rel.csv("tables/e6_topic_ipc_metrics.csv")               # measure, value, note (by position)
    _names = [str(v) for v in met.iloc[:, 0]]
    assert "NMI" in _names[0] and "ARI" in _names[1] and "homogeneity" in _names[2] and "completeness" in _names[3]
    F.update(nmi=float(met.iloc[0, 1]), ari=float(met.iloc[1, 1]), homog=float(met.iloc[2, 1]), compl=float(met.iloc[3, 1]))
    tr = rel.csv("usage_topic_trends.csv")
    D["trends"] = {"years": tr["application_year"].astype(int).tolist(), "topics": list(tr.columns[1:]),
                   "values": [tr[c].round(2).tolist() for c in tr.columns[1:]]}

    # ── ownership and litigation ──
    own = rel.csv("ownership_events.csv")
    ch = own[own["assignee_changed"]]
    F.update(n_changed=len(ch), changed_share=len(ch) / n_corpus * 100, n_supported=int(own["change_supported"].sum()),
             n_ambiguous=int(own["party_ambiguous"].sum()),
             n_effective=int((own["effective_assignee"] != own["current_assignee"]).sum()))
    F["supported_share"] = F["n_supported"] / F["n_changed"] * 100
    F["effective_share"] = F["n_effective"] / n_corpus * 100
    vc = ch["conveyance_type"].value_counts()
    F.update(n_no_record=int(vc.get("no_assignment_record", 0)),
             n_security=int(vc.get("security_interest", 0) + vc.get("release_of_lien", 0)),
             n_corrective=int(vc.get("corrective", 0)),
             n_other_type=int(vc.get("partial_transfer", 0) + vc.get("other", 0)))
    D["conveyance"] = [{"type": t, "n": int(vc.get(t, 0))} for t in CHANGE_TYPES if vc.get(t, 0) > 0]
    assert sum(c["n"] for c in D["conveyance"]) == len(ch)
    ba = rel.csv("usage_assignee_before_after.csv")
    D["before_after"] = [{"rank": int(r["rank"]), "before": r["current_assignee"], "n_before": int(r["n_before"]),
                          "after": r["effective_assignee"], "n_after": int(r["n_after"]), "changed": r["changed"] == "yes"}
                         for _, r in ba.iterrows()]
    lf = rel.csv("litigation_family.csv")
    D["litigation"] = [{"id": r["patent_id"], "group": r["case_group"], "seed": bool(r["is_seed"]),
                        "text": "" if pd.isna(r["litigation"]) else r["litigation"]} for _, r in lf.iterrows()]
    lt = rel.csv("usage_litigation_topics.csv")
    lt = lt[lt["n_litigated"] > 0].sort_values("n_litigated", ascending=False)
    D["litigation_topics"] = [{"topic": r["merged_topic"], "n": int(r["n_litigated"]), "of": int(r["n_patents"])}
                              for _, r in lt.iterrows()]
    F.update(n_lit=len(lf), n_seed=int(lf["is_seed"].sum()), n_fam=int(lf["family_root"].nunique()),
             n_groups=int(lf["case_group"].nunique()), n_lit_topics=len(lt))

    # ── technical validation ──
    cpc = rel.csv("ipc_downstream_comparison.csv").set_index("representation")
    D["cpc"] = {r: float(cpc.loc[r, "ml_f1_micro"]) for r in REPS}
    F.update(cpc_labels=int(cpc["n_labels"].iloc[0]), cpc_test=int(cpc["n_test"].iloc[0]),
             cpc_min=min(D["cpc"].values()), cpc_max=max(D["cpc"].values()))
    boot = rel.csv("ipc_downstream_bootstrap.csv")
    D["boot"] = [{"pair": r["comparison"].replace(" - ", " − "), "est": float(r["point_estimate"]),
                  "lo": float(r["ci_low"]), "hi": float(r["ci_high"])} for _, r in boot.iterrows()]
    F["n_boot"] = int(boot["n_boot"].iloc[0])
    _sa = boot[boot["comparison"] == "synthesized - abstract"].iloc[0]
    F.update(sa_est=float(_sa["point_estimate"]), sa_lo=float(_sa["ci_low"]), sa_hi=float(_sa["ci_high"]))
    dev = rel.csv("developer_demo_comparison.csv").set_index("representation")
    D["applicant"] = {"acc": {r: float(dev.loc[r, "accuracy"]) for r in REPS},
                      "majority": float(dev["majority_share"].iloc[0])}
    F.update(app_classes=int(dev["n_classes"].iloc[0]), app_min=min(D["applicant"]["acc"].values()),
             app_max=max(D["applicant"]["acc"].values()), app_major=D["applicant"]["majority"] * 100)
    D["pairing"] = {l: {"own": D["layers"][l]["rouge1"], "shift": D["layers"][l]["rouge1_shift"]} for l in LAYERS}
    intra = rel.csv("intra_topic_similarity.csv")
    D["intra"] = {r: float(intra[r].mean()) for r in REPS}

    # ── cost ──
    rt = rel.csv("runtime_log_release.csv")
    _sec = rt["step"].astype(str).str.split().str[0]
    D["cost"] = []
    for code, name in GENERATION_STEPS:
        r = rt[_sec == code].iloc[0]
        D["cost"].append({"step": name, "seconds": float(r["seconds"]), "n": int(r["n_items"]), "hw": r["hardware"]})
    F["gen_seconds"] = sum(c["seconds"] for c in D["cost"])
    # the other steps are summed only when the whole log comes from the hardware of the generation steps
    # (a run that reuses generated layers logs the generation steps of the earlier run)
    _hw = {c["hw"] for c in D["cost"]}
    F["other_seconds"] = float(rt["seconds"].sum()) - F["gen_seconds"] if set(rt["hardware"].dropna()) == _hw else None

    # ── files ──
    man = rel.csv("MANIFEST_release.csv")
    D["files"] = [{"file": r["file"], "rows": None if pd.isna(r["rows"]) else int(r["rows"]),
                   "cols": None if pd.isna(r["columns"]) else int(r["columns"]), "mb": float(r["MB"])}
                  for _, r in man.iterrows()]
    return D, F


# ── page text ────────────────────────────────────────────────────────────────

def n(x):
    return f"{int(round(x)):,}"


def p1(x):
    return f"{x:.1f}"


def p3(x):
    return f"{x:.3f}"


def hours(sec):
    return f"{sec / 3600:.1f}"


CHANGE_NAME = {
    "en": {"true_assignment": "assignment", "change_of_name": "change of name", "merger": "merger",
           "no_assignment_record": "no assignment record", "security_interest": "security interest",
           "release_of_lien": "release of a security interest", "corrective": "corrective record",
           "partial_transfer": "partial transfer", "other": "other"},
    "tw": {"true_assignment": "讓與", "change_of_name": "更名", "merger": "合併",
           "no_assignment_record": "沒有讓與紀錄", "security_interest": "擔保權益",
           "release_of_lien": "擔保權益解除", "corrective": "更正紀錄",
           "partial_transfer": "部分讓與", "other": "其他"},
}

FIELD_NAME = {"en": {"detailed_description": "detailed description", "background": "background",
                     "claims_text": "claims", "abstract": "abstract", "title": "title"},
              "tw": {"detailed_description": "詳細說明（detailed description）", "background": "先前技術（background）",
                     "claims_text": "請求項（claims）", "abstract": "摘要（abstract）", "title": "標題（title）"}}

STEP_NAME = {"en": {"facets": "facets (Llama-3.1-8B-Instruct, 4-bit)", "synthesized": "synthesized (Qwen2.5-14B-Instruct)",
                    "direct-brief": "direct-brief (Qwen2.5-14B-Instruct)", "direct-full": "direct-full (Qwen2.5-14B-Instruct)"},
             "tw": {"facets": "分面（Llama-3.1-8B-Instruct，4-bit）", "synthesized": "synthesized（Qwen2.5-14B-Instruct）",
                    "direct-brief": "direct-brief（Qwen2.5-14B-Instruct）", "direct-full": "direct-full（Qwen2.5-14B-Instruct）"}}


def texts(lang, F):
    """Section text. Numbers come from F."""
    if lang == "en":
        return {
            "html_lang": "en", "title": "High-performance computing patent dataset with full text, structured summaries and semantic embeddings",
            "kicker": "HPC-PatentTrace v1.0 · results of the released dataset",
            "switch": ("index_tw.html", "中文"),
            "lead": (f"HPC-PatentTrace covers {n(F['n_corpus'])} United States granted patents, filed from 2015 to 2025, "
                     "that mention high-performance computing. This page shows how the corpus was built and what the "
                     "release contains."),
            "source": ("Every number below is read from release/ in the output folder of the step 6 pipeline notebook "
                       "(colab/step6_patentsum_pipeline.ipynb) by scripts/build_results_page.py of this repository."),
            "nav": ["Overview", "Corpus", "Composition", "Summaries", "Topics", "Ownership", "Validation", "Cost", "Files"],
            "tiles": [(n(F["n_corpus"]), "granted US patents in the corpus"), ("2015–2025", "filing years"),
                      ("3", "summary layers per record"), ("5", "text representations with SPECTER2 embeddings"),
                      (n(F["n_topics"]), f"topics, covering {n(F['n_topic_records'])} records"),
                      (n(F["n_lit"]), "records in litigated patent families")],
            "corpus_h": "Corpus construction",
            "corpus_p": (f"The query returned {n(F['n_query'])} publication numbers. The homonym screening removes "
                         f"{n(F['n_homonym'])} records in which HPC stands for another term, for example hydroxypropyl "
                         f"cellulose or a high-pressure compressor. The completeness filter keeps the {n(F['n_complete'])} "
                         "candidates whose title, abstract, background, detailed description and claims were all "
                         f"extracted. The inclusion rule then keeps the {n(F['n_corpus'])} records whose text contains the "
                         "expanded phrase high performance comput*."),
            "corpus_note": f"{n(F['n_query'])} is the number of records in the crawler output (step 5), one per publication number.",
            "removal_h": "Where the rules remove records",
            "removal_p": (f"Both rules keep most records of sections G (physics, including computing) and H (electricity) "
                          f"and remove most records of the other sections. The completeness filter removes "
                          f"{p1(F['comp_G'])} percent of the G candidates and {p1(F['comp_H'])} percent of H, against "
                          f"{p1(F['comp_A'])} percent of A and {p1(F['comp_C'])} percent of C. The inclusion rule then "
                          f"removes {p1(F['excl_G'])} percent of G and {p1(F['excl_H'])} percent of H, against "
                          f"{p1(F['excl_A'])} percent of A and {p1(F['excl_C'])} percent of C, the sections where the "
                          "other meanings of HPC belong."),
            "missing_p": (f"A candidate is removed by the completeness filter mostly because its detailed description "
                          f"({p1(F['miss_dd'])} percent of all candidates) or its background ({p1(F['miss_bg'])} percent) "
                          "was not extracted from the patent page."),
            "missing_cols": ["section of the patent", "candidates without it", "share of all candidates (%)",
                             "share of removed candidates (%)"],
            "comp_h": "Composition of the corpus",
            "comp_p": (f"Sections G and H hold {p1(F['share_GH'])} percent of the corpus. {F['top1_name']} holds the most "
                       f"records ({n(F['top1_n'])}, {p1(F['top1_share'])} percent), and the ten largest assignees hold "
                       f"{p1(F['top10_share'])} percent. Only granted patents are included, so the years "
                       f"{F['trunc_years'][0]} to {F['trunc_years'][-1]} are right-truncated: many of their applications "
                       "are still under examination."),
            "assignee_cols": ["rank", "current assignee", "records", "share (%)"],
            "sum_h": "Summary layers",
            "sum_p": ("Each record has three facets (background, invention, technical detail), written by "
                      "Llama-3.1-8B-Instruct, and three summaries, written by Qwen2.5-14B-Instruct with the same final "
                      "call. The synthesized layer is written from the title, the abstract and the facets after a "
                      "reasoning step. direct-brief reads short excerpts of the sections in one prompt; direct-full reads "
                      "the same section budgets as the facets in one prompt."),
            "sum_metric_cols": ["layer", "mean words", "compression", "ROUGE-1 F1 (own)", "ROUGE-1 F1 (shifted)",
                                "ROUGE-L F1 (own)"],
            "example_h": "Example records",
            "example_p": ("Six released records drawn at random (seed 42): three of section G, two of section H and one "
                          "of the other sections."),
            "example_labels": {"abstract": "Abstract (from the patent)", "synthesized": "synthesized",
                               "direct-brief": "direct-brief", "direct-full": "direct-full", "open": "Google Patents"},
            "added_p": (f"The summaries carry content that the abstract does not have. A candidate is a long technical "
                        f"word (12 or more letters) that occurs in the specification but not in the title or abstract; "
                        f"a record has {F['av_candidates']:.1f} candidates on average."),
            "added_cols": ["layer", "candidates in the summary (mean per record)", "records with at least one (%)"],
            "ppl_p": (f"GPT-2 perplexity of each summary. The means are {p1(F['ppl_mean']['synthesized'])} for "
                      f"synthesized, {p1(F['ppl_mean']['direct-brief'])} for direct-brief and "
                      f"{p1(F['ppl_mean']['direct-full'])} for direct-full; the highest single value is "
                      f"{p1(F['ppl_max'])}."),
            "topic_h": "Topics",
            "topic_p": (f"SPECTER2 encodes the title and the synthesized summary of each record. UMAP reduces the vectors "
                        f"to three dimensions, HDBSCAN finds {n(F['n_clusters'])} clusters, and the clusters are merged "
                        f"into {n(F['n_topics'])} topics that cover {n(F['n_topic_records'])} records "
                        f"({p1(F['topic_share'])} percent); {n(F['n_noise'])} records ({p1(F['noise_share'])} percent) "
                        "belong to no cluster. Each point of the map is one record. Choose a topic to highlight it, "
                        "or colour the records by class section; drag to rotate."),
            "map_controls": {"mode": "Colour", "topic": "highlight one topic", "section": "class section",
                             "pick": "Topic"},
            "topic_cpc_p": (f"The topics agree with the main four-character class of the records clearly above chance "
                            f"and far from completely: normalised mutual information {p3(F['nmi'])}, adjusted Rand "
                            f"index {p3(F['ari'])}, homogeneity {p3(F['homog'])} and completeness {p3(F['compl'])}. The "
                            "topics are a semantic index next to the patent classification, not a replacement for it."),
            "topic_cols": ["topic", "records", "largest classes", "largest assignees"],
            "trend_p": ("Share of the eight largest topics in each application year (percent of the records of the "
                        "eight topics)."),
            "own_h": "Ownership and litigation",
            "own_p": (f"For {n(F['n_changed'])} records ({p1(F['changed_share'])} percent) the current assignee on the "
                      "patent page differs from the original assignee. The latest assignment record transfers ownership "
                      f"(assignment, change of name or merger) in {n(F['n_supported'])} of them "
                      f"({p1(F['supported_share'])} percent). Of the others, {n(F['n_no_record'])} have no assignment "
                      f"record, and the latest record of {n(F['n_security'])} is a security interest or its release, of "
                      f"{n(F['n_corrective'])} a correction and of {n(F['n_other_type'])} another type. "
                      "ownership_events.csv gives these fields for every record. Its column effective_assignee keeps "
                      "the original assignee when no "
                      f"transfer supports the change; with it, {n(F['n_effective'])} records "
                      f"({p1(F['effective_share'])} percent) change owner."),
            "conv_label": "latest assignment record of the records whose assignee changed",
            "ba_p": "The ten largest assignees by current_assignee and by effective_assignee:",
            "ba_cols": ["rank", "current_assignee", "records", "effective_assignee", "records"],
            "lit_p": (f"litigation_family.csv lists {n(F['n_lit'])} records: {n(F['n_seed'])} patents with their own "
                      "record of a district-court case, an inter partes review or a reexamination, and the other "
                      f"members of their patent families. They form {n(F['n_fam'])} families in {n(F['n_groups'])} "
                      f"case groups and fall in {n(F['n_lit_topics'])} topics."),
            "lit_cols": ["patent", "case group", "own record", "litigation record"],
            "lit_topic_cols": ["topic", "records in litigated families", "records of the topic"],
            "yes": "yes", "no": "",
            "val_h": "Technical validation",
            "pair_p": (f"Pairing check. Each summary is scored against the abstract of its own record and against the "
                       f"abstract shifted by one position. ROUGE-1 F1 is {p3(F['rouge_own_min'])} to "
                       f"{p3(F['rouge_own_max'])} against the own abstract and {p3(F['rouge_shift_min'])} to "
                       f"{p3(F['rouge_shift_max'])} against the shifted one. Over the three layers, {n(F['n_flags'])} "
                       f"summaries ({n(F['n_flags_syn'])} of them synthesized) score no higher against their own "
                       "abstract than against the shifted one, or below 0.10; they are listed in "
                       "pairing_integrity_flags.csv."),
            "cpc_p": (f"CPC classification. A logistic regression on each unit-length embedding predicts the "
                      f"four-character CPC codes of a record ({n(F['cpc_labels'])} codes with at least 20 records, "
                      f"multi-label), with a split that keeps patent families together ({n(F['cpc_test'])} test "
                      f"records). All five representations reach a micro-F1 between {p3(F['cpc_min'])} and "
                      f"{p3(F['cpc_max'])}; the synthesized layer is {p3(F['sa_est'])} above the abstract "
                      f"(paired bootstrap with {n(F['n_boot'])} draws, 95 percent interval {p3(F['sa_lo'])} to "
                      f"{p3(F['sa_hi'])})."),
            "app_p": (f"Original assignee. The same classifier predicts the original assignee among the "
                      f"{n(F['app_classes'])} assignees with at least 20 records. Accuracy stays between "
                      f"{p3(F['app_min'])} and {p3(F['app_max'])}, against {p1(F['app_major'])} percent for always "
                      "choosing the largest assignee. This is a usage example, not evidence for one representation."),
            "intra_p": "Mean cosine similarity of the records of the same topic, for each representation:",
            "intra_cols": ["representation", "mean similarity within topics"],
            "cost_h": "Computing cost",
            "cost_p": (f"Generating the facets and the three summary layers took {hours(F['gen_seconds'])} hours"
                       + (f"; all other logged steps of the run took {hours(F['other_seconds'])} hours on the same hardware"
                          if F["other_seconds"] is not None else "")
                       + ". The table gives the hardware of each step (runtime_log_release.csv)."),
            "cost_cols": ["generation step", "records", "hours", "seconds per record", "hardware"],
            "files_h": "Files of the release",
            "files_p": "MANIFEST_release.csv lists the files of release/ with their size:",
            "files_cols": ["file", "rows", "columns", "MB"],
            "sources": "Files:",
            "footer": ("HPC-PatentTrace v1.0. Page generated from release/ of the pipeline output by scripts/build_results_page.py. "
                       "The crawler and the step 6 notebooks are in this repository."),
            "chart": {
                "funnel": ["query results", "after homonym screening", "five complete sections", "released corpus"],
                "records": "records", "percent_removed": "percent of the section removed",
                "completeness": "completeness filter", "inclusion": "inclusion rule", "section": "class section",
                "year": "application year", "truncated": "right-truncated (applications still pending)",
                "ppl": "GPT-2 perplexity", "share_pct": "share of the eight topics (%)",
                "own": "own abstract", "shifted": "shifted abstract", "rouge1": "ROUGE-1 F1",
                "microf1": "micro-F1 (CPC codes)", "accuracy": "accuracy (original assignee)",
                "majority": "largest assignee always chosen", "difference": "difference in micro-F1",
                "noise": "no cluster", "other_topics": "other topics", "sec_G": "G (physics)",
                "sec_H": "H (electricity)", "sec_other": "other sections", "topic": "topic",
                "records_of": "records",
            },
        }
    return {
        "html_lang": "zh-Hant", "title": "具全文、結構化摘要與語意向量的高效能運算專利資料集",
        "kicker": "HPC-PatentTrace v1.0 · 釋出資料集的成果",
        "switch": ("index.html", "English"),
        "lead": (f"HPC-PatentTrace 收錄 {n(F['n_corpus'])} 件提及高效能運算（high-performance computing）的美國核准專利，"
                 "申請年為 2015 到 2025 年。這一頁說明語料怎麼建立，以及釋出檔裡有什麼。"),
        "source": ("以下所有數字都由本 repository 的 scripts/build_results_page.py 讀取第六步 pipeline notebook"
                   "（colab/step6_patentsum_pipeline_tw.ipynb）輸出資料夾裡的 release/ 產生。"),
        "nav": ["總覽", "語料建立", "語料組成", "摘要層", "主題", "權屬", "技術驗證", "計算成本", "釋出檔"],
        "tiles": [(n(F["n_corpus"]), "件美國核准專利"), ("2015–2025", "申請年"),
                  ("3", "個摘要層（每筆紀錄）"), ("5", "種文字表徵的 SPECTER2 向量"),
                  (n(F["n_topics"]), f"個主題，涵蓋 {n(F['n_topic_records'])} 筆"),
                  (n(F["n_lit"]), "筆屬於曾涉訟的專利家族")],
        "corpus_h": "語料建立",
        "corpus_p": (f"查詢共得到 {n(F['n_query'])} 個公開號。撞字篩選移除 {n(F['n_homonym'])} 筆 HPC 指其他詞的紀錄，"
                     "例如 hydroxypropyl cellulose（羥丙基纖維素）或 high-pressure compressor（高壓壓縮機）。完整性過濾"
                     f"保留標題、摘要、先前技術、詳細說明與請求項都有擷取到的 {n(F['n_complete'])} 筆候選紀錄。最後由收錄"
                     f"規則保留內文出現展開語 high performance comput* 的 {n(F['n_corpus'])} 筆。"),
        "corpus_note": f"{n(F['n_query'])} 是爬蟲輸出（第五步）的筆數，每個公開號一筆。",
        "removal_h": "各規則在哪些部移除紀錄",
        "removal_p": (f"兩條規則都保留 G 部（物理，含運算）與 H 部（電學）的大部分紀錄，其他部則大多被移除。完整性過濾移除 G 部"
                      f"候選的 {p1(F['comp_G'])}%、H 部的 {p1(F['comp_H'])}%，A 部則是 {p1(F['comp_A'])}%、C 部 "
                      f"{p1(F['comp_C'])}%。收錄規則接著移除 G 部的 {p1(F['excl_G'])}%、H 部的 {p1(F['excl_H'])}%，"
                      f"A 部則是 {p1(F['excl_A'])}%、C 部 {p1(F['excl_C'])}%，也就是 HPC 其他意思所屬的部。"),
        "missing_p": (f"候選紀錄被完整性過濾移除，主要是因為專利頁面沒有擷取到詳細說明（佔全部候選的 {p1(F['miss_dd'])}%）"
                      f"或先前技術（{p1(F['miss_bg'])}%）。"),
        "missing_cols": ["專利段落", "缺少的候選筆數", "佔全部候選（%）", "佔被移除者（%）"],
        "comp_h": "語料組成",
        "comp_p": (f"G 部與 H 部合計佔語料的 {p1(F['share_GH'])}%。{F['top1_name']} 的紀錄最多（{n(F['top1_n'])} 筆，"
                   f"{p1(F['top1_share'])}%），前十大權利人合計 {p1(F['top10_share'])}%。語料只收核准專利，所以 "
                   f"{F['trunc_years'][0]} 到 {F['trunc_years'][-1]} 年的件數有右截斷：這幾年的申請案很多還在審查中。"),
        "assignee_cols": ["名次", "目前權利人", "筆數", "佔比（%）"],
        "sum_h": "摘要層",
        "sum_p": ("每筆紀錄有三段分面（先前技術、發明內容、技術細節），由 Llama-3.1-8B-Instruct 撰寫；另有三個摘要，"
                  "由 Qwen2.5-14B-Instruct 以相同的最後一次呼叫撰寫。synthesized 先做一次推理，再由標題、摘要與分面寫成；"
                  "direct-brief 以單一提示讀各段落的短節錄；direct-full 以單一提示讀與分面相同的段落預算。"),
        "sum_metric_cols": ["摘要層", "平均字數", "壓縮比", "ROUGE-1 F1（自己）", "ROUGE-1 F1（錯位）", "ROUGE-L F1（自己）"],
        "example_h": "紀錄範例",
        "example_p": "隨機抽出（seed 42）的六筆釋出紀錄：G 部三筆、H 部兩筆、其他部一筆。",
        "example_labels": {"abstract": "摘要（專利原文）", "synthesized": "synthesized", "direct-brief": "direct-brief",
                           "direct-full": "direct-full", "open": "Google Patents"},
        "added_p": (f"摘要層帶有原文摘要沒有的內容。候選詞指出現在說明書、但不在標題與摘要中的長技術詞（12 個字母以上）；"
                    f"每筆紀錄平均有 {F['av_candidates']:.1f} 個候選詞。"),
        "added_cols": ["摘要層", "摘要含有的候選詞（每筆平均）", "至少含一個的紀錄（%）"],
        "ppl_p": (f"各摘要的 GPT-2 困惑度。平均值為 synthesized {p1(F['ppl_mean']['synthesized'])}、direct-brief "
                  f"{p1(F['ppl_mean']['direct-brief'])}、direct-full {p1(F['ppl_mean']['direct-full'])}；單筆最高為 "
                  f"{p1(F['ppl_max'])}。"),
        "topic_h": "主題",
        "topic_p": (f"SPECTER2 把每筆紀錄的標題與 synthesized 摘要編碼成向量，UMAP 降到三維，HDBSCAN 找出 "
                    f"{n(F['n_clusters'])} 個群，再合併成 {n(F['n_topics'])} 個主題，涵蓋 {n(F['n_topic_records'])} 筆"
                    f"（{p1(F['topic_share'])}%）；另有 {n(F['n_noise'])} 筆（{p1(F['noise_share'])}%）不屬於任何群。"
                    "圖中每一點是一筆紀錄。可以選一個主題標示出來，或依分類部上色；拖曳可以旋轉。"),
        "map_controls": {"mode": "上色方式", "topic": "標示一個主題", "section": "分類部", "pick": "主題"},
        "topic_cpc_p": (f"主題與紀錄的主要四碼分類明顯高於隨機地一致，但遠非完全一致：標準化互資訊 {p3(F['nmi'])}、"
                        f"調整後蘭德指數 {p3(F['ari'])}、同質性 {p3(F['homog'])}、完整性 {p3(F['compl'])}。主題是專利分類之外的"
                        "語意索引，不是用來取代分類。"),
        "topic_cols": ["主題", "筆數", "最大的分類", "最大的權利人"],
        "trend_p": "八個最大主題在各申請年所佔的比例（以這八個主題的紀錄為分母，%）。",
        "own_h": "權屬與訴訟",
        "own_p": (f"有 {n(F['n_changed'])} 筆（{p1(F['changed_share'])}%）專利頁面上的目前權利人與原始權利人不同。其中 "
                  f"{n(F['n_supported'])} 筆（{p1(F['supported_share'])}%）的最新讓與紀錄確實移轉了權利（讓與、更名或"
                  f"合併）；其餘的 {n(F['n_no_record'])} 筆沒有讓與紀錄，{n(F['n_security'])} 筆的最新紀錄是擔保權益或其解除，"
                  f"{n(F['n_corrective'])} 筆是更正，{n(F['n_other_type'])} 筆是其他類型。ownership_events.csv 為每一筆紀錄提供這些"
                  "欄位；其中 effective_assignee 在沒有移轉紀錄支持時保留原始權利人，依此計算有 "
                  f"{n(F['n_effective'])} 筆（{p1(F['effective_share'])}%）換了權利人。"),
        "conv_label": "權利人有變動的紀錄，其最新讓與紀錄的類型",
        "ba_p": "依 current_assignee 與依 effective_assignee 排序的前十大權利人：",
        "ba_cols": ["名次", "current_assignee", "筆數", "effective_assignee", "筆數"],
        "lit_p": (f"litigation_family.csv 列出 {n(F['n_lit'])} 筆：{n(F['n_seed'])} 件本身有地方法院訴訟、多方複審"
                  "（inter partes review）或再審查紀錄的專利，以及它們同一專利家族的其他成員。共 "
                  f"{n(F['n_fam'])} 個家族、{n(F['n_groups'])} 個案件群組，分布在 {n(F['n_lit_topics'])} 個主題。"),
        "lit_cols": ["專利", "案件群組", "本身有紀錄", "訴訟紀錄"],
        "lit_topic_cols": ["主題", "屬於涉訟家族的筆數", "主題總筆數"],
        "yes": "是", "no": "",
        "val_h": "技術驗證",
        "pair_p": (f"配對檢查：每個摘要分別與自己紀錄的摘要、以及錯開一筆的摘要計算 ROUGE。對自己的摘要 ROUGE-1 F1 為 "
                   f"{p3(F['rouge_own_min'])} 到 {p3(F['rouge_own_max'])}，對錯位摘要為 {p3(F['rouge_shift_min'])} 到 "
                   f"{p3(F['rouge_shift_max'])}。三層合計有 {n(F['n_flags'])} 個摘要（其中 synthesized "
                   f"{n(F['n_flags_syn'])} 個）對自己摘要的分數不高於錯位摘要，或低於 0.10，列在 "
                   "pairing_integrity_flags.csv。"),
        "cpc_p": (f"CPC 分類：以邏輯斯迴歸分類器讀取各種單位長度向量，預測紀錄的四碼 CPC 分類（至少 20 筆的 "
                  f"{n(F['cpc_labels'])} 個分類，多標籤），切分時同一專利家族不拆開（測試集 {n(F['cpc_test'])} 筆）。"
                  f"五種表徵的 micro-F1 都在 {p3(F['cpc_min'])} 到 {p3(F['cpc_max'])} 之間；synthesized 比原文摘要高 "
                  f"{p3(F['sa_est'])}（成對 bootstrap {n(F['n_boot'])} 次，95% 區間 {p3(F['sa_lo'])} 到 {p3(F['sa_hi'])}）。"),
        "app_p": (f"原始權利人：同一個分類器在至少 20 筆的 {n(F['app_classes'])} 個權利人中預測原始權利人，正確率在 "
                  f"{p3(F['app_min'])} 到 {p3(F['app_max'])} 之間，一律猜最大權利人則為 {p1(F['app_major'])}%。"
                  "這是用法示範，不是哪一種表徵比較好的證據。"),
        "intra_p": "各表徵在同一主題內紀錄的平均餘弦相似度：",
        "intra_cols": ["表徵", "主題內平均相似度"],
        "cost_h": "計算成本",
        "cost_p": (f"分面與三個摘要層的生成共花 {hours(F['gen_seconds'])} 小時"
                   + (f"；該次執行其餘有計時的步驟在同一硬體上共 {hours(F['other_seconds'])} 小時"
                      if F["other_seconds"] is not None else "")
                   + "。各步驟的硬體見下表（runtime_log_release.csv）。"),
        "cost_cols": ["生成步驟", "筆數", "小時", "每筆秒數", "硬體"],
        "files_h": "釋出檔",
        "files_p": "MANIFEST_release.csv 列出 release/ 的檔案與大小：",
        "files_cols": ["檔案", "列數", "欄數", "MB"],
        "sources": "資料檔：",
        "footer": ("HPC-PatentTrace v1.0。本頁由 scripts/build_results_page.py 從 pipeline 輸出的 release/ 產生；爬蟲與第六步的 notebook "
                   "都在這個 repository。"),
        "chart": {
            "funnel": ["查詢結果", "撞字篩選後", "五欄完整", "釋出語料"],
            "records": "筆數", "percent_removed": "該部被移除的比例（%）",
            "completeness": "完整性過濾", "inclusion": "收錄規則", "section": "分類部",
            "year": "申請年", "truncated": "右截斷（申請案仍在審查）",
            "ppl": "GPT-2 困惑度", "share_pct": "佔八個主題的比例（%）",
            "own": "自己的摘要", "shifted": "錯位摘要", "rouge1": "ROUGE-1 F1",
            "microf1": "micro-F1（CPC 分類）", "accuracy": "正確率（原始權利人）",
            "majority": "一律猜最大權利人", "difference": "micro-F1 差值",
            "noise": "不屬於任何群", "other_topics": "其他主題", "sec_G": "G（物理）",
            "sec_H": "H（電學）", "sec_other": "其他部", "topic": "主題",
            "records_of": "筆",
        },
    }


# ── HTML ────────────────────────────────────────────────────────────────────

def esc(s):
    return html.escape(str(s), quote=True)


def table(cols, rows, num_cols=()):
    head = "".join(f'<th class="{"num" if i in num_cols else ""}">{esc(c)}</th>' for i, c in enumerate(cols))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{esc(v)}</td>'
                                    for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def files_line(T, names):
    return f'<p class="files">{esc(T["sources"])} ' + ", ".join(f"<code>{esc(x)}</code>" for x in names) + "</p>"


def render(lang, D, F):
    T = texts(lang, F)
    ids = ["overview", "corpus", "composition", "summaries", "topics", "ownership", "validation", "cost", "files"]
    nav = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in zip(ids, T["nav"]))
    tiles = "".join(f'<div class="tile"><div class="tile-value">{esc(v)}</div><div class="tile-label">{esc(l)}</div></div>'
                    for v, l in T["tiles"])
    fname = FIELD_NAME[lang]
    missing = table(T["missing_cols"],
                    [(fname.get(m["field"], m["field"]), n(m["n"]), p1(m["share"]), p1(m["share_removed"]))
                     for m in D["missing"] if m["n"] > 0], num_cols=(1, 2, 3))
    assignees = table(T["assignee_cols"], [(a["rank"], a["name"], n(a["n"]), p1(a["share"])) for a in D["assignees"]],
                      num_cols=(0, 2, 3))
    sum_metrics = table(T["sum_metric_cols"],
                        [(l, p1(D["layers"][l]["words"]), f'{D["layers"][l]["compression"]:.4f}',
                          p3(D["layers"][l]["rouge1"]), p3(D["layers"][l]["rouge1_shift"]), p3(D["layers"][l]["rougeL"]))
                         for l in LAYERS], num_cols=(1, 2, 3, 4, 5))
    added = table(T["added_cols"], [(l, f'{D["added"][l]["mean"]:.2f}', p1(D["added"][l]["any"])) for l in LAYERS],
                  num_cols=(1, 2))
    topics = table(T["topic_cols"], [(t["label"], n(t["n"]), t["ipc"], t["company"]) for t in D["topics"]], num_cols=(1,))
    ba = table(T["ba_cols"], [(b["rank"], b["before"], n(b["n_before"]), b["after"], n(b["n_after"]))
                              for b in D["before_after"]], num_cols=(0, 2, 4))
    lit = table(T["lit_cols"], [(l["id"], l["group"], T["yes"] if l["seed"] else T["no"], l["text"])
                                for l in D["litigation"]])
    lit_topics = table(T["lit_topic_cols"], [(t["topic"], n(t["n"]), n(t["of"])) for t in D["litigation_topics"]],
                       num_cols=(1, 2))
    intra = table(T["intra_cols"], [(r, p3(D["intra"][r])) for r in REPS], num_cols=(1,))
    step = STEP_NAME[lang]
    cost = table(T["cost_cols"], [(step[c["step"]], n(c["n"]), hours(c["seconds"]), p1(c["seconds"] / c["n"]), c["hw"])
                                  for c in D["cost"]], num_cols=(1, 2, 3))
    files = table(T["files_cols"], [(f["file"], "–" if f["rows"] is None else n(f["rows"]),
                                     "–" if f["cols"] is None else n(f["cols"]), f'{f["mb"]:.2f}') for f in D["files"]],
                  num_cols=(1, 2, 3))
    page = {"lang": lang, "t": T["chart"], "map": T["map_controls"], "example": T["example_labels"],
            "conv": [CHANGE_NAME[lang][c["type"]] for c in D["conveyance"]], "conv_label": T["conv_label"]}
    other_href, other_label = T["switch"]
    return f"""<!doctype html>
<html lang="{T['html_lang']}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(T['title'])}</title>
<meta name="description" content="{esc(T['lead'])}">
<link rel="stylesheet" href="assets/results.css">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js" charset="utf-8"></script>
<script src="assets/data.js" charset="utf-8"></script>
<script>window.PAGE = {json.dumps(page, ensure_ascii=False)};</script>
<script src="assets/results.js" charset="utf-8" defer></script>
</head>
<body>
<header class="wrap">
  <div class="topline"><span class="kicker">{esc(T['kicker'])}</span><a class="switch" href="{other_href}">{esc(other_label)}</a></div>
  <h1>{esc(T['title'])}</h1>
  <p class="lead">{esc(T['lead'])}</p>
  <p class="source">{esc(T['source'])}</p>
</header>
<nav class="nav"><div class="wrap nav-inner">{nav}</div></nav>
<main class="wrap">
<section id="overview"><div class="tiles">{tiles}</div></section>

<section id="corpus">
  <h2>{esc(T['corpus_h'])}</h2>
  <p>{esc(T['corpus_p'])}</p>
  <div class="chart short" id="chart-funnel"></div>
  <p class="note">{esc(T['corpus_note'])}</p>
  <h3>{esc(T['removal_h'])}</h3>
  <p>{esc(T['removal_p'])}</p>
  <div class="chart" id="chart-removal"></div>
  <p>{esc(T['missing_p'])}</p>
  {missing}
  {files_line(T, ["tables/funnel_from_crawler_output.csv", "tables/completeness_bias_by_ipc_section.csv", "differential_exclusion.csv", "tables/completeness_missing_by_field.csv"])}
</section>

<section id="composition">
  <h2>{esc(T['comp_h'])}</h2>
  <p>{esc(T['comp_p'])}</p>
  <div class="two">
    <div class="chart" id="chart-sections"></div>
    <div class="chart" id="chart-years"></div>
  </div>
  {assignees}
  {files_line(T, ["tables/t5a_ipc_section.csv", "tables/t4_application_year.csv", "tables/t6_top20_assignee.csv"])}
</section>

<section id="summaries">
  <h2>{esc(T['sum_h'])}</h2>
  <p>{esc(T['sum_p'])}</p>
  {sum_metrics}
  <h3>{esc(T['example_h'])}</h3>
  <p>{esc(T['example_p'])}</p>
  <div class="viewer">
    <select id="example-select" aria-label="{esc(T['example_h'])}"></select>
    <div id="example-body"></div>
  </div>
  <p>{esc(T['added_p'])}</p>
  {added}
  <p>{esc(T['ppl_p'])}</p>
  <div class="chart" id="chart-ppl"></div>
  {files_line(T, ["summary_metrics.csv", "summaries_synthesized.csv", "summaries_direct_brief.csv", "summaries_direct_full.csv", "hpc_patenttrace_corpus.csv", "added_value_beyond_abstract.csv", "ppl_summaries.csv"])}
</section>

<section id="topics">
  <h2>{esc(T['topic_h'])}</h2>
  <p>{esc(T['topic_p'])}</p>
  <div class="controls">
    <label>{esc(T['map_controls']['mode'])}
      <select id="map-mode"><option value="topic">{esc(T['map_controls']['topic'])}</option><option value="section">{esc(T['map_controls']['section'])}</option></select>
    </label>
    <label id="map-topic-label">{esc(T['map_controls']['pick'])}
      <select id="map-topic"></select>
    </label>
  </div>
  <div class="chart tall" id="chart-map"></div>
  <p>{esc(T['topic_cpc_p'])}</p>
  {topics}
  <p>{esc(T['trend_p'])}</p>
  <div class="chart mid" id="chart-trends"></div>
  {files_line(T, ["cluster_results.csv", "topic_summary.csv", "tables/e6_topic_ipc_metrics.csv", "usage_topic_trends.csv"])}
</section>

<section id="ownership">
  <h2>{esc(T['own_h'])}</h2>
  <p>{esc(T['own_p'])}</p>
  <div class="chart short" id="chart-conveyance"></div>
  <p>{esc(T['ba_p'])}</p>
  {ba}
  <p>{esc(T['lit_p'])}</p>
  {lit_topics}
  {lit}
  {files_line(T, ["ownership_events.csv", "usage_assignee_before_after.csv", "litigation_family.csv", "usage_litigation_topics.csv"])}
</section>

<section id="validation">
  <h2>{esc(T['val_h'])}</h2>
  <p>{esc(T['pair_p'])}</p>
  <div class="chart short" id="chart-pairing"></div>
  <p>{esc(T['cpc_p'])}</p>
  <div class="two">
    <div class="chart" id="chart-cpc"></div>
    <div class="chart" id="chart-boot"></div>
  </div>
  <p>{esc(T['app_p'])}</p>
  <div class="chart short" id="chart-applicant"></div>
  <p>{esc(T['intra_p'])}</p>
  {intra}
  {files_line(T, ["summary_metrics.csv", "pairing_integrity_flags.csv", "ipc_downstream_comparison.csv", "ipc_downstream_bootstrap.csv", "developer_demo_comparison.csv", "intra_topic_similarity.csv"])}
</section>

<section id="cost">
  <h2>{esc(T['cost_h'])}</h2>
  <p>{esc(T['cost_p'])}</p>
  {cost}
  {files_line(T, ["runtime_log_release.csv"])}
</section>

<section id="files">
  <h2>{esc(T['files_h'])}</h2>
  <p>{esc(T['files_p'])}</p>
  {files}
</section>
</main>
<footer class="wrap"><p>{esc(T['footer'])}</p></footer>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="output folder of a full run of the pipeline notebook (holds release/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"))
    a = ap.parse_args()
    D, F = collect(Release(a.run))
    os.makedirs(os.path.join(a.out, "assets"), exist_ok=True)
    with open(os.path.join(a.out, "assets", "data.js"), "w", encoding="utf-8") as f:
        f.write("window.RESULTS = " + json.dumps(D, ensure_ascii=False, separators=(",", ":")) + ";\n")
    for lang, name in (("en", "index.html"), ("tw", "index_tw.html")):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(render(lang, D, F))
    print(f"read {a.run}: {F['n_corpus']:,} records, {F['n_topics']} topics -> {a.out}")


if __name__ == "__main__":
    main()
