"""Tables of the data descriptor, built from the output folder of the pipeline notebook (opened with open_run).

Table 1 (settings of the summary layers) comes from release/release_assets/config/generation.json, Table 2
(last assignment record) from release/ownership_events.csv, Table 3 (files) from release/MANIFEST_release.csv,
and Table 5 (classification probe) is computed again from the embeddings in release/embeddings/. Table 4
describes the validation design in words and has no data.
"""
import ast
import os
import warnings

import numpy as np
import pandas as pd

LAYERS = ["direct-brief", "direct-full", "synthesized"]
REPS = ["abstract", "fulltext-truncated", "direct-brief", "direct-full", "synthesized"]
CHANGE_CATEGORIES = {   # conveyance_type -> (label, transfers title)
    "true_assignment": ("Assignment (title transfer)", True), "no_assignment_record": ("No assignment record", False),
    "release_of_lien": ("Release of security interest", False),
    "security_interest": ("Security interest (not released)", False), "change_of_name": ("Change of name", True),
    "corrective": ("Corrective record", False), "merger": ("Merger", True), "partial_transfer": ("Partial transfer", False),
    "other": ("Other", False)}


def _save(table, out_dir, name):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        table.to_csv(os.path.join(out_dir, name), index=False)
    return table


def table1_layer_settings(run, out_dir=None):
    """Settings of the three summary layers (release/release_assets/config/generation.json)."""
    g = run.json("release/release_assets/config/generation.json")
    m, dec, bud = g["models"], g["decoding"], g["input_budgets_tokens"]
    final = f"{m['single_pass']['id'].split('/')[-1]}, {m['single_pass']['precision']}"
    budget = lambda b: (f"background {b['background']:,}, summary input {b['summary']:,}, "
                        f"detailed description {b['detailed_description']:,}")
    rows = [
        ["direct-brief", f"one call ({final})", "title, abstract; " + budget(bud["direct_brief"]),
         f"≤{dec['max_new_tokens']['single_pass']} new tokens, 5–7 sentences"],
        ["direct-full", f"one call ({final})", "title, abstract; " + budget(bud["direct_full"]),
         f"≤{dec['max_new_tokens']['single_pass']} new tokens, 5–7 sentences"],
        ["synthesized",
         f"facet step ({m['stage1_facets']['id'].split('/')[-1]}, {m['stage1_facets']['precision']}, three calls) "
         f"and two calls ({m['stage2_condense']['id'].split('/')[-1]}, {m['stage2_condense']['precision']})",
         "facets from " + budget(bud["synthesized_facets"]) + " (facet model tokens); then title, abstract and the facets",
         f"≤{dec['max_new_tokens']['stage2']} new tokens, 5–7 sentences"],
    ]
    out = pd.DataFrame(rows, columns=["Layer", "Model calls", "Input sections and token budgets", "Output"])
    out.attrs["note"] = (f"Greedy decoding (do_sample={dec['do_sample']}), repetition penalty "
                         f"{dec['repetition_penalty']}; token budgets in tokens of the model that reads the section.")
    return _save(out, out_dir, "Table1_summary_layer_settings.csv")


def table2_assignment(run, out_dir=None):
    """Last assignment record of the records whose current assignee differs from the original assignee."""
    own = run.csv("release/ownership_events.csv", dtype={"patent_id": str})
    ch = own[own["assignee_changed"].astype(str).str.lower() == "true"]
    vc = ch["conveyance_type"].value_counts()
    rows = [[CHANGE_CATEGORIES[k][0], int(v), f"{v / len(ch) * 100:.1f}%", "yes" if CHANGE_CATEGORIES[k][1] else "no"]
            for k, v in vc.items()]
    out = pd.DataFrame(rows, columns=["Category", "Records", "Share of changes", "Transfers title"])
    out.attrs["n_changes"] = len(ch)
    return _save(out, out_dir, "Table2_last_assignment_record.csv")


def table3_files(run, out_dir=None):
    """Files of release/ with their size (release/MANIFEST_release.csv)."""
    man = run.csv("release/MANIFEST_release.csv", dtype=str, keep_default_na=False)
    out = man.rename(columns={"file": "File", "rows": "Rows", "columns": "Columns"})[["File", "Rows", "Columns", "MB"]]
    return _save(out, out_dir, "Table3_files_of_the_release.csv")


def _cpc4(s):
    try:
        lst = ast.literal_eval(str(s)) if str(s).strip().startswith("[") else []
    except Exception:
        lst = []
    return {d["cpc"][:4] for d in lst if isinstance(d, dict) and d.get("cpc")}


def table5_probe(run, out_dir=None, min_label_n=20, random_state=42, test_size=0.2):
    """Classification probe on the five embedding sets, computed again from release/.

    Same procedure as section 8.9 of the pipeline: logistic regression on the unit-length embeddings, a
    family-grouped train/test split, CPC codes with at least min_label_n records.
    Returns (table, comparison with release/cpc_downstream_comparison.csv written by the pipeline).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

    ids = run.csv("release/released_corpus_ids.csv", dtype=str)["patent_id"]
    corpus = run.csv("release/hpc_patenttrace_corpus.csv", dtype=str, keep_default_na=False,
                     usecols=["patent_id", "main_cpc_4", "cpc_description", "family"])
    lab = ids.to_frame().merge(corpus, on="patent_id", how="left")
    lab["all4"] = [s | ({m} if m else set()) for s, m in zip(lab["cpc_description"].map(_cpc4), lab["main_cpc_4"])]
    cnt = pd.Series([c for s in lab["all4"] for c in s]).value_counts()
    keep = set(cnt[cnt >= min_label_n].index)
    lab["kept"] = [s & keep for s in lab["all4"]]
    mask = np.array([len(s) > 0 for s in lab["kept"]])
    sub = lab[mask].reset_index(drop=True)
    Y = MultiLabelBinarizer(classes=sorted(keep)).fit_transform(sub["kept"])
    y4 = LabelEncoder().fit_transform(sub["main_cpc_4"].values)
    y3 = LabelEncoder().fit_transform(sub["main_cpc_4"].str.strip().str.upper().str[:3].values)
    root = sub["family"].str.extract(r"^\s*(\S+)", expand=False).fillna("")
    groups = np.where(root != "", root, "self:" + sub["patent_id"])
    ix = np.arange(len(sub))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(ix, y4, groups=groups))

    rows = []
    for rep in REPS:
        emb = run.parquet(f"release/embeddings/embeddings_{rep}.parquet")
        X = ids.to_frame().merge(emb, on="patent_id", how="left").drop(columns=["patent_id"]).values.astype(np.float32)[mask]
        # one process (n_jobs only parallelises the 63 independent fits); some BLAS builds raise spurious
        # floating-point warnings in matmul, which do not change the fitted models
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            multi = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], Y[tr])
            Yp = multi.predict(X[te])
            yp4 = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y4[tr]).predict(X[te])
            yp3 = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y3[tr]).predict(X[te])
        rows.append({"Representation": rep, "cpc3_acc": accuracy_score(y3[te], yp3), "single_acc": accuracy_score(y4[te], yp4),
                     "any_label_hit": float((np.logical_and(Y[te], Yp).sum(1) > 0).mean()),
                     "ml_f1_micro": f1_score(Y[te], Yp, average="micro", zero_division=0)})
    res = pd.DataFrame(rows)
    floor3 = float(pd.Series(y3[tr]).value_counts(normalize=True).iloc[0])
    floor4 = float(pd.Series(y4[tr]).value_counts(normalize=True).iloc[0])

    pct = lambda v: f"{v * 100:.1f}%"
    table = pd.DataFrame({"Representation": res["Representation"], "Three-character accuracy": res["cpc3_acc"].map(pct),
                          "Four-character accuracy": res["single_acc"].map(pct),
                          "At least one label correct": res["any_label_hit"].map(pct),
                          "Multi-label micro-F1": res["ml_f1_micro"].map(lambda v: f"{v:.4f}")})
    table.loc[len(table)] = ["Share of the largest class in the training set (no vectors)", pct(floor3), pct(floor4), "—", "—"]
    table.attrs.update(n_test=len(te), n_records=len(sub), n_labels=Y.shape[1])

    written = run.csv("release/cpc_downstream_comparison.csv").set_index("representation")
    cols = ["cpc3_acc", "single_acc", "any_label_hit", "ml_f1_micro"]
    check = res.set_index("Representation")[cols].round(4).join(written[cols], rsuffix=" (pipeline)")
    check["same"] = ["yes" if all(abs(check.loc[r, c] - check.loc[r, f"{c} (pipeline)"]) < 5e-5 for c in cols) else "NO"
                     for r in check.index]
    return _save(table, out_dir, "Table5_classification_probe.csv"), check.reset_index()


def summary_numbers(run, out_dir=None):
    """Counts quoted in the text of the data descriptor, read from the output folder of the pipeline."""
    from .figures import construction_counts
    c = construction_counts(run)
    own = run.csv("release/ownership_events.csv", dtype=str, keep_default_na=False)
    changed = own["assignee_changed"].str.lower() == "true"
    supported = own["change_supported"].str.lower() == "true"
    topics = run.csv("release/topic_summary.csv")
    clu = run.csv("release/cluster_results.csv", usecols=["cluster_id"])
    rows = [("query result (publication numbers)", c["query"]), ("removed by the homonym screening", c["homonym_removed"]),
            ("candidates", c["candidates"]), ("removed by the completeness filter", c["incomplete_removed"]),
            ("five complete fields", c["complete"]), ("excluded by the inclusion rule", c["excluded"]),
            ("released records", c["released"]),
            ("clusters (HDBSCAN)", int(clu.loc[clu["cluster_id"] != -1, "cluster_id"].nunique())),
            ("merged topics", int((topics["topic_id"] != -1).sum())),
            ("records without a topic", int(topics.loc[topics["topic_id"] == -1, "patent_count"].sum())),
            ("records whose current assignee changed", int(changed.sum())),
            ("changes supported by a title transfer", int((changed & supported).sum())),
            ("records whose effective assignee differs", int((own["effective_assignee"] != own["current_assignee"]).sum()))]
    return _save(pd.DataFrame(rows, columns=["Quantity", "Value"]), out_dir, "numbers_in_the_text.csv")
