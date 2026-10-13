"""Section 7: ownership evidence and usage examples.

Run from the pipeline notebook with run_step("s7_ownership_usage"); the parts below are executed in the namespace of the notebook.
"""

# ── 7.1  Evidence grade of each change of ownership (read-only) ──────────────
# current_assignee is the first current assignee listed on the Google Patents page (corpus/json_to_csv.py) and does not necessarily come from the last assignment record; that record can
#   (a) be no transfer of ownership at all (security interest, release of a security interest, corrective record),
#   (b) name more than one party (up to 16 in this corpus); then the record cannot show which one the page lists,
#   (c) be absent (current_assignee differs from assignee, but there is no assignment record).
# This section parses all assignment segments with code="AS" in legal_events, takes the record with the latest "event date + effective date",
# decides its assignment type, records the number of parties that the record names and writes the release file
# ownership_events.csv, so that users can select their own entity resolution rule.
import os, re, ast
import pandas as pd

# The description of one AS event joins several assignment records, always in this format:
#   Owner name:<assignee>Free format text:<assignment type>;ASSIGNOR:<assignor>
#   REEL/FRAME:<USPTO assignment record number>Effective date:<YYYYMMDD>
SEGMENT_RE = re.compile(
    r"Owner name:(?P<owner>.*?)"
    r"Free format text:(?P<conveyance>.*?)"
    r"(?:REEL/FRAME:(?P<reel>[\d/]+))?"
    r"(?:Effective date:(?P<effective>\d{8}))?"
    r"(?=Owner name:|$)", re.S)

# Decision of the assignment type: compare in order; the first hit decides. RELEASE must come before SECURITY:
# "RELEASE OF SECURITY INTEREST" contains both keywords, and its meaning is a release, not a grant of security.
CONVEYANCE_RULES = [
    ("release_of_lien",   ("RELEASE",)),
    ("security_interest", ("SECURITY",)),
    ("change_of_name",    ("CHANGE OF NAME", "CERTIFICATE OF AMENDMENT")),
    ("merger",            ("MERGER",)),
    ("partial_transfer",  ("PARTIAL",)),
    ("corrective",        ("CORRECTIVE",)),
    ("true_assignment",   ("ASSIGNMENT OF ASSIGNORS INTEREST", "NUNC PRO TUNC",
                           "CONFIRMATORY", "PATENT ASSIGNMENT")),
]

# Only these three assignment types actually transfer ownership; the others cannot support a change of assignee.
TITLE_TRANSFERRING = {"true_assignment", "merger", "change_of_name"}

# Verified data-source error: this name never occurs in any legal event, and the original assignee is the same in all records.
ASSIGNEE_OVERRIDES = {"Everpure Inc": "Pure Storage Inc"}


def _latest_record(legal_events):
    """Take the assignment record with the latest event date + effective date.

    Return (assignment type text, number of parties that the record names); return None when there is no assignment record.
    """
    try:
        events = (ast.literal_eval(legal_events)
                  if str(legal_events).strip().startswith("[") else [])
    except (ValueError, SyntaxError):
        return None
    segments = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("code") != "AS":
            continue
        for m in SEGMENT_RE.finditer(ev.get("description", "")):
            segments.append((ev.get("date", ""), m.group("effective") or "",
                             (m.group("conveyance") or "").strip()))
    if not segments:
        return None
    newest = max((s[0], s[1]) for s in segments)
    same = [s for s in segments if (s[0], s[1]) == newest]
    return same[0][2], len(same)


def _conveyance(text: str) -> str:
    upper = text.upper()
    for label, keys in CONVEYANCE_RULES:
        if any(k in upper for k in keys):
            return label
    return "other"


def _norm(s: pd.Series) -> pd.Series:
    """Only to decide whether two names are the same: removes case and non-letter characters; no fuzzy match."""
    return s.astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)


# ── Read the released corpus ─────────────────────────────────────────────────
_COLS = ["patent_id", "assignee", "current_assignee",
         "reassignment_assignee", "legal_events"]
_src = pd.read_csv(CLEAN_PATH, usecols=_COLS, low_memory=False).fillna("")

_ids_fp = os.path.join(SAVE_DIR, "released_corpus_ids.csv")
if "result_df" in globals():
    _keep = set(result_df["patent_id"])
elif os.path.exists(_ids_fp):
    _keep = set(pd.read_csv(_ids_fp, dtype=str)["patent_id"])
else:
    raise RuntimeError("Released corpus identifiers not found: run 3.5 first.")

own = _src[_src["patent_id"].isin(_keep)].reset_index(drop=True)
if len(own) != len(_keep):
    raise RuntimeError(f"Released corpus mismatch: source hits {len(own):,}, fingerprint {len(_keep):,}")

with timed("7.1  Grading of the assignment evidence", n_items=len(own)):
    _rec = own["legal_events"].map(_latest_record)
    own["conveyance_type"] = _rec.map(
        lambda r: "no_assignment_record" if r is None else _conveyance(r[0]))
    own["n_parties_in_record"] = _rec.map(lambda r: 0 if r is None else r[1])
    own["assignee_changed"] = (
        (_norm(own["current_assignee"]) != _norm(own["assignee"])) &
        (own["current_assignee"].str.strip() != "") &
        (own["assignee"].str.strip() != ""))
    # Whether a record that "transfers ownership" supports the change of assignee
    own["change_supported"] = (own["assignee_changed"] &
                               own["conveyance_type"].isin(TITLE_TRANSFERRING))
    # When the record names more than one party, the record cannot show which party current_assignee is
    own["party_ambiguous"] = own["assignee_changed"] & (own["n_parties_in_record"] > 1)

    # effective_assignee: falls back to the original assignee when no transfer record supports the change; a verified error is renamed directly.
    own["effective_assignee"] = own["current_assignee"]
    _revert = own["assignee_changed"] & ~own["change_supported"]
    own.loc[_revert, "effective_assignee"] = own.loc[_revert, "assignee"]
    own["effective_assignee"] = own["effective_assignee"].replace(ASSIGNEE_OVERRIDES)

# ── Report ───────────────────────────────────────────────────────────────────
_mv = own[own["assignee_changed"]]
print(f"released corpus              : {len(own):>6,}")
print(f"current_assignee != assignee : {len(_mv):>6,}  ({len(_mv)/len(own)*100:.1f}%)\n")

_TYPE_LABEL = {"true_assignment": "true transfer", "no_assignment_record": "no assignment record",
       "release_of_lien": "release of lien", "security_interest": "unreleased security",
       "corrective": "corrective record", "change_of_name": "change of name", "merger": "merger",
       "partial_transfer": "partial transfer", "other": "other"}
print(f"{'last record type':<22}{'records':>6}{'share':>9}{'multiple':>10}{'transfers':>11}")
print("-" * 60)
for k, v in _mv["conveyance_type"].value_counts().items():
    multi = int((_mv[_mv["conveyance_type"] == k]["n_parties_in_record"] > 1).sum())
    mark = "yes" if k in TITLE_TRANSFERRING else "no"
    print(f"{_TYPE_LABEL.get(k, k):<20}{v:>7,}{v/len(_mv)*100:>8.1f}%{multi:>10}{mark:>11}")

print(f"\nassignee changes supported by a transfer : {int(own['change_supported'].sum()):>5,}"
      f"  ({own['change_supported'].sum()/max(len(_mv), 1)*100:.1f}% of the changes)")
print(f"last record names more than one party    : {int(own['party_ambiguous'].sum()):>5,}"
      f"  ({own['party_ambiguous'].sum()/max(len(_mv), 1)*100:.1f}% of the changes; the record does not show which party is listed)")
print(f"assignee changed after the correction    : "
      f"{int((own['effective_assignee'] != own['current_assignee']).sum()):>5,}"
      f"  ({(own['effective_assignee'] != own['current_assignee']).mean()*100:.1f}% of the whole corpus)")

print("\nTop ten assignees: before vs after the correction")
_a = own["current_assignee"].value_counts().head(10)
_b = own["effective_assignee"].value_counts().head(10)
print(f"  {'current_assignee':<40}{'n':>5}   {'effective_assignee':<40}{'n':>5}")
for _i in range(min(10, len(_a), len(_b))):
    print(f"  {_a.index[_i][:38]:<40}{_a.iloc[_i]:>5}   {_b.index[_i][:38]:<40}{_b.iloc[_i]:>5}")

print("\nCheck of data-source errors:")
for _bad, _real in ASSIGNEE_OVERRIDES.items():
    _r = own[own["current_assignee"] == _bad]
    if not len(_r):
        continue
    print(f"  {_bad} ({len(_r)} records)")
    print(f"    original assignee is always {_real}  : {(_r['assignee'] == _real).all()}")
    print(f"    legal events that mention '{_bad.split()[0]}' : "
          f"{int(_r['legal_events'].str.contains(_bad.split()[0], case=False).sum())}")
    print(f"    times '{_real}' occurs as current_assignee : "
          f"{int((own['current_assignee'] == _real).sum())}")

_out = own[["patent_id", "assignee", "current_assignee", "reassignment_assignee",
            "assignee_changed", "conveyance_type", "n_parties_in_record",
            "change_supported", "party_ambiguous", "effective_assignee"]]
_out.to_csv(os.path.join(SAVE_DIR, "ownership_events.csv"), index=False)
print(f"\nsaved -> ownership_events.csv ({len(_out):,} records)")


# ── 7.2  Spelling variants of the assignee names (read-only) ────────────────
import os, re
import pandas as pd

_rd = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
print(f"distinct current_assignee in the original: {_rd['current_assignee'].nunique():,}")
print(f"empty values: {(_rd['current_assignee'].str.strip() == '').sum():,}\n")
print("The 30 most frequent names (scan by eye for different spellings of the same company):")
for name, cnt in _rd["current_assignee"].value_counts().head(30).items():
    print(f"  {str(name)[:55]:57} {cnt:>5}")


# ── 7.3  Current versus original assignee among the candidates (read-only) ──
import pandas as pd, warnings
warnings.filterwarnings("ignore")

_cols = ["patent_id", "current_assignee", "assignee", "reassignment_assignee"]
_src  = pd.read_csv(CLEAN_PATH, usecols=_cols, low_memory=False).fillna("")

# A current assignee that differs from the original assignee means that a transfer happened
_moved = _src[(_src["current_assignee"].str.strip() != "") &
              (_src["assignee"].str.strip() != "") &
              (_src["current_assignee"] != _src["assignee"])]

print(f"patents with a transfer of rights: {len(_moved):,} / {len(_src):,}\n")
print(f"{'current_assignee':42} {'original assignee':38} {'records':>5}")
print("-" * 92)
_pairs = (_moved.groupby(["current_assignee", "assignee"]).size()
          .reset_index(name="n").sort_values("n", ascending=False))
for _, r in _pairs.head(30).iterrows():
    print(f"{str(r['current_assignee'])[:40]:42} {str(r['assignee'])[:36]:38} {r['n']:>5}")


# ── 7.4  Normalised assignee names (adds assignee_canon and assignee_display) ──
import os, re
import pandas as pd

LEGAL_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "llp", "lp", "plc", "gmbh", "ag", "sa", "nv",
    "bv", "kk", "kabushiki kaisha", "oy", "ab", "as", "spa", "srl",
    "pte", "pty", "holdings", "holding",
]

def canon_assignee(name: str) -> str:
    """Normalize to the standard form for comparison: lower case, no punctuation, no legal-entity suffix."""
    s = str(name).strip().lower()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"[.,;:()\"']", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed:                      # repeated suffixes such as "co ltd" must be removed again and again
        changed = False
        for suf in LEGAL_SUFFIXES:
            if s.endswith(" " + suf):
                s = s[: -(len(suf) + 1)].strip()
                changed = True
    return re.sub(r"\s+", " ", s).strip()

# Manual alias table: write here the names that the rules cannot merge, or the names that need a formal display name
ALIAS = {
    "taiwan semiconductor manufacturing tsmc": "Taiwan Semiconductor Manufacturing (TSMC)",
    "taiwan semiconductor manufacturing":      "Taiwan Semiconductor Manufacturing (TSMC)",
    "international business machines":         "IBM",
    "advanced micro devices":                  "AMD",
    "hewlett packard enterprise development":  "Hewlett Packard Enterprise",
    "sk hynix":                                "SK Hynix",
    "nvidia":                                  "NVIDIA",
    "emc":                                     "EMC",
}

def display_assignee(name: str) -> str:
    c = canon_assignee(name)
    if c in ALIAS:
        return ALIAS[c]
    return " ".join(w.capitalize() for w in c.split()) if c else "(unknown)"

result_path = os.path.join(SAVE_DIR, "cluster_results.csv")
result_df   = pd.read_csv(result_path).fillna("")

result_df["assignee_canon"]   = result_df["current_assignee"].map(canon_assignee)
result_df["assignee_display"] = result_df["current_assignee"].map(display_assignee)

print(f"distinct original names: {result_df['current_assignee'].nunique():,}")
print(f"after normalization    : {result_df['assignee_display'].nunique():,}")
print(f"reduction              : {result_df['current_assignee'].nunique() - result_df['assignee_display'].nunique():,}\n")

print("Merged variants (only the first 25 groups):")
_g = result_df.groupby("assignee_display")["current_assignee"].unique()
_shown = 0
for disp, variants in _g.items():
    if len(variants) > 1:
        print(f"  {disp}")
        for v in variants:
            print(f"      <- {v!r}")
        _shown += 1
        if _shown >= 25:
            break

result_df.to_csv(result_path, index=False)
print("\nsaved -> cluster_results.csv (two columns added; the original columns are not changed)")


# ── 7.5  Portfolio of the largest assignees: class codes and clusters ───────
result_df  = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
cluster_df = result_df[result_df["cluster_id"] != -1]

TOP_N_COMPANIES = 15
top_companies = cluster_df["current_assignee"].value_counts().head(TOP_N_COMPANIES).index.tolist()

for co in top_companies:
    co_rows  = cluster_df[cluster_df["current_assignee"] == co]
    co_total = len(co_rows)
    print(f"\n{co[:60]}  (n={co_total})")
    print(f"  {'IPC':8} {'n':>5} {'share':>7}  clusters")
    for ipc, cnt in co_rows["main_ipc_4"].value_counts().head(5).items():
        cdist = co_rows[co_rows["main_ipc_4"] == ipc]["cluster_id"].value_counts().head(4)
        cstr  = " ".join(f"C{c}({n})" for c, n in cdist.items())
        print(f"  {ipc:8} {cnt:>5} {cnt/co_total*100:>6.1f}%  {cstr}")
    print(f"  spans {co_rows['cluster_id'].nunique()} clusters")


# ── 7.6  Interactive three-dimensional maps (plotly; also saved as HTML) ──────
import plotly.express as px

_TOPIC_EXCLUDE = ["__noise__", "Off-topic / Non-HPC", "Unclassified"]
top4 = result_df["current_assignee"].value_counts().head(4).index.tolist()
plot_df = result_df[result_df["current_assignee"].isin(top4)]

fig1 = px.scatter_3d(
    plot_df, x="umap_3d_x", y="umap_3d_y", z="umap_3d_z",
    color="current_assignee", color_discrete_sequence=PALETTE,
    hover_data=["patent_id", "title", "main_ipc_4", "merged_topic"],
    opacity=0.85, title="Figure 7.1  Records of the four largest assignees on the three-dimensional UMAP map")
_ = fig1.update_traces(marker=dict(size=2))
_ = fig1.update_layout(height=650, legend_title="assignee", template="plotly_white")
fig1.show()

plot_all = result_df[~result_df["merged_topic"].isin(_TOPIC_EXCLUDE)]
fig2 = px.scatter_3d(
    plot_all, x="umap_3d_x", y="umap_3d_y", z="umap_3d_z",
    color="merged_topic",
    hover_data=["patent_id", "title", "current_assignee", "main_ipc_4"],
    opacity=0.75, title="Figure 7.2  Merged topics on the three-dimensional UMAP map")
_ = fig2.update_traces(marker=dict(size=2))
_ = fig2.update_layout(height=650, legend_title="topic", template="plotly_white")
fig2.show()

fig1.write_html(os.path.join(FIG_DIR, "fig_7_1_umap3d_top4_assignees.html"))
fig2.write_html(os.path.join(FIG_DIR, "fig_7_2_umap3d_topics.html"))
print("saved -> figures/fig_7_1_umap3d_top4_assignees.html, figures/fig_7_2_umap3d_topics.html")


# ── 7.7  Figure 7.3: the same two-dimensional map coloured by class and by topic (CPU only) ──
# One projection, drawn twice: panel a coloured by the three-character class (first three characters of the
# first CPC path, column main_ipc_3), panel b by the merged topic. The coordinates are identical, so the panels
# differ only in the colouring. The UMAP settings are those of 6.4 (n_neighbors=15, min_dist=0.0, cosine, the
# same seed) with two output dimensions. Files: umap2d_coords.csv (cached coordinates; delete it after new
# embeddings), figures/fig_7_3_umap2d_class_vs_topic.png and an interactive three-dimensional map by class.
# Colours: tab20 for the ten classes and twelve topics of the figure in the manuscript; every other record is grey.
from matplotlib.lines import Line2D

TOP_N_IPC3   = 10          # panel a: the ten largest three-character classes; the others are grey
TOP_N_TOPIC  = 12          # panel b: the twelve largest topics; the other topics and the noise are grey
NOISE_LABELS = ["__noise__", "Off-topic / Non-HPC", "Unclassified", ""]
_CMAP = [plt.get_cmap("tab20")(i) for i in range(20)]

_clu = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_clu["patent_id"] = _clu["patent_id"].astype(str)
if "main_ipc_3" not in _clu.columns:
    _clu = add_ipc3(_clu)

_coord_path = os.path.join(SAVE_DIR, "umap2d_coords.csv")

if os.path.exists(_coord_path):
    _xy = pd.read_csv(_coord_path)
    _xy["patent_id"] = _xy["patent_id"].astype(str)
    print(f"read the existing coordinates -> {_coord_path} ({len(_xy):,} records)")
else:
    # Embedding source: specter2_embeddings.csv of 6.2, otherwise the parquet of 6.13 (both are the synthesized representation)
    _emb, _src = None, None
    for _cand in (os.path.join(SAVE_DIR, "specter2_embeddings.csv"),
                  os.path.join(SAVE_DIR, "embeddings", "embeddings_synthesized.parquet")):
        if os.path.exists(_cand):
            _emb = pd.read_parquet(_cand) if _cand.endswith(".parquet") else pd.read_csv(_cand)
            _src = _cand
            break
    if _emb is None:
        raise RuntimeError("Embedding file not found (specter2_embeddings.csv or "
                           "embeddings/embeddings_synthesized.parquet). Run 6.2 or 6.13 first.")
    _emb["patent_id"] = _emb["patent_id"].astype(str)
    _m = _clu[["patent_id"]].merge(_emb, on="patent_id", how="left")
    _miss = _m.drop(columns=["patent_id"]).isna().all(axis=1)
    if _miss.any():
        raise RuntimeError(f"{int(_miss.sum()):,} records have no embedding; the projection is not possible")
    _X = _m.drop(columns=["patent_id"]).values.astype(np.float32)
    print(f"embedding source -> {_src} | {_X.shape}")

    import umap
    with timed("7.7  UMAP 768d -> 2d (for the figure)", n_items=len(_X)):
        _E2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0,
                        metric="cosine", random_state=RANDOM_STATE,
                        low_memory=False).fit_transform(_X)
    _xy = pd.DataFrame({"patent_id": _m["patent_id"].values,
                        "umap_2d_x": _E2[:, 0], "umap_2d_y": _E2[:, 1]})
    _xy.to_csv(_coord_path, index=False)
    print(f"saved -> {_coord_path}")

_plot = (_clu.merge(_xy, on="patent_id", how="left")
             .dropna(subset=["umap_2d_x", "umap_2d_y"])
             .reset_index(drop=True))
print(f"records in the figure: {len(_plot):,} / {len(_clu):,}")


_ipc3   = _plot["main_ipc_3"].astype(str)
_topic  = _plot["merged_topic"].astype(str)
_n_ipc3 = _ipc3.nunique()
_n_top  = _topic[~_topic.isin(NOISE_LABELS)].nunique()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, lab, top_n, title, other in (
        (axes[0], _ipc3,  TOP_N_IPC3,  f"a  Three-character CPC class (largest {TOP_N_IPC3} of {_n_ipc3})", "other classes"),
        (axes[1], _topic, TOP_N_TOPIC, f"b  Merged topic (largest {TOP_N_TOPIC} of {_n_top})",            "other or no topic")):
    lab  = lab.where(~lab.isin(NOISE_LABELS), "")
    top  = lab[lab != ""].value_counts().head(top_n).index.tolist()
    grey = ~lab.isin(top)
    _ = ax.scatter(_plot.loc[grey, "umap_2d_x"], _plot.loc[grey, "umap_2d_y"], s=0.6,
                   c=[OTHER_GREY], linewidths=0, rasterized=True)
    handles = []
    for k, t in enumerate(top):
        sel = (lab == t).values
        _ = ax.scatter(_plot.loc[sel, "umap_2d_x"], _plot.loc[sel, "umap_2d_y"], s=0.8,
                       c=[_CMAP[k % 20]], linewidths=0, rasterized=True)
        handles.append(Line2D([0], [0], marker="o", ls="", ms=3, color=_CMAP[k % 20],
                              label=(t if len(t) <= 44 else t[:43] + "…")))
    handles.append(Line2D([0], [0], marker="o", ls="", ms=3, color=OTHER_GREY, label=other))
    _ = ax.set_title(title, fontsize=8)
    _ = ax.set_xticks([]); _ = ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4)
    _ = ax.legend(handles=handles, fontsize=4.6, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)
_ = fig.suptitle(f"Figure 7.3  UMAP of the synthesized embeddings ({len(_plot):,} records)", fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
_ = save_fig(fig, "fig_7_3_umap2d_class_vs_topic")
plt.show()

# ── Record counts of the three-character classes (for the figure caption) ───
_tab3 = (_ipc3.value_counts().rename_axis("main_ipc_3").reset_index(name="n"))
_tab3["share_%"] = (_tab3["n"] / len(_plot) * 100).round(1)
os.makedirs(os.path.join(SAVE_DIR, "tables"), exist_ok=True)
_tab3.to_csv(os.path.join(SAVE_DIR, "tables", "ipc3_counts.csv"), index=False)
print(f"\n{_n_ipc3} three-character classes in total; the first {TOP_N_IPC3} are listed below:")
print(_tab3.head(TOP_N_IPC3).to_string(index=False))

# ── Interactive three-dimensional map by class (coordinates of 6.4, no new projection) ───
import plotly.express as px
_top3 = _tab3.head(TOP_N_IPC3)["main_ipc_3"].tolist()
_p3 = _plot.copy()
_p3["ipc3_group"] = np.where(_p3["main_ipc_3"].isin(_top3), _p3["main_ipc_3"], "Other")
_fig3 = px.scatter_3d(_p3, x="umap_3d_x", y="umap_3d_y", z="umap_3d_z",
                      color="ipc3_group",
                      hover_data=["patent_id", "title", "main_ipc_4", "merged_topic"],
                      opacity=0.75, title="Figure 7.4  Three-character CPC classes on the three-dimensional UMAP map")
_ = _fig3.update_traces(marker=dict(size=2))
_ = _fig3.update_layout(height=650, legend_title="class", template="plotly_white")
_fig3.show()
_fig3.write_html(os.path.join(FIG_DIR, "fig_7_4_umap3d_classes.html"))
print("saved -> figures/fig_7_4_umap3d_classes.html")


# ── 7.8  Radar chart of the portfolios of the eight largest assignees (plotly; also saved as HTML) ──
# Each axis is scaled to 0-1 across the eight assignees, so the chart compares profiles, not absolute sizes.
import plotly.graph_objects as go

TOP_N = 8
top_cos = result_df["current_assignee"].value_counts().head(TOP_N).index.tolist()

company_stats = []
for co in top_cos:
    co_rows  = result_df[result_df["current_assignee"] == co]
    hpc_rows = co_rows[~co_rows["merged_topic"].isin(["__noise__","Off-topic / Non-HPC","Unclassified",""])]
    company_stats.append({
        "company"       : co[:25],
        "patent_cnt"    : len(co_rows),
        "topic_breadth" : hpc_rows["merged_topic"].nunique(),
        "topic_depth"   : (hpc_rows["merged_topic"].value_counts().iloc[0] / len(hpc_rows) * 100
                           if len(hpc_rows) else 0),
        "hpc_share"    : len(hpc_rows) / len(co_rows) * 100 if len(co_rows) else 0,
        "ipc_diversity" : co_rows["main_ipc_4"].nunique(),
        "cluster_spread": co_rows[co_rows["cluster_id"] != -1]["cluster_id"].nunique(),
    })
stats_df = pd.DataFrame(company_stats)

# Normalize each metric to 0-1
def norm_col(col):
    lo, hi = col.min(), col.max()
    return pd.Series([0.5]*len(col)) if hi == lo else (col - lo) / (hi - lo)

METRICS = ["patent_cnt","topic_breadth","topic_depth","hpc_share",
           "ipc_diversity","cluster_spread"]
for mcol in METRICS:
    stats_df[f"n_{mcol}"] = norm_col(stats_df[mcol])

categories = ["patents", "topic breadth<br>(number of topics)", "topic depth<br>(share of the largest topic)",
              "share with a topic", "class diversity<br>(four-character codes)", "cluster spread"]
colors = PALETTE

fig = go.Figure()
for i, row in stats_df.iterrows():
    vals = [row[f"n_{m}"] for m in METRICS]; vals += [vals[0]]
    cats = categories + [categories[0]]
    _ = fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself", name=row["company"],
        line=dict(color=colors[i % len(colors)], width=2), opacity=0.6))
_ = fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                               tickvals=[0.25, 0.5, 0.75, 1.0],
                               ticktext=["25%","50%","75%","100%"])),
    title="Figure 7.5  Portfolio profiles of the eight largest assignees (each axis scaled to 0-1)",
    template="plotly_white",
    height=600, showlegend=True)
fig.show()

print(f"\n  {'company':28} {'patents':>8} {'breadth':>8} {'HPC %':>8} {'IPC div':>8} {'clusters':>9}")
for _, row in stats_df.iterrows():
    print(f"  {row['company']:28} {row['patent_cnt']:>8} {row['topic_breadth']:>8} "
          f"{row['hpc_share']:>7.1f}% {row['ipc_diversity']:>8} {row['cluster_spread']:>9}")

fig.write_html(os.path.join(FIG_DIR, "fig_7_5_assignee_radar.html"))
print("saved -> figures/fig_7_5_assignee_radar.html")
stats_df[["company"]+METRICS].to_csv(
    os.path.join(SAVE_DIR, "company_radar_stats.csv"), index=False)


# ── 7.9  Figures 7.6 and 7.7: assignees grouped by their cluster portfolios ──
# Each of the fifteen largest assignees is described by the share of its records in each cluster.
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Patch

plot_df = cluster_df[cluster_df["current_assignee"].isin(top_companies)].copy()

pivot_df = (plot_df.groupby(["current_assignee", "cluster_id"]).size()
            .reset_index(name="count")
            .pivot(index="current_assignee", columns="cluster_id", values="count")
            .fillna(0))
pivot_ratio = pivot_df.div(pivot_df.sum(axis=1), axis=0)

SHORT_MAP = {
    "Taiwan Semiconductor Manufacturing Co TSMC Ltd": "TSMC",
    "International Business Machines Corp": "IBM",
    "Advanced Micro Devices Inc": "AMD",
    "Micron Technology Inc": "Micron",
    "Nvidia Corp": "Nvidia",
    "Intel Corp": "Intel",
    "Amazon Technologies Inc": "Amazon",
    "Samsung Electronics Co Ltd": "Samsung",
    "SK Hynix Inc": "SK Hynix",
    "Google LLC": "Google",
    "Oracle International Corp": "Oracle",
    "Pure Storage Inc": "Pure Storage",
    "Cerebras Systems Inc": "Cerebras",
    "Psiquantum Corp": "PsiQuantum",
    "EMC Corp": "EMC",
    "Nutanix Inc": "Nutanix",
    "Lodestar Licensing Group LLC": "Lodestar",
    "Everpure Inc": "Everpure",
    "Microsoft Technology Licensing LLC": "Microsoft",
    "Hewlett Packard Enterprise Development LP": "HPE",
    "Saudi Arabian Oil Co": "Saudi Aramco",
}
shorten_company = lambda name: SHORT_MAP.get(name, name[:20])
pivot_ratio.index = [shorten_company(x) for x in pivot_ratio.index]

# Assignee types, set by hand for the largest companies (Everpure is the current name of Pure Storage); others are "Other"
COMPANY_GROUP = {
    "Micron": "Memory / Storage", "SK Hynix": "Memory / Storage",
    "Pure Storage": "Memory / Storage", "Everpure": "Memory / Storage", "EMC": "Memory / Storage",
    "TSMC": "Semiconductor / Manufacturing", "Samsung": "Semiconductor / Manufacturing",
    "Intel": "Compute / Platform", "AMD": "Compute / Platform",
    "IBM": "Compute / Platform", "Nvidia": "Compute / Platform",
    "Amazon": "Cloud / AI Application", "Google": "Cloud / AI Application",
    "Oracle": "Cloud / AI Application", "Nutanix": "Cloud / AI Application",
    "Cerebras": "Specialized Startup", "PsiQuantum": "Specialized Startup",
    "Lodestar": "Specialized Startup",
}
GROUP_COLOR = {
    "Memory / Storage": PALETTE[0], "Semiconductor / Manufacturing": PALETTE[1],
    "Compute / Platform": PALETTE[2], "Cloud / AI Application": PALETTE[3],
    "Specialized Startup": PALETTE[6], "Other": OTHER_GREY,
}

Z = linkage(pivot_ratio.values, method="ward")
fig, ax = plt.subplots(figsize=(9.6, 4.4))
ddata = dendrogram(Z, labels=pivot_ratio.index.tolist(), leaf_rotation=90, leaf_font_size=8,
                   color_threshold=0, above_threshold_color=INK_2, ax=ax)
# a coloured square under each leaf shows the assignee type; the label itself stays in the text colour
_xt = ax.get_xticks()
for _x, _name in zip(_xt, ddata["ivl"]):
    _ = ax.scatter(_x, 0, marker="s", s=36, color=GROUP_COLOR[COMPANY_GROUP.get(_name, "Other")],
                   clip_on=False, zorder=3, edgecolors="white", linewidths=0.6)
_ = ax.legend(handles=[Patch(facecolor=c, label=g) for g, c in GROUP_COLOR.items()],
              title="assignee type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=7)
_ = ax.set_ylabel("Ward distance")
_ = ax.set_title(f"Figure 7.6  The {len(pivot_ratio)} largest assignees grouped by their cluster portfolios")
fig.tight_layout()
_ = save_fig(fig, "fig_7_6_assignee_dendrogram")
plt.show()

ordered = ddata["ivl"]
g = sns.clustermap(
    pivot_ratio.loc[ordered], row_cluster=False, col_cluster=False,
    row_colors=[GROUP_COLOR[COMPANY_GROUP.get(n, "Other")] for n in ordered],
    cmap=SEQ_CMAP, figsize=(10, 5.4), linewidths=0.3, linecolor="white",
    dendrogram_ratio=(0.1, 0.02),   # no dendrograms are drawn here; keep only room for the colour bar
    cbar_pos=(0.02, 0.3, 0.015, 0.35), cbar_kws={"label": "share of records"})
_ = g.ax_heatmap.set_title("Figure 7.7  Share of each assignee's records in each cluster (rows in dendrogram order)",
                           pad=14, loc="left", fontsize=10)
_ = g.ax_heatmap.set_xlabel("cluster_id"); _ = g.ax_heatmap.set_ylabel("")
g.savefig(os.path.join(FIG_DIR, "fig_7_7_assignee_cluster_heatmap.png"))
print("saved -> figures/fig_7_7_assignee_cluster_heatmap.png")
plt.show()


# ── 7.10  Figure 7.8: self-organising map of the assignee portfolios ─────────
from minisom import MiniSom

X_som = pivot_ratio.values
names = pivot_ratio.index.tolist()

som_x, som_y = 5, 5
som = MiniSom(x=som_x, y=som_y, input_len=X_som.shape[1],
              sigma=1.0, learning_rate=0.5, random_seed=RANDOM_STATE)
som.random_weights_init(X_som)
som.train_random(X_som, 1000)

fig, ax = plt.subplots(figsize=(6.0, 5.4))
_pc = ax.pcolor(som.distance_map().T, cmap=SEQ_CMAP, edgecolors="white", linewidth=0.5)
_ = fig.colorbar(_pc, ax=ax, label="distance to the neighbouring units (U-matrix)")
_cell_n = {}
for i, name in enumerate(names):
    w = som.winner(X_som[i])
    k = _cell_n.get(w, 0); _cell_n[w] = k + 1          # stack the names that fall into the same unit
    _ = ax.scatter(w[0] + 0.12, w[1] + 0.82 - 0.22 * k, marker="s", s=22,
                   color=GROUP_COLOR[COMPANY_GROUP.get(name, "Other")], edgecolors="white", linewidths=0.4)
    _ = ax.text(w[0] + 0.24, w[1] + 0.82 - 0.22 * k, name[:13], va="center", fontsize=6.5, color=INK)
_ = ax.set_xlim(0, som_x); _ = ax.set_ylim(0, som_y)
_ = ax.set_xticks([]); _ = ax.set_yticks([])
_ = ax.legend(handles=[Patch(facecolor=c, label=g) for g, c in GROUP_COLOR.items()],
              loc="upper left", bbox_to_anchor=(1.25, 1), fontsize=7, title="assignee type", title_fontsize=7)
_ = ax.set_title("Figure 7.8  Self-organising map (5 x 5) of the assignee portfolios")
fig.tight_layout()
_ = save_fig(fig, "fig_7_8_assignee_som")
plt.show()


# ── 7.11  Topic classifier: predict merged_topic from the synthesized embedding ──
# A convenience tool for users who want to assign new records to the topics. The topics come from clusters
# of the same embeddings, so this is circular and is not evidence for the topic layer (that is 7.16).
# The classifiers were chosen with their run time in mind: HistGradientBoostingClassifier bins the features
# and trained in about 35 s on the full corpus (the exact-split GradientBoostingClassifier took about 85 min);
# the SVM does not set probability=True, and 7.12 ranks by decision_function instead.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import normalize, LabelEncoder
import joblib

embed_df  = pd.read_csv(os.path.join(SAVE_DIR, "specter2_embeddings.csv"))
result_df = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")

hpc_df = result_df[~result_df["merged_topic"].isin(
    ["__noise__", "Off-topic / Non-HPC", "Unclassified"])].reset_index(drop=True)

embed_hpc = hpc_df[["patent_id"]].merge(embed_df, on="patent_id", how="left")
X = normalize(embed_hpc.drop(columns=["patent_id"]).values.astype(np.float32), norm="l2")
y = np.array(hpc_df["merged_topic"].values)

# Remove the topics with < 5 samples (a requirement of the stratified split)
topic_counts = pd.Series(y).value_counts()
mask = pd.Series(y).isin(topic_counts[topic_counts >= 5].index).values
X, y = X[mask], y[mask]
le = LabelEncoder(); y_enc = le.fit_transform(y)

# Family-grouped split (1.6): continuations have nearly identical text, so a random split leaks
hpc_mask_df = hpc_df[mask].reset_index(drop=True)
groups = build_family_groups(hpc_mask_df["patent_id"].values)
idx = np.arange(len(X))
idx_train, idx_test, y_train, y_test = family_train_test_split(idx, y_enc, groups)
X_train, X_test = X[idx_train], X[idx_test]
print(f"n={len(X)} | topics={len(set(y))} | families={pd.Series(groups).nunique()} | "
      f"train={len(X_train)} | test={len(X_test)}")
_miss_tr = len(set(y_enc)) - len(set(y_train)); _miss_te = len(set(y_enc)) - len(set(y_test))
if _miss_tr or _miss_te:
    print(f"  Note: a grouped split cannot stratify; train lacks {_miss_tr} classes, test lacks {_miss_te} classes"
          f" (macro-F1 with zero_division=0)")

# ── Safety checks (read-only; writes no file) ─────────────────────────────
assert hpc_df["patent_id"].is_unique, "patent_id has duplicates; check the upstream files first"
_n_topics = len(set(y))
if _n_topics > 50:
    raise RuntimeError(
        f"merged_topic has {_n_topics} values, so the topic merge of 6.9 has not run.\n"
        "Merge the topics to about 20-30 before you run this section: with too many classes, the training time of the gradient-boosted trees (one set of trees for each class) "
        "and of the SVM (support vector machine; one-vs-one needs N×(N-1)/2 binary classifiers) "
        "grows very quickly with the number of classes."
    )
# ──────────────────────────────────────────────────────────────────────────

models = {
    "Logistic Regression"   : LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    "Random Forest"         : RandomForestClassifier(n_estimators=200, max_depth=20,
                                                     random_state=RANDOM_STATE, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, max_depth=5,
                                                             learning_rate=0.1,
                                                             random_state=RANDOM_STATE),
    "SVM (RBF)"             : SVC(kernel="rbf", C=10, gamma="scale", cache_size=1000,
                                  random_state=RANDOM_STATE),
}

results = {}
print(f"\n  {'model':25} {'Acc':>9} {'F1 macro':>10} {'F1 weighted':>12} {'fit+pred':>10}")
for name, clf in models.items():
    with timed(f"7.11  {name}", n_items=len(X_train), quiet=True) as t:
        _ = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    results[name] = {
        "clf": clf,
        "acc": accuracy_score(y_test, y_pred),
        "f1_mac": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_wei": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_pred": y_pred,
        "sec": t.seconds,
    }
    r = results[name]
    print(f"  {name:25} {r['acc']*100:>8.1f}% {r['f1_mac']:>10.4f} "
          f"{r['f1_wei']:>12.4f} {r['sec']:>9.1f}s")

best_name = max(results, key=lambda k: results[k]["f1_mac"])
best = results[best_name]
print(f"\nbest model: {best_name} (F1 macro = {best['f1_mac']:.4f})")
print(classification_report(y_test, best["y_pred"], target_names=le.classes_,
                            zero_division=0))

from sklearn.model_selection import GroupKFold
with timed(f"7.11  5-fold GroupKFold CV ({best_name})", n_items=len(X)):
    cv_scores = cross_val_score(best["clf"], X, y_enc,
                                cv=GroupKFold(n_splits=5).split(X, y_enc, groups),
                                scoring="f1_macro", n_jobs=-1)
print(f"5-fold GroupKFold CV F1 macro: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
      f" (families as the fold unit)")

model_path = os.path.join(SAVE_DIR, f"classifier_{best_name.replace(' ', '_').lower()}.pkl")
_ = joblib.dump({"model": best["clf"], "label_encoder": le}, model_path)
print(f"saved -> {model_path}")

# The training cost of each classifier is saved with its scores
pd.DataFrame([{"model": k, "acc": round(v["acc"], 4),
               "f1_macro": round(v["f1_mac"], 4), "f1_weighted": round(v["f1_wei"], 4),
               "fit_predict_sec": round(v["sec"], 2), "hardware": HARDWARE}
              for k, v in results.items()]).to_csv(
    os.path.join(SAVE_DIR, "topic_classifier_benchmark.csv"), index=False)
print("saved -> topic_classifier_benchmark.csv")

# ── Leakage check: random split vs family-grouped split (the same SVM) ─────
# The difference between the two scores is the optimistic bias that the near-duplicate texts of family
# continuations cause under a random split.
with timed("7.11  Leakage check (random vs family split, SVM)", n_items=len(X)):
    _tr_r, _te_r, _ytr_r, _yte_r = train_test_split(
        np.arange(len(X)), y_enc, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_enc)
    _clf_r  = SVC(kernel="rbf", C=10, gamma="scale", cache_size=1000,
                  random_state=RANDOM_STATE).fit(X[_tr_r], _ytr_r)
    _f1_rand = f1_score(_yte_r, _clf_r.predict(X[_te_r]),
                        average="macro", zero_division=0)
print(f"SVM macro-F1 with random split {_f1_rand:.4f}  vs  with family-grouped split "
      f"{results['SVM (RBF)']['f1_mac']:.4f} (difference = optimistic bias from family leakage)")


# ── 7.12  Batch prediction example and accuracy by assignee and class ───────
def predict_batch(df, top_k=3):
    pred_embed = df[["patent_id"]].merge(embed_df, on="patent_id", how="left")
    missing = pred_embed.drop(columns=["patent_id"]).isna().all(axis=1)
    if missing.any():
        raise ValueError(f"missing embeddings for {missing.sum()} patent_ids")
    X_pred = normalize(pred_embed.drop(columns=["patent_id"]).values.astype(np.float32),
                       norm="l2")
    clf = best["clf"]
    y_pred = le.inverse_transform(clf.predict(X_pred))
    # top-k: use probabilities when they are available; to save about 5.7 times the training time, the SVM does not set
    # probability=True and sorts by the one-vs-rest scores of decision_function instead;
    # the ranking is the same as with probabilities (only no calibrated probability values are output).
    scores = None
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_pred)
    elif hasattr(clf, "decision_function"):
        d = clf.decision_function(X_pred)
        scores = d if d.ndim == 2 else np.column_stack([-d, d])
    if scores is None:
        return y_pred, [[p] for p in y_pred]
    top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
    top_k_labels = [list(le.inverse_transform(r)) for r in top_idx]
    return y_pred, top_k_labels

INFER_N   = min(700, len(hpc_mask_df))
sample_df = hpc_mask_df.sample(INFER_N, random_state=RANDOM_STATE).reset_index(drop=True)
with timed(f"7.12  Batch inference example ({best_name})", n_items=INFER_N):
    preds, top3s = predict_batch(sample_df)
sample_df["predicted_topic"] = preds
sample_df["top3"]    = top3s
sample_df["correct"] = sample_df["predicted_topic"] == sample_df["merged_topic"]

acc = sample_df["correct"].mean()
print(f"sample accuracy: {acc*100:.1f}%  ({sample_df['correct'].sum()}/{INFER_N})")

print("\nper-company accuracy (top 10 by volume):")
co_acc = sample_df.groupby("current_assignee")["correct"].agg(total="count", correct="sum")
co_acc["accuracy"] = (co_acc["correct"] / co_acc["total"] * 100).round(1)
print(co_acc.sort_values("total", ascending=False).head(10).to_string())

print("\nper-IPC accuracy (top 10 by volume):")
ipc_acc = sample_df.groupby("main_ipc_4")["correct"].agg(total="count", correct="sum")
ipc_acc["accuracy"] = (ipc_acc["correct"] / ipc_acc["total"] * 100).round(1)
print(ipc_acc.sort_values("total", ascending=False).head(10).to_string())

sample_df[["patent_id","current_assignee","main_ipc_4","merged_topic",
           "predicted_topic","top3","correct"]].to_csv(
    os.path.join(SAVE_DIR, "inference_results.csv"), index=False)

cross = (sample_df.groupby(["current_assignee","predicted_topic","main_ipc_4"])
         .size().reset_index(name="count"))
cross.to_csv(os.path.join(SAVE_DIR, "cross_analysis.csv"), index=False)
print("\nsaved -> inference_results.csv, cross_analysis.csv")


# ── 7.13  Usage example: topics of the litigated families ───────────────────
# Reads litigation_family.csv of 3.8. Descriptive only: the topics of the litigated families and their
# citation counts next to the whole corpus. The subset is small, so there is no test and no causal reading.
# citation_count is not in m1 (3.7), so it is read from the candidate file.
_lit_path = os.path.join(SAVE_DIR, "litigation_family.csv")
if not os.path.exists(_lit_path):
    raise RuntimeError("litigation_family.csv not found; run 3.8 first.")
lit_df = pd.read_csv(_lit_path).fillna("")
_clu   = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_meta  = pd.read_csv(CLEAN_PATH, usecols=["patent_id", "citation_count"]).fillna(0)

with timed("7.13  Litigated-family landscape", n_items=len(lit_df)):
    lit_ids  = set(lit_df["patent_id"])
    _n_seed  = int((lit_df["is_seed"].astype(str).str.lower() == "true").sum())
    _dfm = _clu[["patent_id", "merged_topic"]].merge(_meta, on="patent_id", how="left")
    _dfm = _dfm[~_dfm["merged_topic"].isin(["__noise__", "Off-topic / Non-HPC", "Unclassified", ""])]
    _dfm["litigated"] = _dfm["patent_id"].isin(lit_ids)
    n_lit = int(_dfm["litigated"].sum())

    print(f"Litigated-family subset: {len(lit_df)} records (seed {_n_seed} records, "
          f"{lit_df['family_root'].nunique()} families, "
          f"{lit_df['case_group'].nunique()} case groups); {n_lit} of them have a topic label")

    topic_tab = (_dfm.groupby("merged_topic")["litigated"]
                 .agg(n_patents="size", n_litigated="sum"))
    topic_tab["litigated_pct"] = (topic_tab["n_litigated"] / topic_tab["n_patents"] * 100).round(2)
    topic_tab = topic_tab.sort_values("n_litigated", ascending=False)
    _hit = topic_tab[topic_tab["n_litigated"] > 0]
    print(f"\nThe litigated families fall in {len(_hit)} topics (only topics with litigation are listed; "
          f"at this sample size the percentages only show the position and must not be read as risk rates):")
    print(_hit.to_string())

    cc = pd.to_numeric(_dfm["citation_count"], errors="coerce").fillna(0)
    print(f"\ncitation_count median: litigated families {cc[_dfm['litigated']].median():.0f}"
          f" (n={n_lit}) vs all {cc.median():.0f} (n={len(_dfm)})"
          f"  — descriptive comparison; the sample is too small for a statistical test; no causal claim")

topic_tab.reset_index().to_csv(os.path.join(SAVE_DIR, "usage_litigation_topics.csv"),
                               index=False)
print("saved -> usage_litigation_topics.csv")


# ── 7.14  Usage example: topic shares by application year (Figure 7.9) ───────
# Descriptive only. The corpus holds granted patents, and many recent applications are still under
# examination, so the drop after 2022 is the grant delay, not a change in technology. Any yearly chart of
# this corpus needs this right-truncation note.
_pipe = pd.read_csv(os.path.join(SAVE_DIR, "patent_full_pipeline.csv"),
                    usecols=["patent_id", "application_year"]).fillna("")
_clu  = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_dfm  = _pipe.merge(_clu[["patent_id", "merged_topic"]], on="patent_id", how="inner")
_dfm  = _dfm[~_dfm["merged_topic"].isin(["__noise__", "Off-topic / Non-HPC", "Unclassified", ""])]
_dfm["application_year"] = pd.to_numeric(_dfm["application_year"], errors="coerce")

with timed("7.14  Topic trends by filing year", n_items=len(_dfm)):
    top8 = _dfm["merged_topic"].value_counts().head(8).index
    tab = (_dfm[_dfm["merged_topic"].isin(top8)]
           .pivot_table(index="application_year", columns="merged_topic",
                        values="patent_id", aggfunc="count").fillna(0))
    share = tab.div(tab.sum(axis=1), axis=0) * 100              # columns in alphabetical order (as written to the CSV)
_plot_share = share[[t for t in top8 if t in share.columns]]    # figure: fixed colour order, largest topic first

fig, ax = plt.subplots(figsize=(8.4, 3.6))
_ = ax.stackplot(_plot_share.index, [_plot_share[c].values for c in _plot_share.columns], labels=_plot_share.columns,
                 colors=PALETTE[:len(_plot_share.columns)], edgecolor="white", linewidth=0.6)
_ = ax.axvspan(2022.5, share.index.max() + 0.5, color="white", alpha=0.45, zorder=3, linewidth=0)
_ = ax.axvline(2022.5, color=INK_2, linestyle="--", linewidth=0.8, zorder=4)
_ = ax.text(2022.6, 101, "right-truncated: granted patents only", fontsize=7, color=INK_2, va="bottom")
_ = ax.set_xlim(share.index.min(), share.index.max())
_ = ax.set_ylim(0, 100)
_ = ax.set_ylabel("share of the eight largest topics (percent)")
_ = ax.set_xlabel("application year")
_ = ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
_ = ax.set_title("Figure 7.9  Shares of the eight largest topics by application year", pad=14)
fig.tight_layout()
_ = save_fig(fig, "fig_7_9_topic_trends")
plt.show()
share.round(2).to_csv(os.path.join(SAVE_DIR, "usage_topic_trends.csv"))
print("saved -> usage_topic_trends.csv")


# ── 7.15  Usage example: citations between topics inside the corpus ─────────
# The citations column is parsed, and an edge is kept when the cited patent is also a released record; the
# edges are then counted by citing and cited topic. citations is read from the candidate file and limited to
# released_corpus_ids.csv (3.5), so "inside the corpus" means the released records.
import ast as _ast

_rel  = set(pd.read_csv(os.path.join(SAVE_DIR, "released_corpus_ids.csv"),
                        dtype=str)["patent_id"])
_pipe = pd.read_csv(CLEAN_PATH, usecols=["patent_id", "citations"], dtype=str).fillna("")
_pipe = _pipe[_pipe["patent_id"].isin(_rel)].reset_index(drop=True)
_clu  = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_topic = dict(zip(_clu["patent_id"], _clu["merged_topic"]))

with timed("7.15  Citation flows between topics", n_items=len(_pipe)):
    in_corpus = set(_pipe["patent_id"])
    edges = []
    for pid, cit in tqdm(zip(_pipe["patent_id"], _pipe["citations"]),
                         total=len(_pipe), desc="parse citations"):
        t_src = _topic.get(pid, "")
        if not t_src or t_src in ("__noise__", "Off-topic / Non-HPC", "Unclassified"):
            continue
        try:
            items = _ast.literal_eval(cit) if cit.startswith("[") else []
        except (ValueError, SyntaxError):
            continue
        for it in items:
            tgt = str(it.get("publication_number", ""))
            if tgt in in_corpus:
                t_tgt = _topic.get(tgt, "")
                if t_tgt and t_tgt not in ("__noise__", "Off-topic / Non-HPC", "Unclassified"):
                    edges.append((t_src, t_tgt))
    print(f"citation edges within the corpus: {len(edges)} (corpus scope: released corpus, {len(in_corpus):,} records)")
    flow = pd.DataFrame(edges, columns=["citing_topic", "cited_topic"])
    mat = pd.crosstab(flow["citing_topic"], flow["cited_topic"])
    diag = sum(mat.loc[t, t] for t in mat.index if t in mat.columns)
    print(f"share of citations within the same topic: {diag / max(len(edges), 1) * 100:.1f}%")
    off = flow[flow["citing_topic"] != flow["cited_topic"]]
    print("\nTop 10 citation flows between topics:")
    print(off.value_counts().head(10).to_string())
mat.to_csv(os.path.join(SAVE_DIR, "usage_citation_flows.csv"))
print("saved -> usage_citation_flows.csv")


# ── 7.16  Agreement of the topics with the patent classification (CPC) ────────
# The topics of section 6 come from clusters of the summary embeddings, and their construction uses no
# classification code (the class columns come from the first CPC path on Google Patents; the column names keep
# "ipc"). The agreement of the two is therefore an external check of whether the topics follow a technical
# structure:
#   no relation at all → the topics can be only a product of the model;
#   complete agreement → the semantic layer gives no information beyond the administrative classification.
# Both directions have a meaning, so this section only reports the measured values and sets no pass threshold.
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                            homogeneity_score, completeness_score)

EXCLUDE_TOPICS = ["__noise__", "Off-topic / Non-HPC", "Unclassified", ""]
TBL_DIR = os.path.join(SAVE_DIR, "tables")
os.makedirs(TBL_DIR, exist_ok=True)

# ── Build the data: topic labels × four-character patent class (CPC) ────────────
_clu = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
if "main_ipc_4" in _clu.columns:
    _dfm = _clu[["patent_id", "cluster_id", "merged_topic", "main_ipc_4"]].copy()
else:
    _ipc = pd.read_csv(CLEAN_PATH, usecols=["patent_id", "main_ipc_4"],
                       low_memory=False).fillna("")
    _dfm = _clu[["patent_id", "cluster_id", "merged_topic"]].merge(
        _ipc, on="patent_id", how="inner")

_n_all = len(_dfm)
_dfm = _dfm[~_dfm["merged_topic"].isin(EXCLUDE_TOPICS)]
_dfm = _dfm[_dfm["main_ipc_4"].astype(str).str.strip() != ""].reset_index(drop=True)

print(f"clustering result           : {_n_all:,} records")
print(f"with topic and class        : {len(_dfm):,} records"
      f" ({_n_all - len(_dfm):,} records excluded: noise, no topic or no class)")
print(f"topic coverage              : {len(_dfm)/_n_all*100:.1f}%"
      f"  <- records without a topic label are not in any metric below")
print(f"distinct topics             : {_dfm['merged_topic'].nunique():,}")
_dfm = add_ipc3(_dfm)
print(f"distinct IPC four-character : {_dfm['main_ipc_4'].nunique():,}")
print(f"distinct IPC three-character: {_dfm['main_ipc_3'].nunique():,}"
      f"  <- closer to the granularity of the {_dfm['merged_topic'].nunique():,} topics")

# Agreement at the three-character scale: the numbers of topics and of three-character classes have a similar order of magnitude; this is the fairer external check
_t3 = _dfm["merged_topic"].astype("category").cat.codes.to_numpy()
_i3 = _dfm["main_ipc_3"].astype("category").cat.codes.to_numpy()
print(f"\nThree-character scale: NMI {normalized_mutual_info_score(_t3, _i3):.4f}"
      f" | ARI {adjusted_rand_score(_t3, _i3):.4f}"
      f" | homogeneity {homogeneity_score(_i3, _t3):.4f}"
      f" | completeness {completeness_score(_i3, _t3):.4f}")
print("(the four-character scale below is the fine-grained comparison)")

with timed("7.16  Agreement of topics and classification", n_items=len(_dfm)):
    _t = _dfm["merged_topic"].astype("category").cat.codes.to_numpy()
    _i = _dfm["main_ipc_4"].astype("category").cat.codes.to_numpy()

    # ── Table 1  Overall agreement metrics ───────────────────────────────────
    _metrics = [
        ("normalized mutual information (NMI)",  normalized_mutual_info_score(_t, _i),
         "0 = independent partitions; 1 = identical partitions"),
        ("adjusted Rand index (ARI)", adjusted_rand_score(_t, _i),
         "corrected for chance; 0 = no better than random"),
        ("homogeneity", homogeneity_score(_i, _t),
         "whether each topic holds a single class only"),
        ("completeness", completeness_score(_i, _t),
         "whether each class falls in a single topic only"),
    ]
    # Random baseline: compute again after the topic labels are shuffled, as a comparison
    _rng = np.random.default_rng(RANDOM_STATE)
    _shuf = _rng.permutation(_t)
    _base_nmi = normalized_mutual_info_score(_shuf, _i)
    _base_ari = adjusted_rand_score(_shuf, _i)

    t_metric = pd.DataFrame([(k, round(v, 4), d) for k, v, d in _metrics],
                            columns=["metric", "value", "note"])

    # ── Table 2  Class concentration of each topic ───────────────────────────
    _rows = []
    for topic, grp in _dfm.groupby("merged_topic"):
        vc = grp["main_ipc_4"].value_counts()
        _rows.append([topic, len(grp), vc.index[0], round(vc.iloc[0] / len(grp) * 100, 1),
                      grp["main_ipc_4"].nunique()])
    t_topic = (pd.DataFrame(_rows, columns=["topic", "records", "largest_cpc_subclass", "largest_cpc_subclass_share_pct",
                                           "distinct_cpc_subclasses"])
               .sort_values("records", ascending=False).reset_index(drop=True))

    # ── Table 3  Number of topics over which each main class spreads ─────────
    _rows = []
    for ipc, grp in _dfm.groupby("main_ipc_4"):
        if len(grp) < 30:                      # a class that is too small is not representative
            continue
        vc = grp["merged_topic"].value_counts()
        _rows.append([ipc, len(grp), vc.index[0], round(vc.iloc[0] / len(grp) * 100, 1),
                      grp["merged_topic"].nunique()])
    t_ipc = (pd.DataFrame(_rows, columns=["cpc_subclass", "records", "largest_topic", "largest_topic_share_pct",
                                          "distinct_topics"])
             .sort_values("records", ascending=False).reset_index(drop=True))

# ── Report ───────────────────────────────────────────────────────────────────
print("\nTable 1  Overall agreement metrics")
print(t_metric.to_string(index=False))
print(f"\n  random baseline (shuffled topic labels): NMI {_base_nmi:.4f} and ARI {_base_ari:.4f}")
print(f"  actual value above the random baseline : NMI +{_metrics[0][1] - _base_nmi:.4f} and "
      f"ARI +{_metrics[1][1] - _base_ari:.4f}")

_dom = t_topic["largest_cpc_subclass_share_pct"]
print(f"\nTable 2  Class concentration of each topic ({len(t_topic)} topics)")
print(f"  share of the largest class: median {_dom.median():.1f}%, "
      f"quartiles {_dom.quantile(.25):.1f}%–{_dom.quantile(.75):.1f}%")
print(f"  topics with a single-class share ≥ 50%: {int((_dom >= 50).sum())}"
      f" ({(_dom >= 50).mean()*100:.0f}%)")
print(f"  mean number of classes covered: {t_topic['distinct_cpc_subclasses'].mean():.1f}")
print("\n  The 15 topics with the most records:")
print(t_topic.head(15).to_string(index=False))

print(f"\nTable 3  Topic dispersion of the main classes (classes with ≥ 30 records: {len(t_ipc)})")
print(f"  median share of the largest topic: {t_ipc['largest_topic_share_pct'].median():.1f}%")
print(f"  spread over {t_ipc['distinct_topics'].mean():.1f} topics on average")
print(t_ipc.head(15).to_string(index=False))

t_metric.to_csv(os.path.join(TBL_DIR, "e6_topic_ipc_metrics.csv"), index=False)
t_topic.to_csv(os.path.join(TBL_DIR, "e6_topic_ipc_concentration.csv"), index=False)
t_ipc.to_csv(os.path.join(TBL_DIR, "e6_ipc_topic_dispersion.csv"), index=False)
print(f"\nsaved -> tables/e6_topic_ipc_metrics.csv, "
      f"e6_topic_ipc_concentration.csv, e6_ipc_topic_dispersion.csv")

# ── Table 4  Largest-class share of each topic and its distribution ────────
# Definition:
#   largest-class share of a topic = records of the most frequent class in the topic ÷ all records of the topic.
#   overall weighted share         = sum of the largest-class records of all topics ÷ all records.
# The reference point for the reading is the random baseline, and the baseline is not a number that someone selected: it is equal to "the share of the largest class in the whole
# corpus". This means: if topics are assigned at random without any look at the content, the most frequent class of each topic already
# has this share. Thus
#   share > baseline → the topic is more concentrated than a random assignment, and the administrative classification captures it;
#   share < baseline → the topic is more spread than a random assignment, and the administrative classification does not capture it.
# Both directions have a meaning (too low means that the topics are unrelated to the administrative classification; too high means that the semantic layer gives no information
# beyond the classification), so only the values and the distribution are reported, with no pass threshold.

def ipc_top_share(frame, ipc_col, topic_col="merged_topic"):
    _g   = frame.groupby(topic_col)[ipc_col]
    _per = pd.DataFrame({
        "n":     _g.size(),
        "top":   _g.agg(lambda s: s.value_counts().index[0]),
        "top_n": _g.agg(lambda s: int(s.value_counts().iloc[0])),
        "n_ipc": _g.nunique(),
    })
    _per["top_share"] = _per["top_n"] / _per["n"]
    return float(_per["top_n"].sum() / _per["n"].sum()), _per.sort_values("n", ascending=False)

_rng2  = np.random.default_rng(RANDOM_STATE)
_share_tables = {}
print("\n" + "=" * 70)
print("Table 4  Largest-class share of each topic")
print("=" * 70)
for _col, _name in (("main_ipc_3", "three-character classes"), ("main_ipc_4", "four-character subclasses")):
    if _col not in _dfm.columns:
        continue
    _ov, _per = ipc_top_share(_dfm, _col)
    # Random baseline: compute again after the topic labels are shuffled. Its value is equal to the share of the largest class in the whole corpus,
    # that is, the share that the most frequent class of each topic already has when topics are assigned at random "without any look at the content".
    _shufd = _dfm.assign(merged_topic=_rng2.permutation(_dfm["merged_topic"].values))
    _ov_r, _ = ipc_top_share(_shufd, _col)
    _p    = _per["top_share"]
    _big  = _dfm[_col].value_counts()
    print(f"\n{_name} ({_dfm[_col].nunique()} classes, {len(_per)} topics)")
    print(f"  largest class         : {_big.index[0]}  {_big.iloc[0]:,} records / {len(_dfm):,} records"
          f" = {_big.iloc[0]/len(_dfm):.4f}  ← this number is the random baseline")
    print(f"  overall weighted share: {_ov:.4f}   random baseline {_ov_r:.4f}   above by {_ov - _ov_r:+.4f}")
    print(f"  share distribution    : min {_p.min():.3f} | 25th percentile {_p.quantile(.25):.3f} | "
          f"median {_p.median():.3f} | 75th percentile {_p.quantile(.75):.3f} | max {_p.max():.3f}")
    _bins = [0, .2, .4, .6, .8, 1.0001]
    _lab  = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    _hist = pd.cut(_p, bins=_bins, labels=_lab, right=False).value_counts().reindex(_lab).fillna(0)
    print("  histogram:")
    for _b, _n in _hist.items():
        print(f"    {_b}  {int(_n):>3}  {'█' * int(_n)}")
    _n_lo = int((_p < _ov_r).sum())
    print(f"  topics below baseline : {_n_lo} / {len(_per)}"
          f" (covering {int(_per.loc[_p < _ov_r, 'n'].sum()):,} records, "
          f"{_per.loc[_p < _ov_r, 'n'].sum() / len(_dfm) * 100:.1f}%)"
          f" ← the administrative classification does not capture these topics")
    print(f"  the 5 topics with the highest share (the administrative classification captures them; a scan by class code is sufficient):")
    for _r in _per.sort_values('top_share', ascending=False).head(5).itertuples():
        print(f"    {str(_r.Index)[:42]:44} n={_r.n:>4}  {_r.top_share:.3f}  ({_r.top})")
    print(f"  the 5 topics with the lowest share (those below the baseline are not captured by the administrative classification; see the line above):")
    for _r in _per.sort_values('top_share').head(5).itertuples():
        print(f"    {str(_r.Index)[:42]:44} n={_r.n:>4}  {_r.top_share:.3f}  ({_r.top}; covers {_r.n_ipc} classes)")
    _per.reset_index().to_csv(os.path.join(TBL_DIR, f"e6_top_share_{_col}.csv"), index=False)
    _share_tables[_col] = {"overall": round(_ov, 4), "random_baseline": round(_ov_r, 4),
                         "n_topics": int(len(_per)), "n_ipc": int(_dfm[_col].nunique())}

pd.DataFrame(_share_tables).T.reset_index().rename(columns={"index": "ipc_level"}).to_csv(
    os.path.join(TBL_DIR, "e6_top_share_summary.csv"), index=False)
print("\nsaved -> tables/e6_top_share_main_ipc_3.csv, e6_top_share_main_ipc_4.csv,\n         e6_top_share_summary.csv")


# ── 7.17  Usage example: one landscape scan done in two ways (CPU only) ─────
# The same landscape scan is done twice, with only two columns changed, and then the differences are shown:
#   Approach A (current practice)  current_assignee   + patent classification (CPC)
#   Approach B (this data)         effective_assignee + semantic topics
# This section only reports what each approach gives. It evaluates no company and makes no causal claim.
#
# The threshold of the screening rule is not selected by people. It is the random baseline of this corpus: the share of the largest single class in the whole corpus.
# When a topic has "a largest-class share < random baseline", it is more spread than a random assignment,
# and the administrative classification cannot resolve it; otherwise a scan by class code is sufficient, and the semantic layer is not necessary.
import os
import numpy as np, pandas as pd

TOP_N_CO    = 5     # number of companies in the company × topic matrix
TOP_HOLDERS = 3     # number of holders listed for each topic
TOP_N_RANK  = 10    # rows of the assignee ranking comparison
NON_TOPIC   = ["__noise__", "Off-topic / Non-HPC", "Unclassified", ""]

TBL_DIR = os.path.join(SAVE_DIR, "tables"); os.makedirs(TBL_DIR, exist_ok=True)

_clu = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_clu["patent_id"] = _clu["patent_id"].astype(str)
if "main_ipc_3" not in _clu.columns:
    _clu = add_ipc3(_clu)

_own_path = os.path.join(SAVE_DIR, "ownership_events.csv")
if not os.path.exists(_own_path):
    raise RuntimeError("ownership_events.csv not found; run 7.1 first.")
_own = pd.read_csv(_own_path).fillna("")
_own["patent_id"] = _own["patent_id"].astype(str)
_df = _clu.merge(_own[["patent_id", "effective_assignee", "change_supported"]],
                 on="patent_id", how="left")
_df["effective_assignee"] = (_df["effective_assignee"].fillna("").astype(str).str.strip()
                             .where(lambda s: s != "", _df["current_assignee"]))

# ── Table S1  Top ten assignees: before vs after the correction ─────────────
_cur = _df["current_assignee"].astype(str).str.strip()
_eff = _df["effective_assignee"].astype(str).str.strip()
_a = _cur[_cur != ""].value_counts().head(TOP_N_RANK)
_b = _eff[_eff != ""].value_counts().head(TOP_N_RANK)
_K = min(TOP_N_RANK, len(_a), len(_b))          # no failure when there are fewer distinct assignees than TOP_N_RANK
_a, _b = _a.head(_K), _b.head(_K)
_S1 = pd.DataFrame({
    "rank": range(1, _K + 1),
    "current_assignee": _a.index, "n_before": _a.values,
    "effective_assignee": _b.index, "n_after": _b.values})
_S1["changed"] = np.where(
    _S1["current_assignee"] != _S1["effective_assignee"], "yes", "")
print("Table S1  Top ten assignees: before vs after the correction")
print(_S1.to_string(index=False))
_gone = [x for x in _a.index if x not in set(_b.index)]
_new  = [x for x in _b.index if x not in set(_a.index)]
if _gone or _new:
    print(f"  dropped out of the top {_K} ranks: {_gone}")
    print(f"  entered the top {_K} ranks: {_new}")
_n_changed = int((_cur != _eff).sum())
print(f"\n  patents with a changed assignee: {_n_changed:,} ({_n_changed/len(_df)*100:.1f}% of the whole corpus)")
(_S1.rename(columns={"rank": "rank", "changed": "changed"})
    .replace({"changed": {"yes": "yes", "": "no"}})
    .to_csv(os.path.join(SAVE_DIR, "usage_assignee_before_after.csv"), index=False))

# ── Topic subset ────────────────────────────────────────────────────────────
_t = _df[~_df["merged_topic"].isin(NON_TOPIC)].copy()
_t = _t[_t["main_ipc_3"].astype(str).str.strip() != ""].reset_index(drop=True)
_N = len(_t)
print(f"\nwith topic and class: {_N:,} records (coverage {_N/len(_df)*100:.1f}%) | "
      f"{_t['merged_topic'].nunique()} topics")

# ── Random baseline = share of the largest single class in the whole corpus (equal to the baseline of Table 4 in 7.16) ───
_BASE = {}
for _lvl in ("main_ipc_3", "main_ipc_4"):
    _BASE[_lvl] = _t[_lvl].value_counts().iloc[0] / _N
print(f"random baseline (share of the largest single class in the whole corpus): "
      f"three-character {_BASE['main_ipc_3']:.4f} | four-character {_BASE['main_ipc_4']:.4f}")
print("  (this value is the screening threshold; it equals the random baseline of Table 4 in 7.16)")

# ── Table S2  Topic screening table ─────────────────────────────────────────
_rows = []
for _tp, _g in _t.groupby("merged_topic"):
    _r = {"topic": _tp, "records": len(_g)}
    for _lvl, _tag in (("main_ipc_3", "three-character"), ("main_ipc_4", "four-character")):
        _vc = _g[_lvl].value_counts()
        _r[f"largest {_tag} class"] = _vc.index[0]
        _r[f"{_tag} share"]    = round(_vc.iloc[0] / len(_g), 3)
        _r[f"{_tag} classes"]  = int(_vc.size)
    _hold = _g["effective_assignee"].astype(str).str.strip()
    _hold = _hold[_hold != ""].value_counts().head(TOP_HOLDERS)
    _r["top three holders"] = "; ".join(f"{k[:26]} {v}" for k, v in _hold.items())
    _r["largest holder share"] = round(_hold.iloc[0] / len(_g), 3) if len(_hold) else np.nan
    _rows.append(_r)

# The threshold is a reference point, not a dichotomy: also output the "distance to the baseline" and a borderline mark,
# so that users see the whole ranking and not only the cut point. A topic within MARGIN is borderline and is not a firm conclusion.
MARGIN = 0.03
_S2 = pd.DataFrame(_rows).sort_values("three-character share").reset_index(drop=True)
_S2["distance to baseline"] = (_S2["three-character share"] - _BASE["main_ipc_3"]).round(3)
_S2["not resolved by administrative classification"] = np.where(_S2["distance to baseline"] < 0, "yes", "")
_S2.loc[_S2["distance to baseline"].abs() <= MARGIN, "not resolved by administrative classification"] += " (borderline)"
_S2 = _S2[["topic", "records", "largest three-character class", "three-character share", "distance to baseline", "three-character classes",
           "largest four-character class", "four-character share", "four-character classes",
           "not resolved by administrative classification", "largest holder share", "top three holders"]]
_hit  = _S2["distance to baseline"] < 0
_edge = _S2["distance to baseline"].abs() <= MARGIN
print(f"\nTable S2  Topic screening (threshold = three-character random baseline {_BASE['main_ipc_3']:.4f})")
print(f"  hits: {_hit.sum()} / {len(_S2)} topics, "
      f"covering {_S2.loc[_hit,'records'].sum():,} records ({_S2.loc[_hit,'records'].sum()/_N*100:.1f}%)")
print(f"  of these, borderline topics within ±{MARGIN} of the baseline: {int(_edge.sum())}"
      f" ({', '.join(_S2.loc[_edge, 'topic'].str[:28])})")
print("\n  Hits (the administrative classification cannot resolve them; inspect them with the semantic layer):")
print(_S2[_hit].to_string(index=False))
print("\n  The five most concentrated topics among the non-hits (a scan by class code is sufficient):")
print(_S2[~_hit].nlargest(5, "three-character share").to_string(index=False))
print(f"\n  Full ranking ({len(_S2)} topics; only topic/records/three-character share/distance to baseline):")
print(_S2[["topic", "records", "three-character share", "distance to baseline"]].to_string(index=False))
_S2_EN = _S2.rename(columns={
    "topic": "topic", "records": "n_records", "largest three-character class": "top_ipc3", "three-character share": "top_ipc3_share",
    "distance to baseline": "distance_to_majority_share", "three-character classes": "n_ipc3_classes",
    "largest four-character class": "top_ipc4", "four-character share": "top_ipc4_share", "four-character classes": "n_ipc4_classes",
    "not resolved by administrative classification": "below_majority_share", "largest holder share": "top_holder_share", "top three holders": "top3_holders"})
_S2_EN["below_majority_share"] = _S2_EN["below_majority_share"].map(
    {"yes": "yes", "yes (borderline)": "yes (borderline)", " (borderline)": "no (borderline)", "": "no"})
_S2_EN.to_csv(os.path.join(SAVE_DIR, "usage_landscape_screen.csv"), index=False)

# ── Table S3  Top five companies × topic matrix ─────────────────────────────
_top_co = (_t["effective_assignee"].astype(str).str.strip()
           .replace("", np.nan).dropna().value_counts().head(TOP_N_CO).index.tolist())
_m = _t[_t["effective_assignee"].isin(_top_co)]
_S3 = (_m.pivot_table(index="effective_assignee", columns="merged_topic",
                      values="patent_id", aggfunc="count")
         .reindex(_top_co).fillna(0).astype(int))
_S3 = _S3.loc[:, _S3.sum(axis=0) > 0]
_S3pct = _S3.div(_S3.sum(axis=1), axis=0).mul(100).round(1)

print(f"\nTable S3  Topic portfolios of the top {TOP_N_CO} companies (share within the row in %; only the top three topics of each company)")
for _co in _top_co:
    _r = _S3pct.loc[_co].sort_values(ascending=False).head(3)
    _tot = int(_S3.loc[_co].sum())
    print(f"  {_co[:34]:36} n={_tot:>4}  " +
          " | ".join(f"{k[:30]} {v:.0f}%" for k, v in _r.items()))
_S3.to_csv(os.path.join(SAVE_DIR, "usage_company_topic_matrix.csv"))
_S3pct.to_csv(os.path.join(SAVE_DIR, "usage_company_topic_matrix_pct.csv"))

print("\nsaved -> usage_assignee_before_after.csv, usage_landscape_screen.csv,")
print("         usage_company_topic_matrix.csv, usage_company_topic_matrix_pct.csv")
