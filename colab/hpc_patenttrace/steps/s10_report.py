"""Section 10: run report.

Run from the pipeline notebook with run_step("s10_report"); the parts below are executed in the namespace of the notebook.
"""

# ── 10.1  Run report: what ran, what was reused, how long each section took ──
import importlib.metadata, sklearn
from IPython.display import display
def _ver(dist):   # version of an installed distribution (hdbscan has no __version__ attribute)
    try:
        return importlib.metadata.version(dist)
    except Exception:
        return "-"

env_df = pd.DataFrame([
    ("profile", PROFILE + (f" (SAMPLE_N = {SAMPLE_N})" if SAMPLE_N else "")),
    ("generated layers", "reused from " + REUSE_DIR if REUSE_DIR else "generated in this run"),
    ("models", f"{S1_MODEL_ID} | {S2_MODEL_NAME}" + (" (small stand-in model)" if SMALL_MODELS else "")),
    ("hardware", HARDWARE),
    ("python / torch / transformers", f"{sys.version.split()[0]} / {torch.__version__} / {_ver('transformers')}"),
    ("pandas / scikit-learn / umap / hdbscan", f"{pd.__version__} / {sklearn.__version__} / {_ver('umap-learn')} / {_ver('hdbscan')}"),
    ("input file", f"{DATA_PATH} ({N_FILE:,} records)"),
    ("output folder", SAVE_DIR),
], columns=["item", "value"])
env_df

# Wall time by section, from the per-cell log that 1.6 starts (cells 1.1-1.5 are not in it; the last run of each cell counts)
cells_df = pd.read_csv(CELL_LOG_PATH)
cells_df["section"] = cells_df["cell"].astype(str).str.extract(r"^(\d+)\.", expand=False)
cells_df = cells_df.drop_duplicates("cell", keep="last")
by_section = (cells_df.dropna(subset=["section"])
              .assign(section=lambda d: d["section"].astype(int))
              .groupby("section").agg(cells=("cell", "size"), seconds=("seconds", "sum"))
              .assign(hh_mm_ss=lambda d: d["seconds"].map(_hms)))
print(f"Total wall time of the executed cells: {_hms(cells_df['seconds'].sum())}")
by_section
_failed = cells_df[~cells_df["ok"].astype(bool)]
print(f"Cells that ended with an error: {len(_failed)}")
if len(_failed):
    display(_failed)

# Key numbers of this run; for the full profile next to the released dataset (2026-10-11)
_clu = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
_topic = _clu["merged_topic"][~_clu["merged_topic"].isin(["__noise__", "Off-topic / Non-HPC", "Unclassified", ""])]
_ev = pd.read_csv(os.path.join(SAVE_DIR, "evidence_flags.csv"))
_nmi = pd.read_csv(os.path.join(TBL_DIR, "e6_topic_ipc_metrics.csv")).iloc[0]["value"]
_audit = [os.path.join(SAVE_DIR, f) for f in ("audit_primary_50.csv", "audit_edge_12.csv", "audit_sentences.csv")]
key_df = pd.DataFrame([
    ("records in the input file",              N_FILE,                                21947),
    ("records used in this run",               N_RAW,                                 21947),
    ("removed: HPC has another sense",         N_RAW - N_CLEAN,                       5513),
    ("candidates",                             N_CLEAN,                               16434),
    ("five complete fields",                   N_COMPLETE,                            7158),
    ("released corpus",                        N_RELEASED,                            6535),
    ("excluded by the inclusion rule",         N_EXCLUDED,                            623),
    ("records that define HPC (sense_basis)",  int((_ev["sense_basis"] == "definition").sum()), 2831),
    ("litigated-family subset",                len(pd.read_csv(os.path.join(SAVE_DIR, "litigation_family.csv"))), 25),
    ("clusters (HDBSCAN)",                     int(_clu.loc[_clu["cluster_id"] != -1, "cluster_id"].nunique()), 51),
    ("noise records",                          int((_clu["cluster_id"] == -1).sum()), 1502),
    ("merged topics",                          int(_topic.nunique()),                 25),
    ("records with a topic",                   int(len(_topic)),                      5033),
    ("NMI, topics vs CPC subclasses",          round(float(_nmi), 4),                 0.4558),
    ("pairing flags (8.4)",                    len(pd.read_csv(os.path.join(SAVE_DIR, "pairing_integrity_flags.csv"))), 143),
    ("human-audit sample: records",            sum(len(pd.read_csv(p)) for p in _audit[:2]), 62),
    ("human-audit sample: sentences",          len(pd.read_csv(_audit[2])),           349),
], columns=["quantity", "this_run", "released_2026_10_11"])
if PROFILE == "full":
    key_df["same"] = np.where(np.isclose(key_df["this_run"].astype(float), key_df["released_2026_10_11"].astype(float)), "yes", "NO")
else:
    key_df = key_df.drop(columns="released_2026_10_11")
_fmt = lambda v: f"{v:,.4f}" if not float(v).is_integer() else f"{int(v):,}"
for _c in ("this_run", "released_2026_10_11"):
    if _c in key_df.columns:
        key_df[_c] = key_df[_c].astype(object).map(_fmt)
key_df.to_csv(os.path.join(SAVE_DIR, "run_report_key_numbers.csv"), index=False)
key_df

if EMBEDDING_CHECK:
    print(f"SPECTER2 check of the reused embeddings ({EMBEDDING_CHECK['records']} records encoded again on {HARDWARE}): "
          f"max |difference| {EMBEDDING_CHECK['max_abs_diff']:.2e}, minimum cosine {EMBEDDING_CHECK['min_cosine']:.6f}")
print("Figures:", ", ".join(sorted(os.listdir(FIG_DIR))))
