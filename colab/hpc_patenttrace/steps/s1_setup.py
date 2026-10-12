"""Section 1 of the pipeline notebook: environment, configuration, token, preflight, helpers, reuse.

Run from the pipeline notebook with run_step("s1_setup"); the parts below are executed in the namespace of the notebook.
"""

# ── 1.2  Imports and environment check ──────────────────────────────────────
import os, sys, gc, re, json, ast, platform, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# library messages (for example "There are adapters available but none are activated") stay out of the notebook
import logging
for _lib in ("adapters", "transformers", "sentence_transformers", "huggingface_hub", "accelerate"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
if "transformers" in sys.modules:
    sys.modules["transformers"].logging.set_verbosity_error()

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)

IN_COLAB = "google.colab" in sys.modules
HAS_CUDA = torch.cuda.is_available()
HAS_MPS  = (not HAS_CUDA) and torch.backends.mps.is_available()
DEVICE   = "cuda" if HAS_CUDA else ("mps" if HAS_MPS else "cpu")

print(f"Run environment : {'Google Colab' if IN_COLAB else 'local computer'} ({platform.system()} {platform.machine()})")
print(f"Python          : {sys.version.split()[0]}")
print(f"PyTorch         : {torch.__version__}")
print(f"pandas          : {pd.__version__}")
print(f"Device          : {DEVICE}")
if HAS_CUDA:
    print(f"GPU             : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    print("No CUDA GPU. Sections 4 and 5 can only reuse existing outputs or use the small stand-in model (sample profile).")


# ── 1.3  Configuration from the parameters of the notebook ───────────────────
# Every path comes from the Parameters cell. A relative path is taken inside DATA_DIR; the folders input/ and
# OUTPUT_DIR are created if they do not exist, and a missing input file stops the run with a list of what to add.
from hpc_patenttrace.notebook import resolve

DATA_DIR   = os.path.expanduser(str(DATA_DIR).strip())
DATA_PATH  = resolve(DATA_DIR, INPUT_CSV)
LITIG_FILE = resolve(DATA_DIR, LITIGATION_FILE)
SAVE_DIR   = os.path.join(resolve(DATA_DIR, OUTPUT_DIR), PROFILE)
REUSE_DIR  = resolve(DATA_DIR, REUSE_FROM) if str(REUSE_FROM).strip() else None   # read in 1.7
if PROFILE not in ("full", "sample"):
    raise ValueError(f'PROFILE must be "full" or "sample", not {PROFILE!r}')
if PROFILE == "sample" and REUSE_DIR:
    print("The sample profile generates every layer; REUSE_FROM is not used.")
    REUSE_DIR = None
if REUSE_DIR and os.path.normpath(REUSE_DIR) == os.path.normpath(SAVE_DIR):
    raise ValueError("REUSE_FROM is the output folder of this run; give the folder of an earlier run")

os.makedirs(os.path.join(DATA_DIR, "input"), exist_ok=True)
_need = [(DATA_PATH, "HPC_15to25_v2.csv, the output of step 5 of this repository"),
         (LITIG_FILE, "patents.dta (or patents.dta.zip) of the USPTO Patent Litigation Docket Reports")]
if REUSE_DIR:
    _need.append((REUSE_DIR, "the output folder of an earlier full run of this notebook (REUSE_FROM)"))
_absent = [(p, what) for p, what in _need if not os.path.exists(p)]
if _absent:
    raise FileNotFoundError("Add these files, then run the notebook again:\n"
                            + "\n".join(f"  {p}\n      {what}" for p, what in _absent))

CLEAN_PATH = os.path.join(SAVE_DIR, "HPC_15to25_clean.csv")   # written by section 2, read by sections 3-9
FIG_DIR    = os.path.join(SAVE_DIR, "figures")
TBL_DIR    = os.path.join(SAVE_DIR, "tables")
PROV_DIR   = os.path.join(SAVE_DIR, "provenance")
for _d in (SAVE_DIR, FIG_DIR, TBL_DIR, PROV_DIR):
    os.makedirs(_d, exist_ok=True)

RANDOM_STATE = 42
MIN_LEN      = 100       # a section shorter than this number of characters is treated as missing

# ── Scale-dependent settings (the "full" values are the settings of the released dataset) ──
if PROFILE == "full":
    SAMPLE_N = None
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES = 30, 10
    N_TARGET, MIN_CLASS_N = 25, 30          # merged topics; minimum records per merged topic
    MIN_LABEL_N, MIN_ASSIGNEE_N, N_BOOT = 20, 20, 1000
else:
    SAMPLE_N = 400                          # input records drawn at random from DATA_PATH
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES = 8, 3
    N_TARGET, MIN_CLASS_N = 6, 5
    MIN_LABEL_N, MIN_ASSIGNEE_N, N_BOOT = 3, 3, 200

# ── Generation models ──
# The sample profile can replace both generation models by one small instruction model. That only tests that
# every step runs; its text is not comparable with the released layers. "auto" uses the real models on a GPU
# with at least 32 GB (Qwen2.5-14B in bf16 needs about 30 GB) and the small model otherwise.
_GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_CUDA else 0.0
SMALL_MODELS = (PROFILE == "sample") and (SAMPLE_MODELS == "small" or (SAMPLE_MODELS == "auto" and _GPU_GB < 32))
if SMALL_MODELS:
    S1_MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
    S2_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    S1_4BIT, GEN_DTYPE = False, torch.float32
    S1_GPU_BATCH, S2_GPU_BATCH, BASE_BATCH, MATCHED_BATCH = 8, 8, 8, 4
else:
    S1_MODEL_ID   = "meta-llama/Llama-3.1-8B-Instruct"   # facets, 4-bit NF4
    S2_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"          # synthesis, direct layers and topic labels, bf16
    S1_4BIT, GEN_DTYPE = True, torch.bfloat16
    # Batch sizes of the released run (A100 80 GB), halved on a smaller GPU: Qwen2.5-14B takes 29.5 GB in bf16,
    # and its cache needs about 0.2 MB per token, so 8 direct-full prompts of 5,000 tokens add another 8 GB.
    # With reused layers nothing is generated, and the release files keep the batch sizes that produced them.
    if _GPU_GB >= 70 or REUSE_DIR:
        S1_GPU_BATCH, S2_GPU_BATCH, BASE_BATCH, MATCHED_BATCH = 32, 24, 16, 8
    else:
        S1_GPU_BATCH, S2_GPU_BATCH, BASE_BATCH, MATCHED_BATCH = 16, 12, 8, 4

np.random.seed(RANDOM_STATE)
_ = torch.manual_seed(RANDOM_STATE)

RUN_SETTINGS = pd.DataFrame([
    ("profile", PROFILE + (f" ({SAMPLE_N} input records drawn at random)" if SAMPLE_N else "")),
    ("device", DEVICE + (f" ({torch.cuda.get_device_name(0)}, {_GPU_GB:.0f} GB)" if HAS_CUDA else "")),
    ("input file", DATA_PATH),
    ("litigation file", LITIG_FILE),
    ("generated layers", f"reused from {REUSE_DIR}" if REUSE_DIR else "generated in this run"),
    ("generation models", f"{S1_MODEL_ID} | {S2_MODEL_NAME}" + ("  (small stand-in model)" if SMALL_MODELS else "")),
    ("output folder", SAVE_DIR),
], columns=["setting", "value"])
print(RUN_SETTINGS.to_string(index=False))


# ── 1.4  Hugging Face token ─────────────────────────────────────────────────
# Only meta-llama/Llama-3.1-8B-Instruct (section 4) is a gated model; the other models download without a token.
# The token is looked up in this order: HF_TOKEN of 1.3, the Colab secret HF_TOKEN, the environment variable
# HF_TOKEN, an earlier `huggingface-cli login`. A prompt appears only when section 4 must download the gated model.
from huggingface_hub import login, get_token
from getpass import getpass

def _find_hf_token():
    if HF_TOKEN:
        return HF_TOKEN, "HF_TOKEN in 1.3"
    if IN_COLAB:
        try:
            from google.colab import userdata
            _t = userdata.get("HF_TOKEN")
            if _t:
                return _t, "Colab secret"
        except Exception:
            pass
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "environment variable"
    _t = get_token()
    if _t:
        return _t, "cached login"
    return None, None

def ensure_hf_login():
    """Log in before a gated model is downloaded; ask for the token only if no source has one."""
    token, source = _find_hf_token()
    if token is None:
        token, source = getpass("Hugging Face token: "), "prompt"
    os.environ["HF_TOKEN"] = token
    login(token=token, add_to_git_credential=False)
    print(f"Hugging Face login: token from {source}")

_tok, _src = _find_hf_token()
if _tok:
    ensure_hf_login()
else:
    print("No Hugging Face token found. That is fine unless section 4 has to generate facets with the gated model;"
          " in that case section 4.1 asks for the token.")


# ── 1.5  Preflight: report missing packages and input columns before the long stages start ──
# Note: importlib.util.find_spec() fails for a module that is already imported and has __spec__ None
# (lazy modules such as adapters), so a loaded module is taken as available directly.
import importlib.util

def _module_available(name):
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False

REQUIRED_MODULES = {          # module -> first section that uses it
    "transformers": "4.1", "accelerate": "4.1", "sentencepiece": "4.1",
    "adapters": "6.1", "umap": "6.4", "hdbscan": "6.4", "minisom": "7.10",
    "sentence_transformers": "5.9", "bert_score": "8.14", "rouge_score": "8.4",
    "sklearn": "7.11", "scipy": "7.9", "joblib": "7.11", "pyarrow": "6.13",
    "matplotlib": "2.2", "seaborn": "7.9", "plotly": "7.6",
}
if S1_4BIT and HAS_CUDA:                  # 4-bit loading needs bitsandbytes and a CUDA GPU
    REQUIRED_MODULES["bitsandbytes"] = "4.1"
REQUIRED_COLS = [             # columns of the crawler output (step 5) that this notebook reads
    "patent_id", "title", "abstract", "BACKGROUND", "SUMMARY", "DETAILED DESCRIPTION",
    "claims_text", "claim_count", "main_ipc_4", "main_ipc_description_4", "ipc_description",
    "current_assignee", "assignee", "reassignment_assignee", "application_year", "application_date",
    "applications_claiming_priority", "legal_events", "citation_count", "citations",
]

_missing = [f"{m} (section {sec})" for m, sec in REQUIRED_MODULES.items() if not _module_available(m)]
_have = list(pd.read_csv(DATA_PATH, nrows=0).columns) if os.path.exists(DATA_PATH) else []
_lack = [c for c in REQUIRED_COLS if c not in _have]
_dta  = [LITIG_FILE] if os.path.exists(LITIG_FILE) else []

print(f"Packages     : {len(REQUIRED_MODULES) - len(_missing)}/{len(REQUIRED_MODULES)} available")
for _m in _missing:
    print(f"  missing {_m}")
print(f"Input file   : {'found' if _have else 'NOT FOUND'} ({DATA_PATH})")
print(f"Input columns: {len(REQUIRED_COLS) - len(_lack)}/{len(REQUIRED_COLS)} present")
for _c in _lack:
    print(f"  missing column {_c}")
print(f"Docket file  : {_dta[0] if _dta else 'NOT FOUND: ' + LITIG_FILE}")
if _missing or _lack or not _have or not _dta:
    raise RuntimeError("Preflight failed: add the items listed above, then run this cell again.")
print("\nPreflight passed.")


# ── 1.6  Helper functions: memory release, run-time log, family split, class level, figure style ──
import time
from contextlib import contextmanager
from IPython import get_ipython

def free_gpu(*var_names):
    """Delete the named global variables (if they exist) and empty the GPU cache.
    Called between stages so that only one large model is in GPU memory at a time."""
    g = globals()
    for name in var_names:
        if name in g:
            del g[name]
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: allocated {torch.cuda.memory_allocated()/1e9:.2f} GB | "
              f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB | "
              f"free {total - torch.cuda.memory_reserved()/1e9:.2f} GB")
    elif HAS_MPS:
        torch.mps.empty_cache()

# ── Run-time log ─────────────────────────────────────────────────────────────
# Every expensive step is wrapped in `with timed("<section>  <name>"):`. The record is appended to
# SAVE_DIR/runtime_log.csv as soon as the step ends, so an interrupted session loses nothing; a rerun of
# a step replaces its own record. Section 9.2 turns the log into the cost table of the release.
RUNTIME_PATH = os.path.join(SAVE_DIR, "runtime_log.csv")
RUNTIME_LOG  = (pd.read_csv(RUNTIME_PATH).to_dict("records")
                if os.path.exists(RUNTIME_PATH) else [])

def _hardware_tag():
    if HAS_CUDA:
        p = torch.cuda.get_device_properties(0)
        return f"GPU {p.name} {p.total_memory/1e9:.0f}GB"
    return f"{'MPS' if HAS_MPS else 'CPU'} {platform.machine()} x{os.cpu_count()}"

def _hms(sec):
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

HARDWARE = _hardware_tag()

class _Step:
    """Record made by timed(). Set n_items/note inside the with block; read .seconds after it."""
    def __init__(self, name, n_items=None):
        self.name, self.n_items = name, n_items
        self.note, self.seconds = "", None

@contextmanager
def timed(step, n_items=None, quiet=False):
    s  = _Step(step, n_items)
    t0 = time.perf_counter()
    try:
        yield s
    finally:
        s.seconds = time.perf_counter() - t0
        rec = {"step": s.name,
               "seconds": round(s.seconds, 2),
               "hms": _hms(s.seconds),
               "n_items": s.n_items,
               "ms_per_item": (round(s.seconds / s.n_items * 1000, 2) if s.n_items else None),
               "hardware": HARDWARE,
               "note": s.note}
        _prev = next((r for r in RUNTIME_LOG if r["step"] == s.name), None)
        if _prev is not None and float(_prev.get("seconds") or 0) > 60 and s.seconds < 5:
            # the step resumed from a complete checkpoint: keep the measured time and add a note
            _old_note = "" if pd.isna(_prev.get("note")) else str(_prev.get("note"))
            _prev["note"] = (_old_note + " | later run resumed from checkpoint").strip(" |")
            rec = _prev
        RUNTIME_LOG[:] = [r for r in RUNTIME_LOG if r["step"] != s.name] + [rec]
        pd.DataFrame(RUNTIME_LOG).to_csv(RUNTIME_PATH, index=False)
        if not quiet:
            extra = (f" | n={s.n_items:,} | {rec['ms_per_item']:,.1f} ms/item" if s.n_items else "")
            print(f"[time] {s.name}: {rec['hms']} ({s.seconds:,.1f} s){extra}")

# ── Wall time of every part (for the run report of section 10) ───────────────
# run_step() of the notebook appends the time of each part of a step to cell_times.csv.
CELL_LOG_PATH = os.path.join(SAVE_DIR, "cell_times.csv")
# ── Family-grouped split (prevents train/test leakage) ───────────────────────
# Continuations in one patent family have almost the same specification (for example, US10592275B2 and
# US12159161B2 belong to family US13/750696 and have a word-for-word identical background). A random split
# puts members of one family in both training and test sets and raises the scores. All supervised examples
# (7.11, 8.9, 8.10) therefore split by family. The family key is the root application number in the
# `family` column ("US13/750696 (n=2)" gives "US13/750696"); a record without a family is its own group.
import re as _re_fam

def build_family_groups(patent_ids, data_path=None):
    """Return the family keys aligned with patent_ids (reads the family column of CLEAN_PATH)."""
    fam_df = pd.read_csv(data_path or CLEAN_PATH,
                         usecols=["patent_id", "family"], dtype=str).fillna("")
    root = {}
    for pid, fam in zip(fam_df["patent_id"], fam_df["family"]):
        m = _re_fam.match(r"\s*(\S+)", fam)
        root[pid] = m.group(1) if m else ""
    return np.array([root.get(p) or f"self:{p}" for p in patent_ids])

def family_train_test_split(idx, y, groups, test_size=0.2):
    """Family-grouped train/test split (one GroupShuffleSplit split).
    A grouped split cannot also stratify; macro-F1 with zero_division=0 handles classes that are missing in one part."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(idx, y, groups=groups))
    return idx[tr], idx[te], y[tr], y[te]

def add_ipc3(frame, src_col="main_ipc_4", col="main_ipc_3"):
    """Add the three-character class (G06F -> G06) as a new column and return the same frame.
    The classification hierarchy is section G / class G06 / subclass G06F. The merged topics meet about
    40 three-character classes, a granularity closer to theirs than the hundreds of subclasses, so the
    three-character class is the scale of the external check; the subclass is kept for the finer examples."""
    if src_col in frame.columns:
        frame[col] = frame[src_col].astype(str).str.strip().str.upper().str[:3]
    return frame

# ── Figure style ─────────────────────────────────────────────────────────────
# One categorical order (checked for colour-vision deficiency), one blue ramp for magnitudes,
# grey for "other". Every figure is also saved to FIG_DIR as a 300 dpi PNG.
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
PALETTE   = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, BASELINE, OTHER_GREY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d3d2cc"
SEQ_CMAP  = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "axes.prop_cycle": cycler(color=PALETTE),
})

def save_fig(fig, name):
    """Save a figure as FIG_DIR/<name>.png (300 dpi) and return the path."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path)
    print(f"saved -> figures/{name}.png")
    return path

print(f"Helpers ready | hardware: {HARDWARE} | run-time log: {os.path.basename(RUNTIME_PATH)}")


# ── 1.7  Reuse the generated layers of an earlier run (only when REUSE_DIR is set) ──
# The language-model outputs are the expensive part of this notebook (about 20 h on one A100 80 GB).
# When REUSE_DIR is the output folder of an earlier run, this cell copies into SAVE_DIR the checkpoints of
# the three generated layers and the outputs of steps whose result depends on the hardware or on a random
# initialisation: the SPECTER2 embeddings, the UMAP coordinates, the cluster assignment with its topic labels
# and the human-audit sample. The generation cells then find complete checkpoints and do not load a model;
# every other step is computed again. Each reused file is checked against the corpus of this run where it is read.
import shutil, hashlib, glob

REUSE_FILES = [   # (file in REUSE_DIR, file in SAVE_DIR, used in)
    ("stage1_checkpoint.csv",        "stage1_checkpoint.csv",          "4.3 facets"),
    ("stage2_checkpoint.csv",        "stage2_checkpoint.csv",          "5.6 synthesized summaries"),
    ("baseline_direct.csv",          "baseline_direct.csv",            "5.10 direct-brief summaries"),
    ("baseline_direct_matched.csv",  "baseline_direct_matched.csv",    "5.11 direct-full summaries"),
    ("specter2_embeddings.csv",      "specter2_embeddings.csv",        "6.2 synthesized embeddings"),
    ("cluster_results.csv",          "reused_cluster_results.csv",     "6.4 UMAP coordinates and clusters, 6.8 topic labels"),
    ("hdbscan_param_sweep.csv",      "reused_hdbscan_param_sweep.csv", "6.5 sweep that goes with the reused clusters"),
    ("umap2d_coords.csv",            "umap2d_coords.csv",              "7.7 two-dimensional map"),
    ("audit_primary_50.csv",         "audit_primary_50.csv",           "8.8 human-audit sample"),
    ("audit_edge_12.csv",            "audit_edge_12.csv",              "8.8 human-audit sample"),
    ("audit_sentences.csv",          "audit_sentences.csv",            "8.8 human-audit sample"),
    ("released_corpus_ids.csv",      "reused_released_corpus_ids.csv", "3.5 check of the corpus"),
    ("runtime_log.csv",              "reused_runtime_log.csv",         "9.2 cost of the reused steps"),
]
if REUSE_DIR:
    REUSE_FILES += [(os.path.join("embeddings", os.path.basename(p)),
                     os.path.join("reused_embeddings", os.path.basename(p)), "6.12 representations")
                    for p in sorted(glob.glob(os.path.join(REUSE_DIR, "embeddings", "*.parquet")))]

def _sha12(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]

reuse_report = pd.DataFrame(columns=["file", "status", "MB", "sha256_12", "used_in"])
if REUSE_DIR is None:
    print("REUSE_DIR is None: every layer is generated in this run.")
elif not os.path.isdir(REUSE_DIR):
    raise FileNotFoundError(f"REUSE_DIR does not exist: {REUSE_DIR} (set it to None to generate everything)")
else:
    os.makedirs(os.path.join(SAVE_DIR, "reused_embeddings"), exist_ok=True)
    _rows = []
    for _src_rel, _dst_rel, _use in REUSE_FILES:
        _src, _dst = os.path.join(REUSE_DIR, _src_rel), os.path.join(SAVE_DIR, _dst_rel)
        if not os.path.exists(_src):
            _rows.append({"file": _src_rel, "status": "not in REUSE_DIR (computed in this run)",
                          "MB": None, "sha256_12": "", "used_in": _use})
            continue
        _status = "already in SAVE_DIR" if os.path.exists(_dst) else "copied"
        if _status == "copied":
            _ = shutil.copy2(_src, _dst)
        _rows.append({"file": _src_rel, "status": _status, "MB": round(os.path.getsize(_dst) / 1e6, 2),
                      "sha256_12": _sha12(_dst), "used_in": _use})
    reuse_report = pd.DataFrame(_rows)
    reuse_report.to_csv(os.path.join(SAVE_DIR, "reuse_manifest.csv"), index=False)
    print(f"Reused from {REUSE_DIR}:")
reuse_report
