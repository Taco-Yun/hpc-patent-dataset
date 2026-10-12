"""Section 6: SPECTER2 embeddings, clusters and merged topics.

Run from the pipeline notebook with run_step("s6_topics"); the parts below are executed in the namespace of the notebook.
"""

# ── 6.1  SPECTER2 encoder with the proximity adapter ────────────────────────
from adapters import AutoAdapterModel
# from_pretrained prints "There are adapters available but none are activated" before the adapter is loaded;
# the adapter is activated by load_adapter(set_active=True) and set_active_adapters below.
from transformers import AutoTokenizer

SPECTER2_MODEL = "allenai/specter2_base"

specter_tokenizer = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
specter_model     = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
_ = specter_model.load_adapter("allenai/specter2", source="hf",
                               load_as="proximity", set_active=True)
specter_model.set_active_adapters("proximity")
specter_model = specter_model.eval().to(DEVICE)

dummy = specter_tokenizer("test", return_tensors="pt").to(specter_model.device)
with torch.no_grad():
    out = specter_model(**dummy)
print(f"SPECTER2 ready | dim = {out.last_hidden_state[:, 0, :].shape[-1]} | max 512 tokens")


# ── 6.2  Embedding of title + synthesized summary ────────────────────────────
# With a reused run (1.7) the embeddings of that run are read, so that the clusters, topics and figures stay
# the same; the first 256 records are encoded again to show how close this hardware comes to the reused values.
def embed_texts(texts, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        inputs = specter_tokenizer(batch, return_tensors="pt", truncation=True,
                                   padding=True, max_length=512).to(specter_model.device)
        with torch.no_grad():
            out = specter_model(**inputs)
        all_embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)

texts = (pipeline_df["title"] + " " + pipeline_df["final_summary"]).tolist()
embed_path = os.path.join(SAVE_DIR, "specter2_embeddings.csv")
EMBEDDING_CHECK = None
if os.path.exists(embed_path):
    embed_df = pd.read_csv(embed_path)
    if list(embed_df["patent_id"].astype(str)) != list(pipeline_df["patent_id"].astype(str)):
        raise RuntimeError(f"{os.path.basename(embed_path)} does not follow the record order of the corpus; "
                           "delete it to encode again.")
    embeddings = embed_df.drop(columns=["patent_id"]).values.astype(np.float32)
    _n = min(256, len(texts))
    with timed("6.2  SPECTER2 check of the reused embeddings", n_items=_n):
        _new = embed_texts(texts[:_n], batch_size=64)
    _a, _b = _new, embeddings[:_n]
    _cos = (_a * _b).sum(1) / (np.linalg.norm(_a, axis=1) * np.linalg.norm(_b, axis=1))
    EMBEDDING_CHECK = {"records": _n, "max_abs_diff": float(np.abs(_a - _b).max()),
                       "min_cosine": float(_cos.min()), "mean_cosine": float(_cos.mean())}
    print(f"read {embed_path} ({embeddings.shape[0]:,} x {embeddings.shape[1]})")
    print(f"encoded again on {HARDWARE}: {_n} records | max |difference| {EMBEDDING_CHECK['max_abs_diff']:.2e} | "
          f"cosine min {EMBEDDING_CHECK['min_cosine']:.6f}, mean {EMBEDDING_CHECK['mean_cosine']:.6f}")
else:
    with timed("6.2  SPECTER2 embedding (title + final_summary)", n_items=len(texts)):
        embeddings = embed_texts(texts, batch_size=64)
    embed_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
    embed_df.insert(0, "patent_id", pipeline_df["patent_id"].values)
    embed_df.to_csv(embed_path, index=False)
    print(f"embeddings: {embeddings.shape} | saved -> {embed_path} ({os.path.getsize(embed_path)/1e6:.1f} MB)")


# ── 6.3  Reload the embeddings and check them against the corpus ─────────────
embed_df   = pd.read_csv(os.path.join(SAVE_DIR, "specter2_embeddings.csv"))
patent_ids = embed_df["patent_id"].values
embeddings = embed_df.drop(columns=["patent_id"]).values.astype(np.float32)
print(f"embeddings: {embeddings.shape}")

# The identifiers must be those of the current corpus (see 4.4)
if "m1" in globals():
    _e, _m = set(embed_df["patent_id"].astype(str)), set(m1["patent_id"].astype(str))
    if _e != _m:
        raise RuntimeError(
            f"specter2_embeddings.csv does not match the current corpus ({len(_e)} vs {len(_m)} records). "
            "SAVE_DIR belongs to another corpus version. Use a new SAVE_DIR.")
    print("patent_id alignment check passed")


# ── 6.4  UMAP (768 -> 3 dimensions), then HDBSCAN ───────────────────────────
# UMAP depends on the hardware and the library version even with a fixed seed. With a reused run, its
# three-dimensional coordinates are read and the clusters are computed again on them. HDBSCAN can still place
# a few boundary records differently under another library version or processor, so the result is compared
# with the reused assignment, and the reused assignment is kept when the two differ.
import umap
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

_reused_clu = os.path.join(SAVE_DIR, "reused_cluster_results.csv")
REUSED_CLUSTERS = None
if os.path.exists(_reused_clu):
    REUSED_CLUSTERS = (pd.read_csv(_reused_clu, usecols=["patent_id", "cluster_id", "umap_3d_x", "umap_3d_y", "umap_3d_z"])
                       .astype({"patent_id": str}).set_index("patent_id")
                       .reindex(pd.Series(patent_ids).astype(str)))
    if REUSED_CLUSTERS["cluster_id"].isna().any():
        raise RuntimeError("The reused cluster file does not cover every record of this corpus.")
    # UMAP returns float32; the CSV holds their shortest decimal form, which converts back to the same float32 values
    embed_3d = REUSED_CLUSTERS[["umap_3d_x", "umap_3d_y", "umap_3d_z"]].values.astype(np.float32)
    umap_3d  = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.0, metric="cosine",
                         random_state=RANDOM_STATE, low_memory=False)   # settings only, for 9.1 (not fitted)
    print(f"UMAP coordinates read from the reused run: {embed_3d.shape}")
else:
    print("UMAP 768d -> 3d ...")
    with timed("6.4  UMAP 768d -> 3d", n_items=len(embeddings)):
        umap_3d = umap.UMAP(
            n_components=3, n_neighbors=15, min_dist=0.0,
            metric="cosine", random_state=RANDOM_STATE, low_memory=False,
        )
        embed_3d = umap_3d.fit_transform(embeddings)
    print(f"embed_3d: {embed_3d.shape}")

print("\nHDBSCAN ...")
with timed("6.4  HDBSCAN clustering", n_items=len(embed_3d)):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True, gen_min_span_tree=True,
    )
    cluster_labels = clusterer.fit_predict(embed_3d)
REUSED_KEPT = False
dbcv_refit = clusterer.relative_validity_
if REUSED_CLUSTERS is not None:
    _reused_ids = REUSED_CLUSTERS["cluster_id"].astype(int).values
    _same = int((_reused_ids == cluster_labels).sum())
    # Cluster numbers follow the order in which HDBSCAN finds the clusters, so one changed boundary renumbers
    # the clusters after it. The adjusted Rand index and the matched count do not depend on the numbers:
    # a record counts as matched when it lies in the new cluster that takes most of its reused cluster
    # (noise counted as one group).
    _ari = adjusted_rand_score(_reused_ids, cluster_labels)
    _matched = int(pd.crosstab(_reused_ids, cluster_labels).max(axis=1).sum())
    print(f"clusters computed again on the reused coordinates: {_same:,} of {len(cluster_labels):,} records keep"
          f" the same cluster_id; {_matched:,} lie in the matching cluster (adjusted Rand index {_ari:.4f})")
    if _same != len(cluster_labels):
        print("  -> the reused assignment is kept, so that topics and audit stay those of the released dataset")
        cluster_labels = _reused_ids
        REUSED_KEPT = True

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_pct  = (cluster_labels == -1).sum() / len(cluster_labels) * 100
mask    = cluster_labels != -1
sil  = silhouette_score(embed_3d[mask], cluster_labels[mask], sample_size=min(500, int(mask.sum())),
                        random_state=RANDOM_STATE)
ch   = calinski_harabasz_score(embed_3d[mask], cluster_labels[mask])
dbcv = dbcv_refit
_dbcv_source = ""
_reused_sweep_path = os.path.join(SAVE_DIR, "reused_hdbscan_param_sweep.csv")
if REUSED_KEPT and os.path.exists(_reused_sweep_path):
    # DBCV needs the cluster tree, which exists only for the clusters computed here; the value of the kept
    # assignment is the one its own run recorded in the sweep.
    _row = pd.read_csv(_reused_sweep_path).query(
        "min_cluster_size == @HDBSCAN_MIN_CLUSTER_SIZE and min_samples == @HDBSCAN_MIN_SAMPLES")
    if len(_row):
        dbcv = float(_row["dbcv"].iloc[0])
        _dbcv_source = f"  (reused run; {dbcv_refit:.4f} for the clusters computed again)"

print(f"\n{'='*56}")
print("clustering metrics")
print(f"{'='*56}")
print(f"  Silhouette        : {sil:.4f}")
print(f"  DBCV              : {dbcv:.4f}{_dbcv_source}")
print(f"  Calinski-Harabasz : {ch:.2f}")
print(f"  noise share       : {noise_pct:.1f}%")
print(f"  clusters          : {n_clusters}")
print("reference values: Silhouette > 0.3 | DBCV > 0.3 | noise < 20%")
if PROFILE == "full":
    print("selection rule: of the six settings of the sweep in 6.5, the only one with DBCV > 0.3 is (30, 10); "
          "Silhouette and noise share are secondary criteria (full sweep in hdbscan_param_sweep.csv)")
else:
    print(f"sample profile: smaller settings ({HDBSCAN_MIN_CLUSTER_SIZE}, {HDBSCAN_MIN_SAMPLES}) for the small number of records")


# ── 6.5  HDBSCAN parameter sweep (six settings) ─────────────────────────────
param_grid = [
    {"min_cluster_size": 10, "min_samples": 3},
    {"min_cluster_size": 15, "min_samples": 5},
    {"min_cluster_size": 20, "min_samples": 5},
    {"min_cluster_size": 20, "min_samples": 10},
    {"min_cluster_size": 25, "min_samples": 5},
    {"min_cluster_size": 30, "min_samples": 10},
]

print(f"  {'min_cluster':>12} {'min_samples':>12} {'n_cluster':>10} "
      f"{'noise%':>8} {'Silhouette':>12} {'DBCV':>8}")
sweep_rows = []
_sweep_timer = timed(f"6.5  HDBSCAN parameter sweep ({len(param_grid)} settings)",
                     n_items=len(param_grid))
_ = _sweep_timer.__enter__()
for params in tqdm(param_grid, desc="param sweep"):
    clf = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean", cluster_selection_method="eom",
        gen_min_span_tree=True, prediction_data=True)
    labels = clf.fit_predict(embed_3d)
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    npct    = (labels == -1).sum() / len(labels) * 100
    msk     = labels != -1
    if msk.sum() < 10:
        continue
    if len(set(labels[msk])) < 2:
        continue
    s = silhouette_score(embed_3d[msk], labels[msk], sample_size=min(500, int(msk.sum())),
                         random_state=RANDOM_STATE)
    d = clf.relative_validity_
    sweep_rows.append({**params, "n_cluster": n_clust, "noise_pct": round(npct,1),
                       "silhouette": round(s,4), "dbcv": round(d,4)})
    print(f"  {params['min_cluster_size']:>12} {params['min_samples']:>12} "
          f"{n_clust:>10} {npct:>7.1f}% {s:>12.4f} {d:>8.4f}")

_ = _sweep_timer.__exit__(None, None, None)

sweep_df   = pd.DataFrame(sweep_rows)
sweep_path = os.path.join(SAVE_DIR, "hdbscan_param_sweep.csv")
if REUSED_KEPT and os.path.exists(_reused_sweep_path):
    # The kept clusters come from the reused run, so its sweep is the one that describes them; the sweep of
    # this run is written next to it.
    _recomputed_path = os.path.join(SAVE_DIR, "hdbscan_param_sweep_recomputed.csv")
    sweep_df.to_csv(_recomputed_path, index=False)
    _ = shutil.copy2(_reused_sweep_path, sweep_path)
    _sweep_cmp = sweep_df.merge(pd.read_csv(_reused_sweep_path), on=["min_cluster_size", "min_samples"],
                                suffixes=("_this_run", "_reused"))
    print("\nthis run against the reused run:")
    print(_sweep_cmp[["min_cluster_size", "min_samples", "n_cluster_this_run", "n_cluster_reused",
                      "dbcv_this_run", "dbcv_reused"]].to_string(index=False))
    for _name, _tab in (("this run", sweep_df), ("reused run", pd.read_csv(_reused_sweep_path))):
        _above = [(int(a), int(b)) for a, b in _tab.loc[_tab["dbcv"] > 0.3, ["min_cluster_size", "min_samples"]].values]
        print(f"settings with DBCV > 0.3, {_name}: {_above}")
    print(f"\nsaved -> {sweep_path} (reused run) and {_recomputed_path} (this run)")
else:
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nsaved -> {sweep_path}")


# ── 6.6  Cluster table cluster_results.csv ──────────────────────────────────
result_df = pd.DataFrame({
    "patent_id"  : patent_ids,
    "cluster_id" : cluster_labels,
    "umap_3d_x"  : embed_3d[:, 0],
    "umap_3d_y"  : embed_3d[:, 1],
    "umap_3d_z"  : embed_3d[:, 2],
})
result_df = result_df.merge(
    pipeline_df[["patent_id", "title", "final_summary"]],
    on="patent_id", how="left")
result_df = result_df.merge(
    add_ipc3(m1)[["patent_id", "current_assignee", "main_ipc_4", "main_ipc_3"]],
    on="patent_id", how="left")

result_path = os.path.join(SAVE_DIR, "cluster_results.csv")
result_df.to_csv(result_path, index=False)
print(f"saved -> {result_path} | columns: {list(result_df.columns)}")

# Overview of each cluster
cluster_df = result_df[result_df["cluster_id"] != -1]
n_noise = (result_df["cluster_id"] == -1).sum()
print(f"\nnoise: {n_noise} ({n_noise/len(result_df)*100:.1f}%)")
print(f"\n  {'cluster':>8} {'n':>6} {'share':>7}  {'top IPC':10} top company")
for cid in sorted(cluster_df["cluster_id"].unique()):
    rows = cluster_df[cluster_df["cluster_id"] == cid]
    top_ipc = rows["main_ipc_4"].value_counts().index[0]
    top_co  = rows["current_assignee"].value_counts().index[0][:32]
    print(f"  {cid:>8} {len(rows):>6} {len(rows)/len(result_df)*100:>6.1f}%  {top_ipc:10} {top_co}")


# ── 6.7  Keywords of each cluster (TF-IDF and KeyBERT style) ────────────────
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cluster_df = result_df[result_df["cluster_id"] != -1]
cluster_ids = sorted(cluster_df["cluster_id"].unique())

# TF-IDF (1-2 grams) on the joined document of each cluster
cluster_docs = {cid: " ".join(cluster_df[cluster_df["cluster_id"] == cid]["final_summary"])
                for cid in cluster_ids}
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english",
                             ngram_range=(1, 2), min_df=2)
tfidf_matrix  = vectorizer.fit_transform([cluster_docs[c] for c in cluster_ids])
feature_names = vectorizer.get_feature_names_out()

tfidf_keywords = {}
print("TF-IDF keywords (top 8 per cluster):")
for i, cid in enumerate(cluster_ids):
    scores  = tfidf_matrix[i].toarray()[0]
    top_idx = scores.argsort()[::-1][:8]
    kws     = [feature_names[j] for j in top_idx if scores[j] > 0]
    tfidf_keywords[cid] = kws
    print(f"  C{cid:>2} ({(cluster_df['cluster_id']==cid).sum():>4}): {', '.join(kws)}")

# KeyBERT style: SPECTER2 similarity between the cluster text and the candidate n-grams
def embed_texts_simple(texts):
    embs = []
    for t in texts:
        inputs = specter_tokenizer(t, return_tensors="pt", truncation=True,
                                   max_length=512, padding=True).to(specter_model.device)
        with torch.no_grad():
            out = specter_model(**inputs)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy()[0])
    return np.array(embs)

def extract_keywords_keybert(text, top_n=5):
    count = CountVectorizer(ngram_range=(1,2), stop_words="english",
                            max_features=50).fit([text])
    candidates = count.get_feature_names_out().tolist()
    if not candidates:
        return []
    doc_emb  = embed_texts_simple([text[:1000]])
    cand_emb = embed_texts_simple(candidates)
    sims = cosine_similarity(doc_emb, cand_emb)[0]
    return [candidates[i] for i in sims.argsort()[::-1][:top_n]]

keybert_keywords = {}
print("\nKeyBERT-style keywords (top 5 per cluster):")
for cid in tqdm(cluster_ids, desc="KeyBERT"):
    rows = cluster_df[cluster_df["cluster_id"] == cid]
    text = " ".join(rows["final_summary"].head(10).tolist())
    keybert_keywords[cid] = extract_keywords_keybert(text, top_n=5)
    print(f"  C{cid:>2}: {', '.join(keybert_keywords[cid])}")

free_gpu("specter_model", "specter_tokenizer")


# ── 6.8  Topic label of each cluster by the synthesis model ──────────────────
# Ten records of each cluster (seeded draw) go into one prompt; the model returns a label and five keywords.
# With a reused run and the same cluster assignment, the labels of that run are read and no model is loaded.
CLUSTER_LABEL_PROMPT = (
    "You are a patent analyst. Based on the following patent titles and summaries "
    "from the same technology cluster, provide:\n"
    "1. A concise cluster label (3-6 words)\n"
    "2. Top 5 technical keywords\n\n"
    "Format your response EXACTLY as:\n"
    "LABEL: <cluster label>\n"
    "KEYWORDS: <kw1>, <kw2>, <kw3>, <kw4>, <kw5>\n\n"
    "Patents:\n{patents}\n"
)

llm_labels, llm_keywords = {}, {}
_reuse_labels = (REUSED_CLUSTERS is not None and
                 (REUSED_CLUSTERS["cluster_id"].astype(int).values == cluster_labels).all())
if _reuse_labels:
    _lab = (pd.read_csv(_reused_clu, usecols=["cluster_id", "llm_label", "llm_keywords"])
              .fillna("").drop_duplicates("cluster_id").set_index("cluster_id"))
    for cid in cluster_ids:
        llm_labels[cid]   = _lab.loc[cid, "llm_label"]
        llm_keywords[cid] = str(_lab.loc[cid, "llm_keywords"]).split(", ")   # joined again with ", " below: identical text
        print(f"  C{cid:>2} ({int((cluster_df['cluster_id'] == cid).sum()):>4}): {llm_labels[cid]} | {', '.join(llm_keywords[cid])}")
    print(f"\n{len(llm_labels)} labels read from the reused run (no model loaded)")
else:
    if "tokenizer_s2" not in globals():
        tokenizer_s2 = AutoTokenizer.from_pretrained(S2_MODEL_NAME, trust_remote_code=True, padding_side="left")
        if tokenizer_s2.pad_token is None:
            tokenizer_s2.pad_token = tokenizer_s2.eos_token
    if "model_s2" not in globals():
        model_s2 = AutoModelForCausalLM.from_pretrained(
            S2_MODEL_NAME, torch_dtype=GEN_DTYPE, trust_remote_code=True,
            device_map="auto" if HAS_CUDA else None)
        if not HAS_CUDA:
            model_s2 = model_s2.to(DEVICE)
        model_s2 = model_s2.eval()
    with timed(f"6.8  Cluster labels ({S2_MODEL_NAME.split('/')[-1]})", n_items=len(cluster_ids)):
        for cid in tqdm(cluster_ids, desc="LLM labels"):
            rows   = cluster_df[cluster_df["cluster_id"] == cid]
            sample = rows.sample(min(10, len(rows)), random_state=RANDOM_STATE)
            patents_text = "\n".join([f"- {r['title']}: {r['final_summary'][:200]}..."
                                       for _, r in sample.iterrows()])
            output = generate_batch_s2([CLUSTER_LABEL_PROMPT.format(patents=patents_text)],
                                       max_new_tokens=100)[0]
            label, keywords = f"Cluster {cid}", []
            for line in output.split("\n"):
                if line.startswith("LABEL:"):
                    label = line.replace("LABEL:", "").strip()
                elif line.startswith("KEYWORDS:"):
                    keywords = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]
            llm_labels[cid], llm_keywords[cid] = label, keywords
            print(f"  C{cid:>2} ({len(rows):>4}): {label} | {', '.join(keywords)}")

# ── The only definition of the non-topic labels ───
NON_TOPIC_SENTINEL = "__noise__"
_JUNK_LABEL_PAT = re.compile(
    r"^(noise|unknown|unclassified|unlabel(l?ed)?|misc(ellaneous)?|other(s)?|"
    r"various|general|n/?a|none|null|cluster\s*-?\d+)(\s*\(\d+\))?$", re.I)
def is_junk_label(s):
    """The language model sometimes outputs uninformative labels such as Noise/Unknown. This function is only for the decision on "naming candidates";
    whether a record belongs to a topic is always decided by cluster_id == -1, not by a string comparison."""
    s = str(s).strip()
    return (not s) or s == NON_TOPIC_SENTINEL or bool(_JUNK_LABEL_PAT.match(s))

# Label clean-up: an uninformative label falls back to a placeholder name, for the naming logic of 6.9
_n_junk = sum(is_junk_label(v) for v in llm_labels.values())
llm_labels = {k: (f"Cluster {k}" if is_junk_label(v) else v) for k, v in llm_labels.items()}
print(f"\nLanguage-model label clean-up: {_n_junk} uninformative labels fell back to Cluster <id> ({len(llm_labels)} clusters in total)")

# cluster_id == -1 marks the HDBSCAN noise points, which have no label; a sentinel is used instead of "Noise",
# so that downstream steps do not treat it as a valid topic name
result_df["llm_label"]    = result_df["cluster_id"].map(llm_labels).fillna(NON_TOPIC_SENTINEL)
result_df["llm_keywords"] = result_df["cluster_id"].map(
    lambda x: ", ".join(llm_keywords.get(x, [])))
result_df.to_csv(result_path, index=False)
print(f"\nsaved LLM labels -> {result_path}")

free_gpu("model_s2", "tokenizer_s2")


# ── 6.9  Merge the clusters into N_TARGET topics (hierarchical, by centroid) ──
# The merge works on the HDBSCAN cluster_id of 6.4, and the names come from the labels that 6.8 made for the
# same cluster_id, so both use the same identifier space.
# Note: merging and naming always happen in the cluster_id space; if you add another clustering result,
#     make sure that the id spaces agree. Otherwise the label look-up takes the label of an arbitrary member.
import numpy as np, pandas as pd, os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# N_TARGET (number of merged topics) and MIN_CLASS_N (minimum records per topic; a smaller topic joins the
# topic with the most similar centroid) are set in 1.3.

if "llm_label" not in result_df.columns:
    raise RuntimeError("Run 6.8 first (topic labels).")

# Check of the id space: llm_label must be a function of cluster_id
_chk = result_df.groupby("cluster_id")["llm_label"].nunique()
if int(_chk.max()) > 1:
    raise RuntimeError("llm_label is not unique within one cluster_id; the id spaces do not agree: "
                       f"{_chk[_chk > 1].index.tolist()[:5]}")

cid      = result_df["cluster_id"].values
label_of = (result_df.drop_duplicates("cluster_id")
                     .set_index("cluster_id")["llm_label"].to_dict())
kw_of    = (result_df.drop_duplicates("cluster_id")
                     .set_index("cluster_id")["llm_keywords"].to_dict())

valid_ids = sorted(c for c in set(cid) if c != -1)
cent  = np.vstack([embeddings[cid == c].mean(axis=0) for c in valid_ids])
sizes = np.array([(cid == c).sum() for c in valid_ids])
print(f"clusters to merge: {len(valid_ids)} | noise(-1): {(cid == -1).sum()}")

try:                                   # sklearn >= 1.2
    agg = AgglomerativeClustering(n_clusters=min(N_TARGET, len(valid_ids)),
                                  metric="cosine", linkage="average")
except TypeError:                      # sklearn < 1.2
    agg = AgglomerativeClustering(n_clusters=min(N_TARGET, len(valid_ids)),
                                  affinity="cosine", linkage="average")
super_id = agg.fit_predict(cent)

# ── Naming: from the largest to the smallest, take the label of a member cluster that is "not yet used" ───
# When the largest members of two groups have the same name, use the next largest member, so that no meaningless "name (2)" is made.
name_of, used = {}, set()
for s in sorted(set(super_id), key=lambda x: -int(sizes[super_id == x].sum())):
    mem = sorted(np.where(super_id == s)[0], key=lambda i: -sizes[i])
    nm  = None
    for i in mem:
        cand = str(label_of.get(valid_ids[i], "")).strip()
        if cand and (cand not in used) and (not is_junk_label(cand)):
            nm = cand
            break
    if nm is None:
        # No member has a usable label: give a structured placeholder name, for manual naming in merge_map_worksheet.csv
        nm = f"Topic {int(s):02d}"
    used.add(nm)
    name_of[s] = nm

# ── Merge topics that are too small: move the whole group to the nearest other super-topic by cosine ───
for _ in range(len(set(super_id))):
    cnt   = {s: int(sizes[super_id == s].sum()) for s in set(super_id)}
    small = [s for s, n in cnt.items() if n < MIN_CLASS_N]
    if not small or len(cnt) <= 2:
        break
    s      = min(small, key=lambda x: cnt[x])
    others = [o for o in cnt if o != s]
    sc_s   = cent[super_id == s].mean(axis=0, keepdims=True)
    sc_o   = np.vstack([cent[super_id == o].mean(axis=0) for o in others])
    tgt    = others[int(np.argmax(cosine_similarity(sc_s, sc_o)[0]))]
    print(f"  merged: {name_of[s]} ({cnt[s]}) -> {name_of[tgt]}")
    super_id[super_id == s] = tgt

MERGE_MAP = {c: name_of[s] for c, s in zip(valid_ids, super_id)}
MERGE_MAP[-1] = NON_TOPIC_SENTINEL

# With a reused run and the same clusters, the topic of each cluster is taken from that run: the merge works on
# floating-point centroids, and a borderline merge can turn out differently on other hardware. The merge of
# this run is compared with it.
if REUSED_CLUSTERS is not None and _reuse_labels:
    _rmap = (pd.read_csv(_reused_clu, usecols=["cluster_id", "merged_topic"]).fillna("")
               .drop_duplicates("cluster_id").set_index("cluster_id")["merged_topic"].to_dict())
    _n_diff = sum(MERGE_MAP.get(c) != _rmap.get(c) for c in valid_ids)
    print(f"merge of this run vs the reused run: {len(valid_ids) - _n_diff} of {len(valid_ids)} clusters in the same topic"
          + ("" if _n_diff == 0 else " -> the topics of the reused run are kept"))
    MERGE_MAP = {c: _rmap.get(c, MERGE_MAP[c]) for c in valid_ids}
    MERGE_MAP[-1] = NON_TOPIC_SENTINEL
result_df["merged_topic"] = result_df["cluster_id"].map(MERGE_MAP).fillna(NON_TOPIC_SENTINEL)

# ── Hard assertion: change "false-topic contamination" from a silent defect into a failure at run time ───
_bad = sorted({t for t in result_df["merged_topic"].unique()
               if t != NON_TOPIC_SENTINEL and is_junk_label(t)})
assert not _bad, f"merged_topic has uninformative topic names; the clean-up of 6.8/6.9 did not work: {_bad}"

# ── Result table ───────────────────────────────────────────────────────────
print(f"\n{'merged topic':42} {'n':>6}  top IPC / top company")
rows = []
for merged, n in result_df["merged_topic"].value_counts().items():
    mr  = result_df[result_df["merged_topic"] == merged]
    ipc = " | ".join(f"{i}({k})" for i, k in mr["main_ipc_4"].value_counts().head(2).items())
    co  = " | ".join(f"{c[:20]}({k})" for c, k in mr["current_assignee"].value_counts().head(2).items())
    print(f"{merged[:42]:42} {n:>6}  {ipc}   {co}")
    rows.append({"merged_topic": merged, "patent_count": n, "top_ipc": ipc, "top_company": co})

NON_TOPIC_VALUES = [NON_TOPIC_SENTINEL, "Off-topic / Non-HPC", "Unclassified", ""]
_topic_only = result_df.loc[~result_df["merged_topic"].isin(NON_TOPIC_VALUES), "merged_topic"]
hpc = len(_topic_only)
print(f"\nTopics: {_topic_only.nunique()} (non-topic not included)"
      f" | with topic {hpc:,} ({hpc/len(result_df)*100:.1f}%)"
      f" | smallest topic {_topic_only.value_counts().min()} records"
      f" | non-topic {len(result_df)-hpc:,} ({(len(result_df)-hpc)/len(result_df)*100:.1f}%)")

result_df.to_csv(result_path, index=False)
pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "topic_summary.csv"), index=False)

# ── Worksheet for manual edits: use this table to rename topics or to mark Off-topic ───
sheet = pd.DataFrame([{
    "cluster_id": c,
    "n": int(sizes[i]),
    "auto_merged_topic": MERGE_MAP[c],
    "llm_label": label_of.get(c, ""),
    "llm_keywords": kw_of.get(c, ""),
    "top_ipc": result_df[cid == c]["main_ipc_4"].value_counts().index[:1].tolist(),
} for i, c in enumerate(valid_ids)]).sort_values(["auto_merged_topic", "n"],
                                                ascending=[True, False])
sheet.to_csv(os.path.join(SAVE_DIR, "merge_map_worksheet.csv"), index=False)
print("saved -> cluster_results.csv, topic_summary.csv, merge_map_worksheet.csv")

# The mapping is printed for review; merge_map_worksheet.csv holds the same information
print("\nMERGE_MAP = {")
for c in valid_ids:
    print(f'    {c:>4}: "{MERGE_MAP[c]}",')
print(f'    -1: "{NON_TOPIC_SENTINEL}",\n}}')


# ── 6.10  Optional renaming of topic labels (no change while RENAME is empty) ──
# The topic names come from the language model. Four pairs are nearly identical:
#   "Cybersecurity & Data Protection"   and "Cyber Security & Anomaly Detection"
#   "Power Management & Efficiency"     and "Power Management & Storage Optimization"
#   "Neural Network Optimization"       and "Neural Network Image Processing"
#   "Network Optimization & Management" and "High Performance Network Management"
# The released version 1.0 keeps the generated names. To rename, fill in RENAME and run this cell, then the
# cells that read the topic names (7.6, 7.7, 7.11 to 7.17, 6.11, 8.13). Only strings change; every number stays.
# 9.3 applies the same RENAME to topic_label.
RENAME = {
    # "old name": "new name",
}
_SENT = globals().get("NON_TOPIC_SENTINEL", "__noise__")
if RENAME:
    _cr = os.path.join(SAVE_DIR, "cluster_results.csv")
    result_df = pd.read_csv(_cr).fillna("")
    _before = set(result_df["merged_topic"])
    assert set(RENAME) <= _before, f"RENAME has old names that do not exist: {set(RENAME) - _before}"
    result_df["merged_topic"] = result_df["merged_topic"].replace(RENAME)
    _labels = [t for t in result_df["merged_topic"].unique() if t != _SENT]
    assert len(_labels) == len(set(_labels)), "duplicate topic names after the renaming"
    if "is_junk_label" in globals():
        assert not [t for t in _labels if is_junk_label(t)], "uninformative names after the renaming"
    result_df.to_csv(_cr, index=False)
    for _f, _col in (("topic_summary.csv", "merged_topic"), ("merge_map_worksheet.csv", "auto_merged_topic")):
        _p = os.path.join(SAVE_DIR, _f)
        if os.path.exists(_p):
            _t = pd.read_csv(_p).fillna(""); _t[_col] = _t[_col].replace(RENAME); _t.to_csv(_p, index=False)
    if "MERGE_MAP" in globals():
        MERGE_MAP = {k: RENAME.get(v, v) for k, v in MERGE_MAP.items()}
    print(f"Renamed {len(RENAME)} labels; {len(_labels)} topics. Run 7.6, 7.7, 7.11 to 7.17, 6.11 and 8.13 again.")
else:
    print("RENAME is empty: the generated topic names are kept.")


# ── 6.11  Evaluation table valid_df: all released records with their three summary layers ──
# The summary-layer measurements of section 8 do not depend on the topics and use every released record,
# so that the clustering does not select the evaluation sample. Only the within-topic similarity (8.13)
# uses topics; topic_mask selects the records with a topic of at least five records.
if "extract_tech_terms" not in globals():
    raise RuntimeError("Run 5.3 first (definition of the term extraction; section 8 uses it).")

pipeline_df = pd.read_csv(os.path.join(SAVE_DIR, "patent_full_pipeline.csv")).fillna("")
result_df   = pd.read_csv(os.path.join(SAVE_DIR, "cluster_results.csv")).fillna("")
stage1_df   = pd.read_csv(os.path.join(SAVE_DIR, "stage1_summaries.csv")).fillna("")

valid_df = (pipeline_df.merge(result_df[["patent_id", "merged_topic"]], on="patent_id", how="left")
                       .fillna("").reset_index(drop=True))
for _path, _col in ((DIRECT_PATH, "baseline_summary"), (MATCHED_PATH, "baseline_matched_summary")):
    _layer = pd.read_csv(_path).fillna("")[["patent_id", _col]]
    valid_df = valid_df.merge(_layer, on="patent_id", how="left").fillna("")
for _col in ("final_summary", "baseline_summary", "baseline_matched_summary"):
    _n_empty = int((valid_df[_col].astype(str).str.strip() == "").sum())
    print(f"  {_col:26} empty in {_n_empty:,} of {len(valid_df):,} records")

NON_TOPIC  = ["__noise__", "Off-topic / Non-HPC", "Unclassified", ""]
_tc        = valid_df.loc[~valid_df["merged_topic"].isin(NON_TOPIC), "merged_topic"].value_counts()
topic_mask = (~valid_df["merged_topic"].isin(NON_TOPIC)
              & valid_df["merged_topic"].isin(_tc[_tc >= 5].index)).values

groups_valid = build_family_groups(valid_df["patent_id"].values)
print(f"valid_df (summary-layer measurements): {len(valid_df):,} records | "
      f"families {pd.Series(groups_valid).nunique():,}")
print(f"records with a topic (for 8.13)       : {topic_mask.sum():,} records | "
      f"topics {valid_df.loc[topic_mask, 'merged_topic'].nunique()}")


# ── 6.12  SPECTER2 embeddings of the five representations ────────────────────
# Five representations are released side by side; their content is described, they are not ranked:
#   abstract             title [SEP] abstract      the existing abstract, about 115 words, no cut
#   fulltext-truncated   five fields joined         what is left after the 512-token cut of the encoder
#   direct-brief         title + baseline_summary   single prompt, short reading (about 2,000 tokens)
#   direct-full          title + baseline_matched   single prompt, full reading (about 5,000 tokens)
#   synthesized          title + final_summary      facets and synthesis, full reading (about 5,000 tokens)
# With a reused run, its released embeddings are read (same record order required); otherwise they are computed.
from transformers import AutoTokenizer as AT
from adapters import AutoAdapterModel
from sklearn.preprocessing import normalize as _l2norm

if "specter_model" not in globals():
    specter_tokenizer = AT.from_pretrained("allenai/specter2_base")
    specter_model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    _ = specter_model.load_adapter("allenai/specter2", source="hf",
                                   load_as="proximity", set_active=True)
    specter_model.set_active_adapters("proximity")
    specter_model = specter_model.eval().to(DEVICE)

EMB_BATCH = 64
def embed_batch(texts, desc="embed", batch_size=EMB_BATCH):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        inputs = specter_tokenizer(texts[i:i+batch_size], return_tensors="pt",
                                   truncation=True, padding=True, max_length=512
                                   ).to(specter_model.device)
        with torch.no_grad():
            out = specter_model(**inputs)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return _l2norm(np.vstack(embs), norm="l2")

def reused_or_embed(rep, texts, label):
    """Read the reused embeddings of one representation, or compute them."""
    _p = os.path.join(SAVE_DIR, "reused_embeddings", f"embeddings_{rep}.parquet")
    if os.path.exists(_p):
        _e = pd.read_parquet(_p)
        if list(_e["patent_id"].astype(str)) == list(valid_df["patent_id"].astype(str)):
            print(f"  {rep:20} read from the reused run")
            return _e.drop(columns=["patent_id"]).values.astype(np.float32)
        print(f"  {rep:20} reused file has another record order; computed again")
    with timed(f"6.12  SPECTER2 embedding ({label})", n_items=len(texts)):
        return embed_batch(texts, desc=f"embed [{rep}]")

_SUM_COL = "summary_filled" if "summary_filled" in valid_df.columns else "summary"
orig_texts = (valid_df["title"].astype(str) + " " + valid_df["abstract"].astype(str) +
              " " + valid_df["background"].astype(str) +
              " " + valid_df[_SUM_COL].astype(str) +
              " " + valid_df["detailed_description"].astype(str)).tolist()
X_orig = reused_or_embed("fulltext-truncated", orig_texts,
                         "fulltext-truncated: five fields joined, encoder cut at 512")
X_direct = reused_or_embed("direct-brief", (valid_df["title"].astype(str) + " " +
                                            valid_df["baseline_summary"].astype(str)).tolist(), "direct-brief")
# abstract: the native input of SPECTER2 is title + separator token + abstract; the abstracts fit into 512 tokens
_SEP = specter_tokenizer.sep_token or "[SEP]"
X_abs = reused_or_embed("abstract", (valid_df["title"].astype(str) + _SEP +
                                     valid_df["abstract"].astype(str)).tolist(), "abstract, title[SEP]abstract")
X_match = None
if valid_df["baseline_matched_summary"].astype(str).str.len().gt(0).any():
    X_match = reused_or_embed("direct-full", (valid_df["title"].astype(str) + " " +
                                              valid_df["baseline_matched_summary"].astype(str)).tolist(), "direct-full")

# synthesized: the embeddings of 6.2, normalised to unit length
embed_df = pd.read_csv(os.path.join(SAVE_DIR, "specter2_embeddings.csv"))
emb_cols = [c for c in embed_df.columns if c != "patent_id"]
_m = valid_df[["patent_id"]].merge(embed_df, on="patent_id", how="left")
assert _m[emb_cols].notna().all().all(), "the embeddings of 6.2 miss records; run 6.2 to the end first"
X_twostage = _l2norm(_m[emb_cols].values.astype(np.float32), norm="l2")
labels_valid = valid_df["merged_topic"].values

# The five representations for all later steps (a None entry is skipped)
REPRESENTATIONS = {
    "abstract":           X_abs,
    "fulltext-truncated": X_orig,
    "direct-brief":       X_direct,
    "direct-full":        X_match,
    "synthesized":        X_twostage,
}
for _k, _v in REPRESENTATIONS.items():
    print(f"  {_k:20} {'-' if _v is None else str(_v.shape)}")


# ── 6.13  Write the embeddings of the five representations (parquet, float32) ──
# Full text, three condensed layers and the embedding of each are released for the same records; this cell
# writes the third part. A CSV of 768-dimensional floats would be more than three times larger.
import os
import numpy as np, pandas as pd

EMB_DIR = os.path.join(SAVE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)
_ids   = valid_df["patent_id"].astype(str).values
_saved = []

for _rep, _X in REPRESENTATIONS.items():
    if _X is None:
        print(f"  (skipped {_rep}: not built)")
        continue
    if len(_X) != len(_ids):
        raise RuntimeError(f"{_rep}: {len(_X):,} rows, but valid_df has {len(_ids):,}")
    _cols = [f"dim_{i}" for i in range(_X.shape[1])]
    _out  = pd.DataFrame(np.asarray(_X, dtype=np.float32), columns=_cols)
    _out.insert(0, "patent_id", _ids)
    _base = f"embeddings_{_rep.replace(' ', '_').replace('/', '_')}"
    try:                                    # needs pyarrow or fastparquet
        _p = os.path.join(EMB_DIR, _base + ".parquet")
        _out.to_parquet(_p, index=False, compression="gzip")
    except Exception as _e:                 # falls back to a compressed CSV when no parquet engine is available
        _p = os.path.join(EMB_DIR, _base + ".csv.gz")
        _out.to_csv(_p, index=False, compression="gzip")
        print(f"  (parquet not available; saved as csv.gz: {type(_e).__name__})")
    _mb = os.path.getsize(_p) / 1e6
    _saved.append({"representation": _rep, "file": os.path.basename(_p),
                   "rows": int(len(_X)), "dim": int(_X.shape[1]), "MB": round(_mb, 2)})
    print(f"  {os.path.basename(_p):46} {len(_X):>6,} x {_X.shape[1]:>4}  {_mb:>7.2f} MB")

if not _saved:
    raise RuntimeError("No representation was written: run 6.12 first.")
pd.DataFrame(_saved).to_csv(os.path.join(EMB_DIR, "MANIFEST.csv"), index=False)
print(f"\n{len(_saved)} representations released -> {EMB_DIR}")
print("(specter2_embeddings.csv of 6.2 holds the synthesized representation before normalisation; the later steps read it)")
