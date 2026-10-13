"""Section 5.1-5.9: the synthesized summary layer (Qwen2.5-14B-Instruct).

Run from the pipeline notebook with run_step("s5_synthesized"); the parts below are executed in the namespace of the notebook.
"""

# ── 5.1  Synthesis model (the model is loaded only when summaries are still missing) ──
# The tokenizer is needed in every run: the token budgets of 5.10, 5.11, 8.1 and 8.6 are counted with it.
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_S2 = os.path.join(SAVE_DIR, "stage2_checkpoint.csv")
FULL_OUTPUT   = os.path.join(SAVE_DIR, "patent_full_pipeline.csv")
_done_s2 = (len(pd.read_csv(CHECKPOINT_S2, usecols=["patent_id"])) if os.path.exists(CHECKPOINT_S2) else 0)
S2_TODO  = max(len(stage1_df) - _done_s2, 0)

tokenizer_s2 = AutoTokenizer.from_pretrained(S2_MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer_s2.pad_token is None:
    tokenizer_s2.pad_token = tokenizer_s2.eos_token

if S2_TODO and "model_s2" not in globals():
    if not (HAS_CUDA or SMALL_MODELS):
        raise RuntimeError(f"{S2_TODO:,} synthesized summaries are missing and this machine has no CUDA GPU.")
    model_s2 = AutoModelForCausalLM.from_pretrained(
        S2_MODEL_NAME, torch_dtype=GEN_DTYPE, trust_remote_code=True,
        device_map="auto" if HAS_CUDA else None)
    if not HAS_CUDA:
        model_s2 = model_s2.to(DEVICE)
    # On a GPU with less than 32 GB, load in 4-bit instead:
    #   from transformers import BitsAndBytesConfig
    #   bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    #   model_s2 = AutoModelForCausalLM.from_pretrained(S2_MODEL_NAME, quantization_config=bnb, device_map="auto")
    model_s2 = model_s2.eval()
    print(f"{S2_MODEL_NAME} loaded | {S2_TODO:,} of {len(stage1_df):,} records still need a synthesized summary")
    if HAS_CUDA:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: allocated {torch.cuda.memory_allocated()/1e9:.1f} GB | "
              f"free {total - torch.cuda.memory_reserved()/1e9:.1f} GB")
else:
    print(f"Tokenizer of {S2_MODEL_NAME} loaded; the model is not loaded "
          f"({'already in memory' if 'model_s2' in globals() else 'all synthesized summaries are in the checkpoint'}).")


# ── 5.2  Generation helpers of the synthesis model ───────────────────────────
# The prompt is never cut silently: a prompt longer than MAX_INPUT_TOKENS_S2 stops the run with an error.
def truncate_text_s2(text: str, max_tokens: int = 1500) -> str:
    toks = tokenizer_s2.encode(str(text), add_special_tokens=False)
    if len(toks) > max_tokens:
        text = tokenizer_s2.decode(toks[:max_tokens], skip_special_tokens=True)
    return text

MAX_INPUT_TOKENS_S2 = 8192   # the longest prompt of the full corpus has 5,304 tokens with the chat format; model limit 32,768
def generate_batch_s2(prompts, max_new_tokens=256):
    if not prompts:
        return []
    tokenizer_s2.padding_side = "left"
    if tokenizer_s2.pad_token is None:
        tokenizer_s2.pad_token = tokenizer_s2.eos_token
    chats = [tokenizer_s2.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in prompts]
    _n_in = [len(x) for x in tokenizer_s2(chats)["input_ids"]]
    if max(_n_in) > MAX_INPUT_TOKENS_S2:   # no silent cut: stop when the input exceeds the limit
        raise ValueError(f"prompt of {max(_n_in)} tokens exceeds MAX_INPUT_TOKENS_S2 = {MAX_INPUT_TOKENS_S2}")
    inputs = tokenizer_s2(chats, return_tensors="pt", padding=True, truncation=False).to(model_s2.device)
    with torch.no_grad():
        out = model_s2.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer_s2.eos_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [tokenizer_s2.decode(g, skip_special_tokens=True).strip() for g in gen]

print("generate_batch_s2 ready")


# ── 5.3  Terms to keep (must_keep): the only definition of the extractor ─────
# The whitelist and the extractor are defined only here; 5.6 and the checks of section 8 use them.
HPC_WHITELIST = {
    # architecture
    "parallel","distributed","concurrent","pipelined","multiprocessor",
    "multi-core","many-core","heterogeneous","accelerator","coprocessor",
    "superscalar","out-of-order","speculative","vectorized",
    # memory
    "cache","bandwidth","latency","throughput","prefetch",
    "coherence","scratchpad","hierarchy","non-volatile","volatile",
    "high-bandwidth","near-memory","shared-memory","distributed-memory",
    "dram","sram","hbm","nvm","nvram","flash","persistent",
    # interconnect
    "interconnect","topology","routing","infiniband","fabric",
    "switched","mesh","torus","hypercube","crossbar","bisection",
    # computing models
    "sparse","stencil","convolution","scheduler","workload",
    "dataflow","pipeline","streaming","vectorization","parallelism",
    "simd","mimd","spmd","openmp","cuda","opencl","mpi",
    # semiconductors
    "transistor","semiconductor","photonic","cmos","wafer",
    "monolithic","chiplet","interposer","substrate",
    # networks
    "network","multicast","broadcast","unicast","packet","switch",
    # storage
    "storage","filesystem","checkpoint","journal","snapshot",
    "erasure","replication","deduplication","compression",
    # algorithms
    "gradient","optimization","quantization","pruning","sparsity",
    "pooling","normalization","backpropagation",
    # mathematics/geometry
    "manifold","diffeomorphic","topological","differentiable",
    "euclidean","riemannian","voxel","quaternion","tensor",
    # security
    "encryption","decryption","authentication","cryptographic",
    # HPC core
    "scalable","bottleneck","multi-level","high-performance",
    "fault-tolerant","load-balancing","real-time","multi-threaded",
    "multi-node","multi-gpu","multi-chip","many-to-many",
    # memory types
    "memory","buffer","register",
    # other
    "high","multi",
}

GENERIC_SUFFIXES = re.compile(
    r'(ing|tions|sion|tion|ments|ment|ness|ities|ity|ally|arly|onal|ical|ival|inal|ive|ous|ful|less|ary|ery|ory|ular|ated|ence|ance|ward|wise|ently|ingly|able|ible|ship|hood|ture|work|tics|ors|ers|ists|isms)$',
    re.I
)

TITLE_STOPWORDS = {
    "method","system","apparatus","device","process",
    "technique","approach","based","using","without","within",
    "through","between","among","their","these","those","which",
    "that","with","from","into","onto","upon","over","under",
    "about","after","before","during","while","when","where",
    "assessment","selection","image","media","level","type",
    "large","small","general","specific","various","certain",
    "given","known","existing","current","conventional",
    "proposed","present","consequently","therefore","however",
    "moreover","furthermore","meshes","methods","systems",
    "devices","approaches","results","issues","challenges",
    "problems","solutions","aspects","generation","serial",
    "digital","virtual","logical","physical","efficient",
}

def extract_tech_terms(title, *texts):
    """Return at most 10 must-keep technical terms:
    priority order: whitelist hits > informative title words > long words (>= 12 characters)."""
    combined = " ".join(texts)
    whitelist_hits, seen_wl = [], set()
    for w in re.findall(r'\b[a-zA-Z][a-zA-Z-]{2,}\b', title + " " + combined):
        lw = w.lower()
        if lw in HPC_WHITELIST and lw not in seen_wl:
            seen_wl.add(lw); whitelist_hits.append(lw)
    title_words, seen_tw = [], set()
    for w in re.findall(r'\b[a-zA-Z][a-zA-Z-]{4,}\b', title):
        lw = w.lower()
        if (lw not in TITLE_STOPWORDS and not GENERIC_SUFFIXES.search(lw)
                and lw not in seen_wl and lw not in seen_tw):
            seen_tw.add(lw); title_words.append(lw)
    long_words, seen_lw = [], set()
    for w in re.findall(r'\b[a-zA-Z][a-zA-Z-]{11,}\b', combined):
        lw = w.lower()
        if (lw not in TITLE_STOPWORDS and not GENERIC_SUFFIXES.search(lw)
                and lw not in seen_wl and lw not in seen_tw and lw not in seen_lw):
            seen_lw.add(lw); long_words.append(lw)
    return ", ".join((whitelist_hits + title_words + long_words)[:10])

print("extract_tech_terms ready |", len(HPC_WHITELIST), "whitelist terms")


# ── 5.4  Prompt templates of the synthesized layer ──────────────────────────
# The prompts are part of the method and are released as they are (9.1).
COT_REASONING_PROMPT = (
    "You are an expert patent analyst specializing in computing and "
    "engineering technologies.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY information explicitly stated in the Title, Abstract, "
    "and Section Summaries below.\n"
    "- Do NOT infer, assume, or add any applications, use cases, or "
    "benefits not directly mentioned in the input.\n"
    "- You MUST include key terms from the Title verbatim in your output.\n"
    "- If the Background Summary merely describes existing technology "
    "without stating a problem, derive the problem from the Abstract "
    "or Invention Summary instead.\n\n"
    "Given the following section summaries of a patent, output EXACTLY "
    "4 steps:\n\n"
    "Step 1 – The core technical PROBLEM or LIMITATION (1 sentence). "
    "Focus on what is broken, missing, or insufficient — "
    "NOT a description of existing technology.\n"
    "Step 2 – The proposed solution or key invention (1 sentence).\n"
    "Step 3 – The 2-3 most technically significant mechanisms. "
    "Write full descriptive sentences, do NOT just list keywords.\n"
    "Step 4 – The primary technical advantage (1 sentence, "
    "no inferred applications).\n\n"
    "IMPORTANT: Your output must naturally incorporate these technical "
    "terms where relevant: {must_keep}.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n"
    "Background Summary: {bg_summary}\n"
    "Invention Summary: {inv_summary}\n"
    "Technical Details Summary: {tech_summary}\n\n"
    "REASONING:"
)

FINAL_SYNTHESIS_PROMPT_V2 = (
    "You are a scientific writer producing a patent summary "
    "for a research journal audience.\n\n"
    "STRICT RULES:\n"
    "- Summarize ONLY what is explicitly stated in the inputs below.\n"
    "- Do NOT add applications, industries, or use cases not mentioned.\n"
    "- You MUST preserve these exact technical terms verbatim in your output: "
    "{must_keep}.\n"
    "- Every term listed above MUST appear at least once in your paragraph.\n"
    "\n"
    "Using the structured reasoning below, produce a single coherent "
    "paragraph of 5-7 sentences covering: the technical problem, "
    "proposed solution, key mechanisms, and technical advantages.\n\n"
    "Do NOT use bullet points, headers, bold, or markdown. "
    "Do NOT start with 'Here is', 'This patent', or 'This summary'.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n"
    "Structured Reasoning:\n{cot_reasoning}\n\n"
    "FINAL SUMMARY:"
)

# Optional third refinement pass (off by default, use_compression=False)
COMPRESSION_PROMPT = (
    "You are a technical editor. Refine the following patent summary to "
    "be concise and fluent. The refined summary must:\n"
    "- Be exactly 4-5 sentences\n"
    "- Preserve ALL key technical terms from the original\n"
    "- Remove redundancy and improve sentence transitions\n"
    "- Contain NO bullet points, headers, or markdown formatting\n\n"
    "ORIGINAL SUMMARY:\n{raw_summary}\n\n"
    "REFINED SUMMARY:"
)

print("Prompt templates ready")


# ── 5.5  Check on one record: reasoning call and synthesis call (only when the model is loaded) ──
if "model_s2" in globals():
    test_row = stage1_df.iloc[0]
    meta_row = m1[m1["patent_id"] == test_row["patent_id"]].iloc[0]
    must_keep = extract_tech_terms(
        meta_row["title"],
        test_row["bg_summary"], test_row["inv_summary"], test_row["tech_summary"])
    print(f"patent_id : {test_row['patent_id']}")
    print(f"title     : {meta_row['title']}")
    print(f"must_keep : {must_keep}\n")

    cot_out = generate_batch_s2([COT_REASONING_PROMPT.format(
        title=meta_row["title"],
        abstract=truncate_text_s2(meta_row["abstract"], 300),
        bg_summary=test_row["bg_summary"],
        inv_summary=test_row["inv_summary"],
        tech_summary=test_row["tech_summary"],
        must_keep=must_keep)], max_new_tokens=350)[0]
    print("Reasoning:\n" + cot_out)

    final_out = generate_batch_s2([FINAL_SYNTHESIS_PROMPT_V2.format(
        title=meta_row["title"],
        abstract=truncate_text_s2(meta_row["abstract"], 300),
        cot_reasoning=cot_out,
        must_keep=must_keep)], max_new_tokens=300)[0]
    print("\nSynthesized summary:\n" + final_out)

    _terms = [t.strip() for t in must_keep.split(",") if t.strip()]
    _kept_terms = [t for t in _terms if t.lower() in final_out.lower()]
    print(f"\nmust_keep terms in the summary: {len(_kept_terms)}/{len(_terms)}")
else:
    print("The synthesis model is not loaded (all synthesized summaries exist); the check on one record is skipped.")


# ── 5.6  Synthesized summaries: reasoning call, then synthesis call (checkpoint every 50) ──
# The output file patent_full_pipeline.csv joins m1, the facets and the two generated texts.

def run_stage2(stage1_df, meta_df, gpu_batch=S2_GPU_BATCH, save_every=50,
               use_compression=False):
    df = stage1_df.merge(
        meta_df[["patent_id","title","abstract"]],
        on="patent_id", how="left").fillna("")

    results = {"patent_id":[],"cot_reasoning":[],"raw_summary":[],"final_summary":[]}
    start_idx = 0
    if os.path.exists(CHECKPOINT_S2):
        existing  = pd.read_csv(CHECKPOINT_S2).fillna("")
        start_idx = len(existing)
        # The checkpoint must be a prefix of the same input, in the same order (as in 4.3)
        _prev = existing["patent_id"].astype(str).tolist()
        _curr = df["patent_id"].astype(str).tolist()
        if start_idx > len(_curr) or _prev != _curr[:start_idx]:
            raise RuntimeError(
                f"The synthesis checkpoint does not match the current corpus"
                f" (checkpoint {start_idx} records, input {len(_curr)} records). "
                "SAVE_DIR or REUSE_DIR belongs to another corpus version: use a new SAVE_DIR, "
                "or a REUSE_DIR whose corpus is the same.")
        for col in results:
            results[col] = existing[col].tolist()
        print(f"checkpoint found: {start_idx}/{len(df)} done, resuming")
    else:
        print(f"starting from scratch: {len(df)} records")

    if start_idx >= len(df):
        stage2_df = pd.DataFrame(results)
        if not os.path.exists(FULL_OUTPUT):
            meta_df \
                .merge(stage1_df, on="patent_id", how="left") \
                .merge(stage2_df[["patent_id","cot_reasoning","raw_summary","final_summary"]],
                       on="patent_id", how="left") \
                .to_csv(FULL_OUTPUT, index=False)
        print(f"All {len(df):,} synthesized summaries are in the checkpoint; nothing to generate.")
        return stage2_df
    print(f"range {start_idx} -> {len(df)} | batch={gpu_batch} | "
          f"compression={'ON' if use_compression else 'OFF'}\n")

    pbar = tqdm(total=len(df) - start_idx, desc="synthesis")
    for chunk_start in range(start_idx, len(df), gpu_batch):
        chunk_end = min(chunk_start + gpu_batch, len(df))
        rows = [df.iloc[i] for i in range(chunk_start, chunk_end)]

        must_keeps = [
            extract_tech_terms(r["title"], r["bg_summary"],
                               r["inv_summary"], r["tech_summary"])
            for r in rows]

        cot_outs = generate_batch_s2([
            COT_REASONING_PROMPT.format(
                title=r["title"],
                abstract=truncate_text_s2(r["abstract"], 300),
                bg_summary=r["bg_summary"],
                inv_summary=r["inv_summary"],
                tech_summary=r["tech_summary"],
                must_keep=must_keeps[i],
            ) for i, r in enumerate(rows)], max_new_tokens=350)

        raw_outs = generate_batch_s2([
            FINAL_SYNTHESIS_PROMPT_V2.format(
                title=rows[i]["title"],
                abstract=truncate_text_s2(rows[i]["abstract"], 300),
                cot_reasoning=cot_outs[i],
                must_keep=must_keeps[i],
            ) for i in range(len(rows))], max_new_tokens=300)

        final_outs = generate_batch_s2([
            COMPRESSION_PROMPT.format(raw_summary=raw_outs[i])
            for i in range(len(rows))], max_new_tokens=200) if use_compression else raw_outs

        for i, r in enumerate(rows):
            results["patent_id"].append(r["patent_id"])
            results["cot_reasoning"].append(cot_outs[i])
            results["raw_summary"].append(raw_outs[i])
            results["final_summary"].append(final_outs[i])

        pbar.update(chunk_end - chunk_start)
        processed = chunk_end - start_idx
        if processed % save_every == 0 or chunk_end == len(df):
            pd.DataFrame(results).to_csv(CHECKPOINT_S2, index=False)
    pbar.close()

    stage2_df = pd.DataFrame(results)
    stage2_df.to_csv(CHECKPOINT_S2, index=False)

    meta_df \
        .merge(stage1_df, on="patent_id", how="left") \
        .merge(stage2_df[["patent_id","cot_reasoning","raw_summary","final_summary"]],
               on="patent_id", how="left") \
        .to_csv(FULL_OUTPUT, index=False)

    print(f"\nSynthesized summaries complete: {len(stage2_df)} records")
    for col in ["cot_reasoning", "raw_summary", "final_summary"]:
        empty = (stage2_df[col] == "").sum()
        print(f"  {col:20s}: empty {empty:>5}/{len(stage2_df)} ({empty/len(stage2_df)*100:.1f}%)")
    print(f"saved -> {FULL_OUTPUT}")
    return stage2_df

_done_before = len(pd.read_csv(CHECKPOINT_S2)) if os.path.exists(CHECKPOINT_S2) else 0
with timed(f"5.6  Synthesized summary generation ({S2_MODEL_NAME.split('/')[-1]})") as t:
    stage2_df = run_stage2(stage1_df, m1, gpu_batch=S2_GPU_BATCH, save_every=50)
    t.n_items = max(len(stage2_df) - _done_before, 0) or None
    t.note    = f"{t.n_items or 0} records generated in this run; batch={S2_GPU_BATCH}"


# ── 5.7  Reload the synthesized layer and check it against the corpus ────────
FULL_OUTPUT = os.path.join(SAVE_DIR, "patent_full_pipeline.csv")
pipeline_df = pd.read_csv(FULL_OUTPUT).fillna("")
print(f"pipeline_df: {len(pipeline_df)} records")
for col in ["patent_id", "title", "abstract", "final_summary"]:
    print(f"  {col:15s} empty: {(pipeline_df[col]=='').sum()}")

# The identifiers must be those of the current corpus (see 4.4)
if "m1" in globals():
    _p, _m = set(pipeline_df["patent_id"].astype(str)), set(m1["patent_id"].astype(str))
    if _p != _m:
        raise RuntimeError(
            f"patent_full_pipeline.csv does not match the current corpus ({len(_p)} vs {len(_m)} records). "
            "SAVE_DIR belongs to another corpus version. Use a new SAVE_DIR.")
    print("patent_id alignment check passed")


# ── 5.8  Completeness, length and term retention of the synthesized layer (20 records) ──
# (a) completeness and length
test_stage2 = pipeline_df.sample(min(20, len(pipeline_df)), random_state=RANDOM_STATE).reset_index(drop=True)
for col in ["cot_reasoning", "raw_summary", "final_summary"]:
    empty = (test_stage2[col] == "").sum()
    short = (test_stage2[col].str.split().str.len() < 30).sum()
    print(f"  {col:20s}: empty {empty} | short(<30 words) {short}")
for col in ["cot_reasoning", "final_summary"]:
    lengths = test_stage2[col].str.split().str.len()
    print(f"  {col:20s}: mean={lengths.mean():.0f} | min={lengths.min()} | max={lengths.max()} words")

# (b) must_keep retention on the same records
hit, total_kw = 0, 0
for _, row in test_stage2.iterrows():
    s1_row = stage1_df[stage1_df["patent_id"] == row["patent_id"]]
    mrow   = m1[m1["patent_id"] == row["patent_id"]]
    if len(s1_row) == 0 or len(mrow) == 0:
        continue
    s1 = s1_row.iloc[0]
    mk = extract_tech_terms(mrow.iloc[0]["title"],
                            s1["bg_summary"], s1["inv_summary"], s1["tech_summary"])
    terms = [t.strip().lower() for t in mk.split(",") if t.strip()]
    hit      += sum(1 for t in terms if t in row["final_summary"].lower())
    total_kw += len(terms)
print(f"\nmust_keep retention ({len(test_stage2)} records): {hit}/{total_kw} ({hit/max(total_kw, 1)*100:.1f}%)")


# ── 5.9  Perplexity and similarity to the source facets (100 records) ───────
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

def compute_perplexity(text: str) -> float:
    if not text.strip():
        return float("inf")
    inputs = gpt2_tok(text, return_tensors="pt", truncation=True,
                      max_length=512).to(DEVICE)
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def cosine_to_facets(summary: str, reference: str) -> float:
    """Cosine similarity between the synthesized summary and the three facets that it was made from."""
    if not summary.strip() or not reference.strip():
        return 0.0
    embs = sbert.encode([summary, reference])
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])

QC_N = min(100, len(pipeline_df))
qc_sample = pipeline_df.sample(QC_N, random_state=RANDOM_STATE)

ppl_list, cos_to_facets_list = [], []
for _, row in tqdm(qc_sample.iterrows(), total=QC_N, desc="synthesis checks"):
    ppl_list.append(compute_perplexity(row["final_summary"]))
    reference = " ".join([row.get("bg_summary",""), row.get("inv_summary",""),
                          row.get("tech_summary","")])
    cos_to_facets_list.append(cosine_to_facets(row["final_summary"], reference))

ppl_arr    = np.array([p for p in ppl_list if p != float("inf")])
cos_to_facets_arr = np.array(cos_to_facets_list)

print(f"\n{'='*56}")
print(f"Synthesized layer, measurements (n={QC_N} records drawn at random)")
print(f"{'='*56}")
print(f"Perplexity : mean {ppl_arr.mean():.2f} | median {np.median(ppl_arr):.2f} | "
      f"p90 {np.percentile(ppl_arr, 90):.2f}")
print(f"Cosine to facets : mean {cos_to_facets_arr.mean():.4f} | median {np.median(cos_to_facets_arr):.4f} | "
      f"p10 {np.percentile(cos_to_facets_arr, 10):.4f}")
print("reference values used during development: PPL mean < 80 | PPL p90 < 150 | cos_to_facets mean > 0.75")

low_cos_to_facets = [(qc_sample.iloc[i]["patent_id"], cos_to_facets_list[i])
              for i in range(QC_N) if cos_to_facets_list[i] < 0.60]
print(f"\nrecords with cos_to_facets < 0.60: {len(low_cos_to_facets)}")
for pid, pur in low_cos_to_facets[:5]:
    print(f"  {pid}: {pur:.4f}")

free_gpu("gpt2_model", "gpt2_tok", "sbert")
