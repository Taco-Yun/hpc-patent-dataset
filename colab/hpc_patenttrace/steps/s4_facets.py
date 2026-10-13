"""Section 4: three facets per record (Llama-3.1-8B-Instruct, 4-bit).

Run from the pipeline notebook with run_step("s4_facets"); the parts below are executed in the namespace of the notebook.
"""

# ── 4.1  Facet model (loaded only when facets are still missing) ─────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CHECKPOINT_S1 = os.path.join(SAVE_DIR, "stage1_checkpoint.csv")
STAGE1_OUTPUT = os.path.join(SAVE_DIR, "stage1_summaries.csv")
_done_s1 = (len(pd.read_csv(CHECKPOINT_S1, usecols=["patent_id"])) if os.path.exists(CHECKPOINT_S1) else 0)
S1_TODO  = max(len(m1) - _done_s1, 0)

tokenizer, model = None, None
if S1_TODO:
    if not (HAS_CUDA or SMALL_MODELS):
        raise RuntimeError(f"{S1_TODO:,} facets are missing and this machine has no CUDA GPU. "
                           "Run on a GPU, set REUSE_DIR to a run with complete facets, or use PROFILE = 'sample'.")
    if S1_MODEL_ID.startswith("meta-llama/"):
        ensure_hf_login()                        # gated model
    tokenizer = AutoTokenizer.from_pretrained(S1_MODEL_ID)
    if S1_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            S1_MODEL_ID, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(S1_MODEL_ID, torch_dtype=GEN_DTYPE).to(DEVICE)
    model = model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    print(f"{S1_MODEL_ID} loaded | {model.get_memory_footprint()/1e9:.2f} GB | "
          f"{S1_TODO:,} of {len(m1):,} records still need facets")
else:
    print(f"All {len(m1):,} facets are in {os.path.basename(CHECKPOINT_S1)}; the facet model is not loaded.")


# ── 4.2  Facet prompts and batch generation ─────────────────────────────────
def truncate_text(text: str, max_tokens: int = 1500) -> str:
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) > max_tokens:
        text = tokenizer.decode(toks[:max_tokens], skip_special_tokens=True)
    return text

# The prompts are part of the method and are released as they are (9.1 writes them to release_assets/prompts/).
SECTION_PROMPTS = {
    "BACKGROUND": (
        "You are a patent analyst specializing in High-Performance Computing (HPC). "
        "Summarize the BACKGROUND section below in 2-3 sentences as a single "
        "continuous paragraph. Do NOT use bullet points, headers, bold text, "
        "or markdown formatting. Do NOT start with phrases like 'Here is' or "
        "'This section describes'. Just write the summary directly.\n\n"
        "BACKGROUND:\n{text}\n\nSUMMARY:"
    ),
    "SUMMARY": (
        "You are a patent analyst specializing in High-Performance Computing (HPC). "
        "Condense the following patent SUMMARY into 2-3 sentences as a single "
        "continuous paragraph covering the core invention and key technical "
        "contribution. Do NOT use bullet points, headers, bold text, or markdown "
        "formatting. Do NOT start with phrases like 'Here is' or 'This patent'. "
        "Just write the summary directly.\n\nPATENT SUMMARY:\n{text}\n\nCONDENSED:"
    ),
    "DETAILED DESCRIPTION": (
        "You are a patent analyst specializing in High-Performance Computing (HPC). "
        "Summarize the DETAILED DESCRIPTION below in 3-4 sentences as a single "
        "continuous paragraph covering the architecture or method, key components, "
        "and the novel technical mechanism. Omit implementation-specific details. "
        "Do NOT use bullet points, headers, bold text, or markdown formatting. "
        "Do NOT start with phrases like 'Here is' or 'This section'. "
        "Just write the summary directly.\n\nDETAILED DESCRIPTION:\n{text}\n\nSUMMARY:"
    ),
}

def generate_batch(prompts, max_new_tokens=256):
    """Batch greedy generation (left padding, with the chat template)."""
    if len(prompts) == 0:
        return []
    tokenizer.padding_side = "left"
    chats = [tokenizer.apply_chat_template(
                [{"role":"user","content":p}],
                tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tokenizer(chats, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

def generate_summary(prompt, max_new_tokens=256):
    return generate_batch([prompt], max_new_tokens)[0]

# Check on the first record (only when the model is loaded)
if model is not None:
    row = m1.iloc[0]
    print(f"Title: {row['title']}\n")
    print("Background facet:")
    print(generate_summary(SECTION_PROMPTS["BACKGROUND"].format(
        text=truncate_text(row["background"], 1500)), 150))
else:
    print("Prompts ready (the facet model is not loaded; see 4.1).")


# ── 4.3  Facet generation (checkpoint every 50 records) ──────────────────────
# Each section that has at least MIN_LEN characters is cut to its token budget and summarised; a shorter
# section gives an empty facet. The checkpoint must be a prefix of the current input in the same order.

def run_stage1(df_input, gpu_batch=S1_GPU_BATCH, save_every=50):
    results = {"patent_id": [], "bg_summary": [], "inv_summary": [], "tech_summary": []}
    start_idx = 0
    if os.path.exists(CHECKPOINT_S1):
        existing = pd.read_csv(CHECKPOINT_S1).fillna("")
        start_idx = len(existing)
        # A row count alone would accept the checkpoint of another corpus version, so the identifiers are compared in order.
        _prev = existing["patent_id"].astype(str).tolist()
        _curr = df_input["patent_id"].astype(str).tolist()
        if start_idx > len(_curr) or _prev != _curr[:start_idx]:
            raise RuntimeError(
                f"The facet checkpoint does not match the current corpus"
                f" (checkpoint {start_idx} records, input {len(_curr)} records). "
                "SAVE_DIR or REUSE_DIR belongs to another corpus version: use a new SAVE_DIR, "
                "or a REUSE_DIR whose corpus is the same.")
        for col in results:
            results[col] = existing[col].tolist()
        print(f"checkpoint found: {start_idx}/{len(df_input)} done, resuming")
    else:
        print(f"starting from scratch: {len(df_input)} records")

    end_idx = len(df_input)
    if start_idx >= end_idx:
        stage1_df = pd.DataFrame(results)
        if not os.path.exists(STAGE1_OUTPUT):
            stage1_df.to_csv(STAGE1_OUTPUT, index=False)
        print(f"All {end_idx:,} facets are in the checkpoint; nothing to generate.")
        return stage1_df
    print(f"range {start_idx} -> {end_idx} | gpu_batch={gpu_batch}\n")

    def _collect_prompts(rows, section_key, field, in_trunc):
        """Collect the prompts only for the rows whose section length passes MIN_LEN."""
        prompts, valid_idx = [], []
        for i, r in enumerate(rows):
            text = r.get(field, "")
            if len(text) >= MIN_LEN:
                prompts.append(
                    SECTION_PROMPTS[section_key].format(text=truncate_text(text, in_trunc))
                )
                valid_idx.append(i)
        return prompts, valid_idx

    pbar = tqdm(total=end_idx - start_idx, desc="facets")
    for chunk_start in range(start_idx, end_idx, gpu_batch):
        chunk_end = min(chunk_start + gpu_batch, end_idx)
        rows = [df_input.iloc[i] for i in range(chunk_start, chunk_end)]
        n = len(rows)

        # Join the three sections into one large batch for better GPU use
        bg_prompts,   bg_idx   = _collect_prompts(rows, "BACKGROUND",           "background",           1500)
        inv_prompts,  inv_idx  = _collect_prompts(rows, "SUMMARY",              "summary_filled",       1500)
        tech_prompts, tech_idx = _collect_prompts(rows, "DETAILED DESCRIPTION", "detailed_description", 2000)

        all_prompts = bg_prompts + inv_prompts + tech_prompts
        all_gens    = generate_batch(all_prompts, max_new_tokens=200) if all_prompts else []

        b1 = len(bg_prompts); b2 = b1 + len(inv_prompts)
        bg_gens, inv_gens, tech_gens = all_gens[:b1], all_gens[b1:b2], all_gens[b2:]

        bgs, invs, techs = [""]*n, [""]*n, [""]*n
        for k, g in enumerate(bg_gens):   bgs[bg_idx[k]]     = g
        for k, g in enumerate(inv_gens):  invs[inv_idx[k]]   = g
        for k, g in enumerate(tech_gens): techs[tech_idx[k]] = g

        for i, r in enumerate(rows):
            results["patent_id"].append(r["patent_id"])
            results["bg_summary"].append(bgs[i])
            results["inv_summary"].append(invs[i])
            results["tech_summary"].append(techs[i])

        pbar.update(chunk_end - chunk_start)
        processed = chunk_end - start_idx
        if processed % save_every == 0 or chunk_end == end_idx:
            pd.DataFrame(results).to_csv(CHECKPOINT_S1, index=False)
    pbar.close()

    stage1_df = pd.DataFrame(results)
    stage1_df.to_csv(CHECKPOINT_S1, index=False)
    stage1_df.to_csv(STAGE1_OUTPUT, index=False)
    m1.merge(stage1_df, on="patent_id", how="left").to_csv(
        os.path.join(SAVE_DIR, "patent_with_stage1.csv"), index=False)

    print(f"\nFacets complete: {len(stage1_df)} records")
    for col in ["bg_summary", "inv_summary", "tech_summary"]:
        empty = (stage1_df[col] == "").sum()
        pct   = empty / len(stage1_df) * 100
        print(f"  {col:15s}: empty {empty:>5}/{len(stage1_df)} ({pct:.1f}%)")
    print(f"saved -> {STAGE1_OUTPUT}")
    return stage1_df

_done_before = len(pd.read_csv(CHECKPOINT_S1)) if os.path.exists(CHECKPOINT_S1) else 0
with timed(f"4.3  Facet generation ({S1_MODEL_ID.split('/')[-1]})") as t:
    stage1_df = run_stage1(m1, gpu_batch=S1_GPU_BATCH, save_every=50)
    t.n_items = max(len(stage1_df) - _done_before, 0) or None
    t.note    = f"{t.n_items or 0} records generated in this run; batch={S1_GPU_BATCH}"


# ── 4.4  Reload the facets and check them against the corpus (also for a new session) ──
STAGE1_OUTPUT = os.path.join(SAVE_DIR, "stage1_summaries.csv")
assert os.path.exists(STAGE1_OUTPUT), f"missing {STAGE1_OUTPUT}"
stage1_df = pd.read_csv(STAGE1_OUTPUT).fillna("")
print(f"stage1_df: {len(stage1_df)} records | columns: {list(stage1_df.columns)}")

# Compare the patent_id sets (not only the row counts): a mismatch means that this output belongs to another corpus version
_s1, _m1 = set(stage1_df["patent_id"].astype(str)), set(m1["patent_id"].astype(str))
if _s1 != _m1:
    raise RuntimeError(
        f"stage1_summaries.csv does not match the current corpus: stage1={len(stage1_df)} records, "
        f"m1={len(m1)} records; only in stage1: {len(_s1 - _m1)}, "
        f"only in m1: {len(_m1 - _s1)}. "
        "SAVE_DIR or REUSE_DIR belongs to another corpus version: use a new SAVE_DIR and run 4.3 again.")
print("patent_id alignment check passed")


# ── 4.5  Completeness, length and repetition of the facets ───────────────────
S1_COLS = ["bg_summary", "inv_summary", "tech_summary"]
S1_NAMES = {"bg_summary": "background facet", "inv_summary": "invention facet", "tech_summary": "technical facet"}

facet_check = pd.DataFrame([{
    "facet": col,
    "empty": int((stage1_df[col] == "").sum()),
    "empty_percent": round((stage1_df[col] == "").mean() * 100, 1),
    "shorter_than_50_chars": int((stage1_df[col].str.len() < 50).sum()),
    "mean_words": round(stage1_df[col].str.split().str.len().mean(), 1),
    # a repeated block of 30 or more characters is a sign of a generation loop
    "suspected_loops": int(stage1_df[col].str.contains(r"(.{30,})\1", regex=True).sum()),
} for col in S1_COLS])
print(f"records: {len(stage1_df):,}")
facet_check

print("Five facets drawn at random, for reading:")
for _, row in stage1_df.sample(min(5, len(stage1_df)), random_state=RANDOM_STATE).iterrows():
    print(f"\n  {row['patent_id']}")
    print(f"  background : {row['bg_summary'][:140]}...")
    print(f"  invention  : {row['inv_summary'][:140]}...")
    print(f"  technical  : {row['tech_summary'][:140]}...")

# Figure 4.1: length of the three facets
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
for ax, col in zip(axes, S1_COLS):
    lengths = stage1_df[col].str.split().str.len()
    lengths = lengths[lengths > 0]
    _ = ax.hist(lengths, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    _ = ax.axvline(lengths.mean(), color=INK_2, linestyle="--", linewidth=1)
    _ = ax.text(lengths.mean(), ax.get_ylim()[1] * 0.92, f" mean {lengths.mean():.0f}", fontsize=8, color=INK_2)
    _ = ax.set_title(S1_NAMES[col])
    _ = ax.set_xlabel("words")
_ = axes[0].set_ylabel("records")
_ = fig.suptitle("Figure 4.1  Length of the facets (non-empty facets only)", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
_ = save_fig(fig, "fig_4_1_facet_length")
plt.show()


# ── 4.6  Rule checks and GPT-2 perplexity of the facets (100 records) ──────
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

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

PROMPT_LEAKAGE = ["Just write", "Do NOT", "You are a", "SUMMARY:", "CONDENSED:",
                  "Here is", "This section", "I'm happy to help", "I must point out",
                  "This background assumes that unless explicitly stated",
                  "unless explicitly stated as published prior art"]

def check_rule_violations(text: str) -> dict:
    """Rule-based violation check: prompt leftovers, loops, too short, markdown traces."""
    issues = []
    if any(p.lower() in text.lower() for p in PROMPT_LEAKAGE):
        issues.append("prompt_leak")
    if re.search(r"(.{30,})\1", text):
        issues.append("loop")
    if len(text.split()) < 30:
        issues.append("too_short")
    if re.search(r"[\*\#\-]{2,}|^\s*[\*\-]\s", text, re.MULTILINE):
        issues.append("markdown")
    return {"is_pure": len(issues) == 0, "issues": issues}

QC_N = min(100, len(stage1_df))
qc_sample = stage1_df.sample(QC_N, random_state=RANDOM_STATE)
qc = {col: {"ppls": [], "pure": 0, "issues": {}} for col in S1_COLS}

for _, row in tqdm(qc_sample.iterrows(), total=QC_N, desc="facet checks"):
    for col in S1_COLS:
        text = row[col]
        qc[col]["ppls"].append(compute_perplexity(text))
        pur = check_rule_violations(text)
        if pur["is_pure"]:
            qc[col]["pure"] += 1
        for issue in pur["issues"]:
            qc[col]["issues"][issue] = qc[col]["issues"].get(issue, 0) + 1

print(f"\n{'='*64}")
print(f"Facet measurements (n={QC_N} records drawn at random)")
print(f"{'='*64}")
print(f"{'facet':<16} {'rule_pass':>8} {'PPL mean':>10} {'PPL med':>9} {'PPL p90':>9}")
for col in S1_COLS:
    ppls = [p for p in qc[col]["ppls"] if p != float("inf")]
    rule_pass_pct = qc[col]["pure"] / QC_N * 100
    print(f"{col:<16} {rule_pass_pct:>7.1f}% {np.mean(ppls):>10.1f} "
          f"{np.median(ppls):>9.1f} {np.percentile(ppls, 90):>9.1f}")
    for issue, cnt in qc[col]["issues"].items():
        print(f"    - {issue}: {cnt}")
print("\nreference values used during development: rule_pass >= 90% | PPL mean < 80 | PPL p90 < 150")

free_gpu("gpt2_model", "gpt2_tok")


# ── 4.7  Release the facet model before section 5 ─────────────────────────
free_gpu("model", "tokenizer")
