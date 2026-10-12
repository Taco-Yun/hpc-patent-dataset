"""Section 5.10-5.12: the direct-brief and direct-full layers.

Run from the pipeline notebook with run_step("s5_direct"); the parts below are executed in the namespace of the notebook.
"""

# ── 5.10  direct-brief layer: one prompt over a short reading of the sections ──
# The same model reads the main fields directly with one common prompt. The target length is the same as for
# the synthesized layer (5-7 sentences), so that length-sensitive measures compare layers of similar length.
# Budgets (tokens of the synthesis model): background 800, summary 400, detailed description 800.
# The prompt is part of the method and is released as it is (9.1). Checkpoint: baseline_direct.csv.
DIRECT_PROMPT = (
    "You are a patent analyst specializing in High-Performance Computing (HPC).\n\n"
    "Read the following patent text and write a concise technical summary "
    "of 5-7 sentences as a single paragraph.\n"
    "Cover: the core technical problem, proposed solution, and key mechanisms.\n"
    "Do NOT use bullet points, headers, or markdown.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n\n"
    "Background:\n{background}\n\n"
    "Summary:\n{summary}\n\n"
    "Detailed Description:\n{detail}\n\n"
    "SUMMARY:"
)

DIRECT_PATH = os.path.join(SAVE_DIR, "baseline_direct.csv")
_SUM_COL = "summary_filled" if "summary_filled" in pipeline_df.columns else "summary"
_BATCH = BASE_BATCH

def _direct_brief_prompt(row):
    return DIRECT_PROMPT.format(
        title      = str(row["title"]),
        abstract   = str(row["abstract"]),
        background = truncate_text_s2(row["background"],           800),
        summary    = truncate_text_s2(row[_SUM_COL],               400),
        detail     = truncate_text_s2(row["detailed_description"], 800),
    )

pids = pipeline_df["patent_id"].tolist()
if os.path.exists(DIRECT_PATH):
    done_df  = pd.read_csv(DIRECT_PATH).fillna("")
    done_ids = set(done_df["patent_id"])
    remain   = [i for i, p in enumerate(pids) if p not in done_ids]
    results  = done_df.to_dict("records")
    print(f"checkpoint: {len(done_ids)} done, {len(remain)} remaining")
else:
    remain, results = list(range(len(pids))), []
# prompts are built only for the records that still need a summary (tokenising long sections takes minutes)
prompts = {i: _direct_brief_prompt(pipeline_df.iloc[i]) for i in tqdm(remain, desc="build prompts")}

if remain and "model_s2" not in globals():
    model_s2 = AutoModelForCausalLM.from_pretrained(
        S2_MODEL_NAME, torch_dtype=GEN_DTYPE, trust_remote_code=True,
        device_map="auto" if HAS_CUDA else None)
    if not HAS_CUDA:
        model_s2 = model_s2.to(DEVICE)
    model_s2 = model_s2.eval()

with timed(f"5.10  direct-brief generation ({S2_MODEL_NAME.split('/')[-1]}, single prompt)",
           n_items=len(remain) or None) as t:
    t.note = f"{len(remain)} records generated in this run; batch={_BATCH}"
    pbar = tqdm(range(0, len(remain), _BATCH), desc="direct-brief")
    for b in pbar:
        chunk = remain[b : b + _BATCH]
        outs = generate_batch_s2([prompts[i] for i in chunk], max_new_tokens=300)
        for i, out in zip(chunk, outs):
            results.append({"patent_id": pids[i], "baseline_summary": out})
        pbar.set_description(f"direct-brief ({len(results)}/{len(pids)})")
        if len(results) % 100 < _BATCH:
            pd.DataFrame(results).to_csv(DIRECT_PATH, index=False)
pd.DataFrame(results).to_csv(DIRECT_PATH, index=False)
_wl = pd.DataFrame(results)["baseline_summary"].astype(str).str.split().str.len()
print(f"saved -> {DIRECT_PATH} ({len(results):,} records) | mean {_wl.mean():.0f} words | empty {(_wl == 0).sum()}")


# ── 5.11  direct-full layer: the same prompt with the input budget of the facets ──
# direct-brief reads about 2,000 tokens, while the facet step reads about 5,000. This layer gives the single
# prompt the facet budgets section by section (see _collect_prompts in 4.3) and changes nothing else: the same
# model, precision, greedy decoding, repetition penalty, output limit, target length and prompt template.
# The three layers then form a simple factor structure:
#   direct-brief  short reading, single prompt   direct-full  full reading, single prompt
#   synthesized   full reading, facets and synthesis
#   direct-full - direct-brief = reading-depth difference; synthesized - direct-full = process difference
# The generation helper of 5.2 never cuts a prompt (the longest prompt of the full corpus has 5,304 tokens).
# Checkpoint: baseline_direct_matched.csv. About 5-6 hours for the full corpus on one A100 80 GB.
MATCHED_BUDGETS = {"background": 1500, "summary": 1500, "detail": 2000}
MATCHED_PATH = os.path.join(SAVE_DIR, "baseline_direct_matched.csv")
_SUM_COL = "summary_filled" if "summary_filled" in pipeline_df.columns else "summary"
_BATCH   = MATCHED_BATCH   # the prompts are longer; reduce it when the GPU memory is not sufficient

def _direct_full_prompt(row):
    return DIRECT_PROMPT.format(
        title      = str(row["title"]),
        abstract   = str(row["abstract"]),
        background = truncate_text_s2(row["background"],           MATCHED_BUDGETS["background"]),
        summary    = truncate_text_s2(row[_SUM_COL],               MATCHED_BUDGETS["summary"]),
        detail     = truncate_text_s2(row["detailed_description"], MATCHED_BUDGETS["detail"]),
    )

pids = pipeline_df["patent_id"].tolist()
_first = [_direct_full_prompt(pipeline_df.iloc[i]) for i in range(min(200, len(pipeline_df)))]
_ntok = [len(tokenizer_s2.encode(p, add_special_tokens=False)) for p in _first]
DIRECT_FULL_PROMPT_MEDIAN = int(np.median(_ntok))      # written to release_assets/config/generation.json (9.1)
print(f"Prompt length with the facet budgets (first {len(_first)} records): median {DIRECT_FULL_PROMPT_MEDIAN} tokens"
      f" | 90th percentile {int(np.percentile(_ntok, 90))} | max {int(max(_ntok))}")

if os.path.exists(MATCHED_PATH):
    done_df  = pd.read_csv(MATCHED_PATH).fillna("")
    done_ids = set(done_df["patent_id"])
    remain   = [i for i, p in enumerate(pids) if p not in done_ids]
    results  = done_df.to_dict("records")
    print(f"checkpoint: {len(done_ids)} done, {len(remain)} remaining")
else:
    remain, results = list(range(len(pids))), []
# prompts are built only for the records that still need a summary (tokenising long sections takes minutes)
prompts = {i: (_first[i] if i < len(_first) else _direct_full_prompt(pipeline_df.iloc[i]))
           for i in tqdm(remain, desc="build prompts (matched)")}

if remain and "model_s2" not in globals():
    model_s2 = AutoModelForCausalLM.from_pretrained(
        S2_MODEL_NAME, torch_dtype=GEN_DTYPE, trust_remote_code=True,
        device_map="auto" if HAS_CUDA else None)
    if not HAS_CUDA:
        model_s2 = model_s2.to(DEVICE)
    model_s2 = model_s2.eval()

with timed(f"5.11  direct-full generation ({S2_MODEL_NAME.split('/')[-1]}, facet budgets)",
           n_items=len(remain) or None) as t:
    t.note = f"{len(remain)} records generated in this run; batch={_BATCH}; budgets={MATCHED_BUDGETS}"
    pbar = tqdm(range(0, len(remain), _BATCH), desc="direct-full")
    for b in pbar:
        chunk = remain[b : b + _BATCH]
        outs = generate_batch_s2([prompts[i] for i in chunk], max_new_tokens=300)
        for i, out in zip(chunk, outs):
            results.append({"patent_id": pids[i], "baseline_matched_summary": out})
        pbar.set_description(f"direct-full ({len(results)}/{len(pids)})")
        if len(results) % 100 < _BATCH:
            pd.DataFrame(results).to_csv(MATCHED_PATH, index=False)
pd.DataFrame(results).to_csv(MATCHED_PATH, index=False)
_wl = pd.DataFrame(results)["baseline_matched_summary"].astype(str).str.split().str.len()
print(f"saved -> {MATCHED_PATH} ({len(results):,} records) | mean {_wl.mean():.0f} words"
      f" | median {_wl.median():.0f} | empty {(_wl == 0).sum()}")


# ── 5.12  Release the synthesis model before section 6 ─────────────────────
free_gpu("model_s2", "tokenizer_s2")
