"""Section 2: homonym screening (21,947 -> 16,434 records in the released run).

Run from the pipeline notebook with run_step("s2_homonym"); the parts below are executed in the namespace of the notebook.
"""

# ── 2.1  Load the crawler output and rename the section columns ──────────────
# dtype=str reads every column as text. Columns such as claim_count have missing values, and an inferred
# numeric type would turn 17 into 17.0; the screening does no arithmetic, so the output stays identical to
# the input file character by character.
from collections import Counter

# The section columns of the crawler output carry the heading on the Google Patents page (upper case with
# spaces). They are renamed here; every later cell and output file uses the new names.
RENAME_MAP = {
    "BACKGROUND": "background",
    "SUMMARY": "summary",
    "DETAILED DESCRIPTION": "detailed_description",
    "CROSS-REFERENCE TO RELATED APPLICATION": "cross-reference_to_related_application",
    "BRIEF DESCRIPTION OF THE DRAWINGS": "brief_description_of_drawings",
}
# The six text fields that are joined for the sense evidence: a "(HPC)" definition can be in any of them.
HOMONYM_TEXT_COLS = ["title", "abstract", "background", "summary", "detailed_description", "claims_text"]

print(f"input: {DATA_PATH} ({os.path.getsize(DATA_PATH)/1e9:.2f} GB)")
with timed("2.1  Load the crawler output"):
    df = pd.read_csv(DATA_PATH, dtype=str)
df = df.rename(columns=RENAME_MAP)
N_FILE = len(df)
if SAMPLE_N:
    # sample profile: a seeded random subset; the original row order is kept
    df = df.sample(min(SAMPLE_N, len(df)), random_state=RANDOM_STATE).sort_index().reset_index(drop=True)

# The decision steps add temporary columns to df; the output files keep only these original columns.
ORIG_COLS = df.columns.tolist()
N_RAW = len(df)
print(f"Records in the input file: {N_FILE:,}" + (f" | sample used in this run: {N_RAW:,}" if SAMPLE_N else ""))
print(f"Columns ({len(ORIG_COLS)}): {ORIG_COLS}")

# Records by application year
yr_df = (df["application_year"].value_counts().sort_index()
         .rename_axis("application_year").reset_index(name="records"))
yr_df


# ── 2.2  Coverage of the six text fields ────────────────────────────────────
# A field "has content" when it is not missing, not empty after stripping spaces and not the literal
# string "nan". pandas already reads the crawler placeholders "NA" and "N/A" as missing, so a section whose
# heading the parser did not recognise counts as missing here.
# Four tables: coverage per field, missing-content patterns, best available text, shortest abstracts.
present = {}                        # field -> bool Series (True = has content)
length  = {}                        # field -> characters (0 when missing)
for c in HOMONYM_TEXT_COLS:
    s = df[c].fillna("")
    ok = (s.str.strip() != "") & (s.str.lower() != "nan")
    present[c] = ok
    length[c]  = s.str.len().where(ok, 0)

# Table 1: coverage per field
rows = []
for c in HOMONYM_TEXT_COLS:
    ok, L = present[c], length[c]
    rows.append({
        "field": c,
        "has_content": int(ok.sum()),
        "coverage_percent": round(ok.mean() * 100, 1),
        "missing_content": int((~ok).sum()),
        # a title is short by nature, so the check for very short text does not apply to it
        "very_short_under_200_chars": "-" if c == "title" else int((ok & (L < 200)).sum()),
        "median_length": int(L[ok].median()) if ok.any() else 0,
    })
field_coverage = pd.DataFrame(rows)
field_coverage

# Figure 2.1: coverage of the six text fields
_qs = field_coverage.set_index("field")["coverage_percent"].sort_values()
fig, ax = plt.subplots(figsize=(6.4, 2.6))
_ = ax.barh(_qs.index, _qs.values, color=PALETTE[0], height=0.6)
for i, v in enumerate(_qs.values):
    _ = ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
_ = ax.set_xlim(0, 110)
_ = ax.set_xlabel("records with content (percent)")
_ = ax.set_title(f"Figure 2.1  Coverage of the six text fields ({N_RAW:,} records)")
fig.tight_layout()
_ = save_fig(fig, "fig_2_1_field_coverage")
plt.show()

# Table 2: missing-content patterns (one row per combination of present and missing fields)
pat = pd.DataFrame({c: present[c].map({True: "yes", False: "-"}) for c in HOMONYM_TEXT_COLS})
pat.columns = ["title", "abstract", "background", "summary", "detailed", "claims"]
combo = pat.value_counts().rename("records").reset_index()
combo["share_percent"] = (combo["records"] / len(df) * 100).round(1)
combo.head(12)

# Table 3: the most complete text that each record offers, in the order
# detailed_description > summary or background > claims > title and abstract only
best = pd.Series(np.select(
    [present["detailed_description"],
     present["summary"] | present["background"],
     present["claims_text"]],
    ["has detailed_description", "no detailed_description, has summary or background", "claims and title/abstract only"],
    default="title and abstract only"), index=df.index, name="best_available_text")
best_df = best.value_counts().rename_axis("best_available_text").reset_index(name="records")
best_df["share_percent"] = (best_df["records"] / len(df) * 100).round(1)
best_df

# Table 4: the five shortest non-empty abstracts (a visual check for junk values)
shortest = (df.loc[present["abstract"], ["patent_id", "abstract"]]
              .assign(n_chars=length["abstract"][present["abstract"]])
              .nsmallest(5, "n_chars"))
shortest["abstract"] = shortest["abstract"].str.slice(0, 100)
shortest.reset_index(drop=True)

deep4 = present["background"] | present["summary"] | present["detailed_description"] | present["claims_text"]
print(f"Records with none of background, summary, detailed_description and claims (title and abstract only): {int((~deep4).sum()):,}")
print(f"Records without title or abstract: {int((~(present['title'] & present['abstract'])).sum()):,}")


# ── 2.3  Sense rules: what HPC means in other domains ───────────────────────
# In different domains, HPC means at least:
#   high-performance computing (the meaning that we want)
#   hydroxypropyl cellulose   (excipient in drugs/cosmetics)
#   high pressure compressor  (jet engines/gas turbines)
#   hematopoietic progenitor cell (blood-forming progenitor cell) … and others
# These patents have no relation to high-performance computing; they are noise from the literal match of the query term.

# ---- (1) Detection of parenthetical definitions ----
# find "(hpc)" or "(hpcs)" in the full text; the full text is already in lower case.
# \s* allows spaces inside the parentheses, for example "( HPC )".
DEF_PAT = re.compile(r"\(\s*(hpcs?)\s*\)")

def sh_longform(text_before, sf="hpc", max_words=8):
    """Schwartz–Hearst abbreviation algorithm: find the long form of the abbreviation in the text before the parenthesis.

    Principle: each letter of the abbreviation occurs "in order" in the long form (not necessarily as a first letter).
    Method: take at most max_words words before the parenthesis and match the abbreviation characters one by one, from last to first:

        Example: '... comprising hydroxypropyl cellulose (hpc) ...'
            search 'c' from right to left → matches the c in cellulose
            search 'p' further left     → matches the p of propyl
            search 'h' further left     → matches the h of hydroxypropyl
            all characters match → go back from the last match to the start of that word,
            result: long form = 'hydroxypropyl cellulose'

    Parameters
        text_before : a short text before the parenthesis (the caller passes only the 120 characters before it; the definition is always next to the parenthesis)
        sf          : the abbreviation (lower case)
        max_words   : look back only 8 words; a longer look-back takes ordinary sentences as definitions

    Returns the long form string; returns None if the match fails or the length is not plausible.
    """
    cand = " ".join(text_before.split()[-max_words:])
    i = len(cand) - 1
    for ch in reversed(sf):                 # match the abbreviation characters from last to first
        while i >= 0 and cand[i] != ch:
            i -= 1
        if i < 0:                           # a character is not found → not a definition
            return None
        i -= 1
    start = cand.rfind(" ", 0, i + 1) + 1 if i > 0 else 0   # go back to the start of the word
    lf = cand[start:].strip(" ,;:.")
    # Schwartz–Hearst plausibility check: the long form must not have more words than
    # min(|abbreviation|+5, |abbreviation|×2); a longer one is usually a false match, so drop it
    if not lf or len(lf.split()) > min(len(sf) + 5, len(sf) * 2):
        return None
    return lf

# ---- (2)(3) Sense groups ----
# key = internal code; value = (display name in reports, computing sense or not, regex for the full-text phrase)
#
# Notes on the regex:
#   [\- ]      hyphen or space → matches both high-performance and high performance
#   comput\w*  suffixes such as computing / computer / computation
#   hydroxyl?  both spellings hydroxy and hydroxyl (? = the previous character is optional)
#   [ \-]?     optional space or hyphen → also matches the joined form hydroxypropylcellulose
SENSE_GROUPS = {
    "computing":  ("Computing: high-performance computing",                         True,
                   r"high[\- ]performance comput\w*|supercomput\w*|high[\- ]performance cluster"),
    "cellulose":  ("Chemistry/pharma: hydroxypropyl cellulose",                     False,
                   r"hydroxyl?[ \-]?propyl[ \-]?cellulose"),
    "compressor": ("Aerospace/mechanical: high pressure compressor",                False,
                   r"high[\- ]pressure compressor"),
    "progenitor": ("Biomedical: hematopoietic progenitor cell",                     False,
                   r"hematopoietic progenitor"),
    "counter":    ("Computing peripheral (not HPC): hardware performance counter",  False,
                   r"hardware performance counter"),
    "coherency":  ("Computing term (not HPC): highest point of coherency (cache)",  False,
                   r"highest point of coherency"),
    "concrete":   ("Construction: high performance concrete",                       False,
                   r"high[\- ]performance concrete"),
    "pincount":   ("Electronics (not HPC): high pin count",                         False,
                   r"high[\- ]pin count"),
}
# Note: the counter and coherency groups belong to the computing "domain"
# (hardware performance counter = CPU performance counter;
#  highest point of coherency = an IBM term for cache coherency in multiprocessors),
# but "HPC" in these texts does not mean high-performance computing → it is still another meaning and is removed.

def defgroup(lf):
    """Map a long form found by SH to a sense group.

    Pass 1: exact match with the phrase regex of each group.
    Pass 2: loose match with one keyword, for definitions with suffixes or other forms
            (example: 'high performance computer system' contains 'comput' → computing).
    No match in both passes → 'other' (rare or one-time definitions; a manual look at the list in section 3 is enough).
    """
    for g, (_, _, rx) in SENSE_GROUPS.items():
        if re.search(rx, lf):
            return g
    for g, kw in [("computing", "comput"), ("cellulose", "cellulose"),
                  ("compressor", "compressor"), ("progenitor", "progenitor"),
                  ("counter", "counter"), ("coherency", "coheren"),
                  ("concrete", "concrete")]:
        if kw in lf:
            return g
    return "other"


# ── 2.4  Evidence extraction (parenthetical definitions and domain phrases) ──
# 1. Join the six text fields of each record into one lower-case text (missing values count as empty).
# 2. Count the word "hpc" and the full phrase of each sense group with vectorised regular expressions.
# 3. Run the Schwartz-Hearst algorithm on every "(HPC)" to recover the long form that the authors wrote.
# The joined text is deleted afterwards to free memory.
with timed("2.4  Sense evidence extraction", n_items=len(df)):
    txt_parts = []
    for c in HOMONYM_TEXT_COLS:
        s = df[c].fillna("")
        s = s.mask((s.str.strip() == "") | (s.str.lower() == "nan"), "")
        txt_parts.append(s)
    txt = txt_parts[0].str.cat(txt_parts[1:], sep=" ").str.lower()
    del txt_parts

    # \b is a word boundary: only the separate words hpc and hpcs count, not strings such as "hpcl"
    df["hpc_mentions"] = txt.str.count(r"\bhpcs?\b")
    for g, (_, _, rx) in SENSE_GROUPS.items():
        df[f"n_{g}"] = txt.str.count(rx)

    # Only the 120 characters before the parenthesis go to sh_longform: a definition is always next to the
    # parenthesis, and 120 characters hold the max_words = 8 words that the algorithm looks back.
    defs = []
    for k, t in enumerate(txt):
        found = set()
        for m in DEF_PAT.finditer(t):
            lf = sh_longform(t[max(0, m.start() - 120):m.start()])
            if lf:
                found.add(lf)
        defs.append(";".join(sorted(found)))   # a record can define the abbreviation more than once; keep all
    df["sh_defs"] = defs

    # Section of the class code (first letter): G physics and computing, H electricity, A human necessities,
    # C chemistry, F mechanical engineering. Used only in the reports, never in the decision.
    df["ipc_section"] = df["main_ipc_4"].fillna("?").str[0]
    del txt
    _ = gc.collect()

# How many records each kind of evidence covers
evidence_summary = pd.DataFrame({
    "evidence": ['(1) "(HPC)" parenthetical definition', "(2) domain phrase (any group)", "word hpc occurs"],
    "records": [int((df["sh_defs"] != "").sum()),
                int((df[[f"n_{g}" for g in SENSE_GROUPS]].sum(axis=1) > 0).sum()),
                int((df["hpc_mentions"] > 0).sum())],
})
evidence_summary["share_percent"] = (evidence_summary["records"] / len(df) * 100).round(1)
evidence_summary


# ── 2.5  The "(HPC)" definitions that occur in the data ────────────────────
# Collect the SH long forms of all patents and count them. This table has two uses:
#   1. Direct evidence: which other meanings HPC has in this corpus, and how many patents have each
#   2. Check whether SENSE_GROUPS misses a sense (rows with group = other need a manual look)
inv = Counter()
for ds in df["sh_defs"]:
    if ds:
        inv.update(ds.split(";"))              # several definitions in one patent count once each
inv_df = (pd.DataFrame(inv.most_common(), columns=["long form (HPC)", "patents"])
          .assign(group=lambda d: d["long form (HPC)"].map(defgroup)))

n_withdef = int((df["sh_defs"] != "").sum())
print(f'Patents with an explicit "…(HPC)" parenthetical definition: {n_withdef:,} / {N_RAW:,} ({n_withdef/N_RAW*100:.1f}%)')
print(f"Distinct long forms: {len(inv_df)}; top 30 below")
inv_df


# ── 2.6  Sense of each record from three levels of evidence ─────────────────
# The three levels of evidence apply in order of priority:
#   (1) SH parenthetical definition present → the definition decides (the authors state what HPC means; most reliable)
#        - definitions in one patent map to several sense groups (very rare) → take the group with the most full-text phrases
#   (2) no definition, but sense phrases occur in the full text → take the group with the most occurrences
#   (3) no text evidence at all → always keep:
#        a removal needs "explicit text that shows the other meaning" (the parenthetical definition of (1) or the full phrase of (2));
#        without evidence, the IPC class is not used to infer it.
#        (missing evidence is mostly caused by incomplete full text in the CSV: the query term is in a section that was not parsed,
#          not because the patent does not mention HPC; there is no valid reason to remove such records.)
#
# Output columns:
#   sense        sense group code (unknown = no text evidence)
#   basis        decision basis ((1)/(2)/(3); for later spot checks or for the second-level validation)
#   sense_cat    display name in reports
#   is_hpc_sense True = keep (computing sense, or no evidence of another meaning) / False = other meaning (remove)
GLIST = list(SENSE_GROUPS)

def decide(row):
    # ---- (1) SH parenthetical definition ----
    dgs = {defgroup(x) for x in row["sh_defs"].split(";") if x} - {"other"}
    if len(dgs) == 1:
        return dgs.pop(), "(1) SH definition"
    if len(dgs) > 1:                       # one patent defines several senses (rare)
        return max(dgs, key=lambda g: row[f"n_{g}"]), "(1) SH definition (several senses: most phrases)"
    # ---- (2) Full-text phrase count ----
    counts = {g: row[f"n_{g}"] for g in GLIST}
    if max(counts.values()) > 0:
        return max(counts, key=counts.get), "(2) full-text phrase"
    # ---- (3) No text evidence → always keep (no inference from IPC) ----
    return "unknown", "(3) no text evidence (kept)"

with timed("2.6  Sense decision", n_items=len(df)):
    df[["sense", "basis"]] = df.apply(decide, axis=1, result_type="expand")

# sense code → (display name, keep or not); add the class "no evidence → keep" of (3)
CAT = {g: (c, h) for g, (c, h, _) in SENSE_GROUPS.items()}
CAT["unknown"] = ("No text evidence (cannot decide → keep)", True)
df["sense_cat"]    = df["sense"].map(lambda g: CAT[g][0])
df["is_hpc_sense"] = df["sense"].map(lambda g: CAT[g][1])

# ---- Report 1: counts per other meaning ----
print("=== Counts per other meaning ===")
homonym_tab = (df[~df.is_hpc_sense].groupby("sense_cat").size()
               .rename("patents").sort_values(ascending=False).to_frame())
homonym_tab["percent_of_all"] = (homonym_tab.patents / N_RAW * 100).round(1)
homonym_tab
n_unknown = int((df["sense"] == "unknown").sum())
print(f"Other meaning, total (explicit text evidence): {int((~df.is_hpc_sense).sum()):,} records"
      f" ({(~df.is_hpc_sense).mean()*100:.1f}%); "
      f"kept {int(df.is_hpc_sense.sum()):,} records"
      f" = computing sense {int(df.is_hpc_sense.sum()) - n_unknown:,} + no text evidence {n_unknown:,}")

# ---- Breakdown of the no-text-evidence group ((3))  ----
# "no evidence" is almost always caused by incomplete full text in the CSV: the word hpc is not in the available text at all
noev = df[df["sense"] == "unknown"]
noev_break = pd.DataFrame({
    "condition": ["the word hpc is not in the six text fields (query term in a section that was not parsed)",
             "hpc occurs, but no parenthetical definition and no known phrase"],
    "records": [int((noev["hpc_mentions"] == 0).sum()),
             int((noev["hpc_mentions"] > 0).sum())],
})
noev_break["percent_of_group3"] = (noev_break["records"] / max(len(noev), 1) * 100).round(1)
noev_break

# Class section of the kept records without evidence
noev_sec = (noev["ipc_section"].value_counts()
            .rename_axis("cpc_section").reset_index(name="records"))
noev_sec

# ---- Report 2: decision basis × computing or not ----
# shows how many decisions each level of evidence makes; the "(3) no text evidence (kept)" row has no clear basis for removal, so it is kept
print("\nDecision basis × computing sense or not:")
pd.crosstab(df["basis"], df["is_hpc_sense"], margins=True)

# Figure 2.2: records per sense class (blue = kept, orange = other sense, removed)
from matplotlib.patches import Patch
sc = df["sense_cat"].value_counts().sort_values()
ishpc_by_cat = {c: h for _, (c, h) in CAT.items()}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
_ = ax.barh(sc.index, sc.values, height=0.6,
            color=[PALETTE[0] if ishpc_by_cat.get(c, False) else PALETTE[1] for c in sc.index])
for i, v in enumerate(sc.values):
    _ = ax.text(v + sc.max() * 0.01, i, f"{v:,}", va="center", fontsize=8, color=INK_2)
_ = ax.set_xlim(0, sc.max() * 1.15)
_ = ax.set_xlabel("records")
_ = ax.legend(handles=[Patch(color=PALETTE[0], label="kept (computing sense or no evidence)"),
                       Patch(color=PALETTE[1], label="removed (other sense)")], loc="lower right")
_ = ax.set_title(f"Figure 2.2  Sense of the abbreviation HPC in {N_RAW:,} records")
fig.tight_layout()
_ = save_fig(fig, "fig_2_2_sense_classes")
plt.show()

# Three titles of each removed class, for a visual check of the decision
samples = (df.loc[~df.is_hpc_sense, ["sense_cat", "patent_id", "title", "basis"]]
             .groupby("sense_cat", group_keys=False).head(3)
             .sort_values("sense_cat").reset_index(drop=True))
samples


# ── 2.7  Remove the other senses and write the candidate file ───────────────
# The only removal condition is is_hpc_sense == False. If the removal is clean, the sections A (human
# necessities, including pharmaceuticals), C (chemistry) and F (mechanical engineering) shrink sharply.
keep_mask = df["is_hpc_sense"]
N_CLEAN = int(keep_mask.sum())
kept = df[keep_mask]

funnel = pd.DataFrame([
    {"stage": "records in the input",          "records": N_RAW,   "change": ""},
    {"stage": "removed: HPC has another sense", "records": N_RAW - N_CLEAN, "change": f"-{(N_RAW - N_CLEAN)/N_RAW*100:.1f}%"},
    {"stage": "candidates",                     "records": N_CLEAN, "change": f"kept {N_CLEAN/N_RAW*100:.1f}%"},
])
funnel

# Class section before and after the removal
sec_df = (pd.concat([df["ipc_section"].value_counts().rename("before_removal"),
                     kept["ipc_section"].value_counts().rename("after_removal")], axis=1)
          .fillna(0).astype(int).rename_axis("cpc_section").reset_index())
sec_df["removal_rate_percent"] = ((1 - sec_df["after_removal"] / sec_df["before_removal"]) * 100).round(1)
sec_df

# Figure 2.3: records per class section before and after the removal
_x = np.arange(len(sec_df))
fig, ax = plt.subplots(figsize=(7.2, 3.0))
_ = ax.bar(_x - 0.2, sec_df["before_removal"], width=0.38, color=OTHER_GREY, label=f"input ({N_RAW:,})")
_ = ax.bar(_x + 0.2, sec_df["after_removal"],  width=0.38, color=PALETTE[0], label=f"candidates ({N_CLEAN:,})")
_ = ax.set_xticks(_x, sec_df["cpc_section"])
_ = ax.set_xlabel("class section (first letter of the class code)")
_ = ax.set_ylabel("records")
_ = ax.legend(loc="upper left")
_ = ax.set_title("Figure 2.3  Class sections before and after the removal of the other senses")
fig.tight_layout()
_ = save_fig(fig, "fig_2_3_sections_before_after")
plt.show()

# Kept records outside sections G and H: computing evidence in another field (for example reservoir
# simulation or medical image computing), or no text evidence at all. This notebook does not filter by class.
nongh = (kept.loc[~kept["ipc_section"].isin(["G", "H"]), "ipc_section"]
         .value_counts().rename_axis("cpc_section").reset_index(name="records"))
print("Kept records outside sections G and H:")
nongh

# Only the original columns are written; the temporary decision columns stay in memory.
with timed("2.7  Write the candidate file", n_items=N_CLEAN):
    kept[ORIG_COLS].to_csv(CLEAN_PATH, index=False)
print(f"\nCandidates -> {CLEAN_PATH} ({N_CLEAN:,} records x {len(ORIG_COLS)} columns)")


# ── 2.8  Rules for the family and litigation marks ──────────────────────────
import ast

# ---- Family-root parsing ----
# in applications_claiming_priority, application_number is "application number + publication number" joined together,
# example: 'US15/235,004US10193762B2(en)' → the application number is only 'US15/235,004'
# Two rules split it:
#   PCT_PAT: catch PCT numbers (such as PCT/US2015/012345) first:
#            the middle part contains 'US' + digits, which would mislead the general rule and cut it to 'PCT/'
#   GEN_PAT: a general application number = 2–3 upper-case letters + digits/slashes/commas/hyphens,
#            and it stops at the next "two upper-case letters + digit" (= start of the publication number) or at '('
# Finally, remove commas and spaces to normalize ('US15/235,004' → 'US15/235004'),
# so that the same application number in a different format is not taken as a different family.
PCT_PAT = re.compile(r"^(PCT/[A-Z]{2}\d{4}/\d+)")
GEN_PAT = re.compile(r"^([A-Z]{2,3}[\d/,\-\. ]+?[A-Z]?)(?=[A-Z]{2}\d|\(|$)")

def clean_app_no(no):
    """Get the clean application number from the joined string and normalize it"""
    no = str(no).strip()
    m = PCT_PAT.match(no)
    if m:
        return m.group(1)
    m = GEN_PAT.match(no)
    s = m.group(1).strip() if m else no
    return s.replace(",", "").replace(" ", "")

def family_root(raw, own_id):
    """Return the family-root application number of this patent.

    applications_claiming_priority is a list[dict];
    a continuation lists the full chain of ancestors (itself included); each entry has application_number / filing_date.
    Rule: the application number of the entry with the earliest filing_date is the "family root";
          all members of a family give the same root, so grouping by the root gives the family.
    Parsing fails or the field is empty → the record's own patent_id is the root (singleton; not marked as a family).
    """
    try:
        apps = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip() not in ("", "nan") else []
    except Exception:
        apps = []
    entries = []
    for a in apps:
        no = clean_app_no(a.get("application_number", ""))
        fd = str(a.get("filing_date", "") or "")
        if no:
            entries.append((fd, no))
    if not entries:
        return own_id
    dated = [e for e in entries if e[0]]              # entries with a filing_date first
    return min(dated)[1] if dated else entries[0][1]  # no dates at all → take the first entry (ancestors come first)

# ---- PTAB proceeding event extraction ----
# legal_events is a string of a list[dict]; scan each {...} event;
# an event whose title contains one of the keywords below is a proceeding event; extract its date for the note.
# The order is important: check 'inter partes review certificate' before 'inter partes review';
# otherwise a certificate event is marked as an ordinary IPR.
PTAB_KINDS = [
    ("inter partes review certificate", "IPR certificate"),   # certificate at the end of an IPR (the source text says whether claims were cancelled)
    ("inter partes review",             "IPR"),       # challenge to patent validity (often with an infringement suit)
    ("post-grant review",               "PGR"),       # validity challenge within 9 months after grant
    ("reexamination",                   "reexamination"),     # ex parte reexamination
]

def ptab_note(ev):
    """Get the PTAB proceeding note from the legal_events string, such as 'IPR 2021-12-21; IPR certificate 2025-11-04'"""
    if not isinstance(ev, str) or not ev:
        return ""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", ev):          # take each event dict in turn (no eval, to avoid format traps)
        low = m.group(0).lower()
        for key, label in PTAB_KINDS:
            if key in low:
                dm = re.search(r"'date':\s*'(\d{4}-\d{2}-\d{2})'", m.group(0))
                out.append(f"{label} {dm.group(1)}" if dm else label)
                break                                  # one label per event
    return "; ".join(dict.fromkeys(out))               # remove duplicates, keep the order

# ---- District-court litigation match (built from the USPTO docket source file at run time; no fixed results) ----
# Source: patents.dta of the USPTO Patent Litigation Docket Reports:
#       patent litigation records of US district courts; each row = one case × one patent in suit;
#       three columns are used: case_number (case number), date_filed (filing date of the case), patent (patent number).
#       File: LITIGATION_FILE, patents.dta or patents.dta.zip (pandas reads the zip directly).
# Two cleaning steps; both are necessary:
#   1) Exclude values that contain "/": the patent column of the docket also has application "serial numbers" such as "11/082327";
#      without the non-digits this becomes 11082327, which collides with later patent numbers US11,0xx,xxx and gives false positives.
#   2) Make the join key "digits only, no leading zeros": the docket has values such as '5696571', 'RE38609',
#      and our patent_id is 'US10016403B2'; both sides must be reduced to the digit core to match.
# Note: the litigation status covers only the filing period of the data file (the load step prints the actual range); later cases are not found.
LITIG_DTA = LITIG_FILE                 # LITIGATION_FILE of the Parameters cell (patents.dta or patents.dta.zip)
assert os.path.exists(LITIG_DTA), f"Docket file not found: {LITIG_DTA}"
with timed("2.8  Read the litigation docket"):
    DOCKET = pd.read_stata(LITIG_DTA)
DOCKET["patent"] = DOCKET["patent"].astype(str)
DOCKET = DOCKET[~DOCKET["patent"].str.contains("/")]               # 1) exclude application serial numbers
DOCKET["_key"] = DOCKET["patent"].str.replace(r"\D", "", regex=True).str.lstrip("0")
DOCKET = (DOCKET[DOCKET["_key"] != ""]                             # 2) digit core as the join key
          .drop_duplicates(["_key", "case_number"])
          [["_key", "case_number", "date_filed"]])

def district_notes(pids):
    """For a Series of patent_id values, return the district-court litigation notes (no case → empty string); the index matches the input.

    Note format = "district court <n> cases (<filing years>): <first three case numbers>"; more than three case numbers end with "…".
    On the patent_id side, a regex takes the number after 'US'; do not keep all digits of the string:
    the 2 of the kind code (B2) would join the number, and then nothing matches.
    """
    keys = pids.str.extract(r"^US(\d+)", expand=False).str.lstrip("0")
    notes = {}
    for k, g in DOCKET[DOCKET["_key"].isin(keys)].groupby("_key"):
        cases = sorted(g["case_number"].unique())
        years = sorted(g["date_filed"].dt.year.unique())
        yr = str(years[0]) if years[0] == years[-1] else f"{years[0]}–{years[-1]}"
        notes[k] = (f"district court {len(cases)} cases ({yr}): "
                    + "; ".join(cases[:3]) + ("…" if len(cases) > 3 else ""))
    return keys.map(notes).fillna("")

print(f"Rules loaded: {os.path.basename(LITIG_DTA)} → docket {len(DOCKET):,} litigation rows, "
      f"{DOCKET['_key'].nunique():,} patents"
      f" (filed {DOCKET['date_filed'].min():%Y-%m} to {DOCKET['date_filed'].max():%Y-%m})")


# ── 2.9  Apply the family and litigation marks ──────────────────────────────
# Two columns are added as notes; no record is removed. The candidate file is then written again with them.
# In a new session this cell reads the candidate file of 2.7 (sections 1, 2.8 and this cell are enough);
# a rerun computes both columns again and overwrites them.
if "kept" in globals() and isinstance(kept, pd.DataFrame):
    ann = kept.copy()
    base_cols = ORIG_COLS
else:
    ann = pd.read_csv(CLEAN_PATH, dtype=str)
    base_cols = [c for c in ann.columns if c not in ("family", "litigation")]
print(f"Records to mark: {len(ann):,}")

# ---- family column ----
# family root → number of records with the same root in "this corpus"; mark only if ≥2 (singletons stay empty).
# Note: n is the family size in the cleaned corpus, not the worldwide family size.
ann["_root"] = [family_root(r, p) for r, p in
                zip(ann["applications_claiming_priority"], ann["patent_id"])]
fam_size = ann.groupby("_root")["patent_id"].transform("size")
ann["family"] = np.where(fam_size >= 2,
                         ann["_root"] + " (n=" + fam_size.astype(str) + ")",
                         "")

n_fam_pat = int((ann["family"] != "").sum())
n_fam_grp = ann.loc[ann["family"] != "", "_root"].nunique()

# family mark statistics
fam_stats = pd.DataFrame([
    {"item": "families (≥2 records in this corpus)", "value": f"{n_fam_grp:,}"},
    {"item": "patents with a family mark",           "value": f"{n_fam_pat:,}"},
    {"item": "singletons (family empty)",            "value": f"{int((ann['family']=='').sum()):,}"},
])
fam_stats

# ---- litigation column ----
# merge the two sources: district-court docket match + PTAB events in legal_events, joined with " | "
dc   = district_notes(ann["patent_id"])
ptab = ann["legal_events"].map(ptab_note)
ann["litigation"] = [" | ".join(x for x in (a, b) if x) for a, b in zip(dc, ptab)]

n_dc, n_pt = int((dc != "").sum()), int((ptab != "").sum())
n_lit = int((ann["litigation"] != "").sum())

# litigation mark statistics
lit_stats = pd.DataFrame([
    {"source": "district court (docket lookup)",   "records": n_dc},
    {"source": "PTAB (IPR / PGR / reexamination)", "records": n_pt},
    {"source": "both",                             "records": n_dc + n_pt - n_lit},
    {"source": "litigation column marked (total)", "records": n_lit},
])
lit_stats

# ---- Sample view ----
print("\nSample of family marks (largest family):")
big = ann[ann["family"] != ""].copy()
big["_n"] = fam_size[big.index]
big.nlargest(5, "_n")[["patent_id", "title", "family"]]
print("All litigation marks:")
(ann.loc[ann["litigation"] != "", ["patent_id", "title", "litigation"]]
 .sort_values("patent_id").reset_index(drop=True))

# Write the candidate file again: original columns + family + litigation (helper columns such as _root are dropped)
with timed("2.9  Write the candidate file with the marks", n_items=len(ann)):
    ann[base_cols + ["family", "litigation"]].to_csv(CLEAN_PATH, index=False)
print(f"\nCandidates with marks -> {CLEAN_PATH} ({len(ann):,} records x {len(base_cols) + 2} columns)")


# ── 2.10  Derived files: one record per family, and the litigated families ──
# Two derived files on the candidate level; neither is used by the later sections, and the candidate file is not changed.
#   HPC_15to25_clean_dedup.csv        one record per family (the earliest application), for analyses that count
#                                     each invention once
#   HPC_15to25_litigation_family.csv  every family with at least one litigated member, with case groups
if "ann" in globals():
    src_df   = ann
    out_cols = base_cols + ["family", "litigation"]
else:
    src_df = pd.read_csv(CLEAN_PATH, dtype=str)
    src_df[["family", "litigation"]] = src_df[["family", "litigation"]].fillna("")
    out_cols = [c for c in src_df.columns]
print(f"Source: {len(src_df):,} records × {len(out_cols)} columns")

# ============================================================
# Output A: family-deduplicated version
# ------------------------------------------------------------
# Group key: records with a family mark use family (= family root application number);
#         singletons (empty family) use their own patent_id: each singleton is its own group.
# Each group keeps the record with the "earliest application date" (on the same date, sort by patent_id for a stable result).
# Reason: continuations share the same description, so their embeddings are almost the same;
#       keeping the earliest record = keeping the first disclosure of the invention.
# ============================================================
grp_key = src_df["family"].where(src_df["family"] != "", src_df["patent_id"])
dedup = (src_df.assign(_grp=grp_key)
               .sort_values(["application_date", "patent_id"])   # earliest application first
               .drop_duplicates("_grp", keep="first"))

DEDUP_CSV = os.path.join(SAVE_DIR, "HPC_15to25_clean_dedup.csv")
dedup[out_cols].to_csv(DEDUP_CSV, index=False)

n_fam_grp2  = int((src_df["family"] != "").sum())                 # total number of family members
n_fam_root2 = src_df.loc[src_df["family"] != "", "family"].nunique()
dedup_stats = pd.DataFrame([
    {"item": "before deduplication",                            "n_rows": f"{len(src_df):,}"},
    {"item": "family members (reduced to 1 record per family)", "n_rows": f"{n_fam_grp2:,} → {n_fam_root2:,}"},
    {"item": "singletons (kept as is)",                         "n_rows": f"{int((src_df['family']=='').sum()):,}"},
    {"item": "after deduplication (output)",                    "n_rows": f"{len(dedup):,}"},
])
dedup_stats
print(f"Family-deduplicated version → {DEDUP_CSV}\n")

# ============================================================
# Output B: litigated-family subset
# ------------------------------------------------------------
# Selection rule: the record has a litigation mark, or any member of its family is litigated.
# Reason: litigation is about the "invention": continuations in a family share the same description and priority;
#       if any member is asserted or challenged, the whole family is relevant for litigation analysis.
# ============================================================
lit_mask = src_df["litigation"] != ""
lit_fams = set(src_df.loc[lit_mask & (src_df["family"] != ""), "family"])
sel      = lit_mask | src_df["family"].isin(lit_fams)
litfam   = src_df[sel].sort_values(["family", "application_date"])

# ============================================================
# Same-case mark (case_group) + shared-family mark (family_root)
# ------------------------------------------------------------
# Basis of case_group = case_number in the docket:
#   each docket row = one case × one patent in suit;
#   patents listed under the same case number = asserted together in the same lawsuit → same case group.
#   Transitive merge across cases (union-find): A and B in case 1, B and C in case 2
#   → A/B/C form one "case group" (one litigation campaign; common for serial NPE suits and multiple ANDA cases).
#   Group numbers L01, L02, … (groups with more members come first).
# PTAB proceedings (IPR/PGR/reexamination) have no district-court case number → the record's own case_group stays empty;
#   if a family member is in a district-court case, the record inherits that family's case group; pure PTAB families are grouped by family_root.
# family_root = shared family root application number (the family column without the " (n=N)" suffix; singleton = own patent_id)
# ============================================================
if "DOCKET" not in globals():        # load it here when this cell runs alone (same cleaning rules as 2.8)
    _p = LITIG_FILE
    assert os.path.exists(_p), f"Docket file not found: {_p}"
    DOCKET = pd.read_stata(_p)
    DOCKET["patent"] = DOCKET["patent"].astype(str)
    DOCKET = DOCKET[~DOCKET["patent"].str.contains("/")]
    DOCKET["_key"] = DOCKET["patent"].str.replace(r"\D", "", regex=True).str.lstrip("0")
    DOCKET = (DOCKET[DOCKET["_key"] != ""].drop_duplicates(["_key", "case_number"])
              [["_key", "case_number", "date_filed"]])

_keys = litfam["patent_id"].str.extract(r"^US(\d+)", expand=False).str.lstrip("0")
_sub  = DOCKET[DOCKET["_key"].isin(set(_keys.dropna()))]

# --- union-find: connect patents into case groups, with "same case number" as the edge ---
_parent = {}
def _find(x):
    while _parent.setdefault(x, x) != x:
        _parent[x] = _parent[_parent[x]]   # path halving, to avoid long chains
        x = _parent[x]
    return x
def _union(a, b):
    _parent[_find(a)] = _find(b)
for _, _grp in _sub.groupby("case_number"):
    _ks = _grp["_key"].tolist()
    for _k in _ks[1:]:
        _ = _union(_ks[0], _k)                 # connect all patents with the same case number

# --- Group numbers: groups with more members first (L01, L02, …) ---
_comp = {}
for _k in _sub["_key"].unique():
    _comp.setdefault(_find(_k), []).append(_k)
_order = sorted(_comp.items(), key=lambda kv: (-len(kv[1]), kv[0]))
_label = {k: f"L{i+1:02d}" for i, (_, ks) in enumerate(_order) for k in ks}

_case_list = _sub.groupby("_key")["case_number"].apply(lambda s: "; ".join(sorted(set(s))))
litfam["case_numbers"] = _keys.map(_case_list).fillna("")   # all case numbers that list this record (not cut)
litfam["case_group"]   = _keys.map(_label).fillna("")       # case group of this record

# --- Family members that are not in a district-court case inherit the case group of the litigated family members ---
_grpkey = litfam["family"].where(litfam["family"] != "", litfam["patent_id"])
_has = litfam["case_group"] != ""
_fam2grp = (litfam.loc[_has].assign(_g=_grpkey[_has])
            .groupby("_g")["case_group"]
            .agg(lambda s: "; ".join(sorted(set(s)))))
litfam["case_group"] = litfam["case_group"].where(_has, _grpkey.map(_fam2grp).fillna(""))

# --- Shared-family mark ---
litfam["family_root"] = (litfam["family"].str.split(" (", regex=False).str[0]
                         .where(litfam["family"] != "", litfam["patent_id"]))

# --- Case group summary ---
# the case_group of an inherited family member can be a combined value (a family in several case groups, such as the two Sysmex waves of suits),
# so split it (explode) before the summary; each group counts as one row
_cg = pd.DataFrame()
if (litfam["case_group"] != "").any():
    _cg = (litfam[litfam["case_group"] != ""]
           .assign(case_group=lambda d: d["case_group"].str.split("; "))
           .explode("case_group")
           .groupby("case_group")
           .agg(n_patents=("patent_id", "nunique"),
                n_case_numbers=("case_numbers", lambda s: len({c for x in s if x for c in x.split("; ")})),
                family_summary=("family_root", lambda s: "; ".join(sorted(set(s))[:3])))
           .sort_values("n_patents", ascending=False))
print("Case group summary (same group = same litigation campaign):")
_cg.head(15)

LITFAM_CSV = os.path.join(SAVE_DIR, "HPC_15to25_litigation_family.csv")
litfam[out_cols + ["family_root", "case_numbers", "case_group"]].to_csv(LITFAM_CSV, index=False)

litfam_stats = pd.DataFrame([
    {"item": "litigated patents (litigation not empty)",      "n_rows": int(lit_mask.sum())},
    {"item": "families of the litigated patents in a family", "n_rows": len(lit_fams)},
    {"item": "non-litigated members added by the family",     "n_rows": int((sel & ~lit_mask).sum())},
    {"item": "litigated-family subset (output)",              "n_rows": int(sel.sum())},
])
litfam_stats

# family-level summary table: size and litigation summary of each litigated family
roll = (litfam.assign(litigation_summary=lit_mask[litfam.index])
        .groupby(litfam["family"].where(litfam["family"] != "", litfam["patent_id"]))
        .agg(n_members=("patent_id", "size"),
             litigation_members=("litigation_summary", "sum"),
             earliest_filing=("application_date", "min"),
             litigation_note=("litigation", lambda s: next((x for x in s if x), "")[:60]),
             case_group=("case_group", lambda s: "; ".join(sorted({g for x in s if x
                                                             for g in x.split("; ")}))))
        .sort_values(["litigation_members", "n_members"], ascending=False))
roll.index.name = "family_root_or_patent_id"
roll
print(f"Litigated-family subset → {LITFAM_CSV}")


# ── 2.11  How a case group and a shared family are decided (two examples) ──
# Case group (case_group)
#   Basis = case_number in the USPTO docket. Each docket row = one case × one patent in suit,
#   so "patents listed under the same case number" were asserted together in the same lawsuit → same case group.
#   Transitive across cases: patents A and B in case 1, B and C in case 2 → A/B/C form one case group
#   (union-find connected components; serial suits of one plaintiff against several defendants are merged correctly).
#   PTAB (IPR/PGR/reexamination) has no district-court case number → not part of the merge; the shared family does the grouping.
# Shared family (family_root)
#   Basis = the priority chain in applications_claiming_priority. Under 37 CFR 1.78, a continuation
#   must state its parent application in the description and claim the same priority date, so the chain of each member contains
#   the same "earliest application"; it is the family root, and the same root = shared family (≈ simple family).
# Needs 2.8 (family_root) and 2.10 (_sub, _keys, litfam).
rowsA, evidence_b_family_root = None, None
if len(_sub):
    # Evidence A: the case that lists the most records of this corpus, with its docket rows
    top_case = _sub.groupby("case_number")["_key"].nunique().idxmax()
    rowsA = (_sub[_sub["case_number"] == top_case]
             .merge(litfam[["patent_id", "title"]].assign(_key=_keys.values), on="_key"))
    rowsA["date_filed"] = pd.to_datetime(rowsA["date_filed"]).dt.date
    rowsA["title"] = rowsA["title"].str.slice(0, 60)
    rowsA = rowsA[["case_number", "date_filed", "patent_id", "title"]].reset_index(drop=True)
    print(f"Evidence A | records listed under the same case number {top_case} (one case group):")
else:
    print("Evidence A | no record of this run is in the docket.")
rowsA

# Evidence B: the litigated family with the most members; the root is parsed for each member separately
_fams = litfam.loc[litfam["family"] != "", "family"]
if len(_fams):
    top_fam = _fams.value_counts().idxmax()
    memb = litfam[litfam["family"] == top_fam].copy()
    memb["family_root_parsed"] = [family_root(r, p) for r, p in
                                  zip(memb["applications_claiming_priority"], memb["patent_id"])]
    evidence_b_family_root = (memb[["patent_id", "application_date", "family_root_parsed", "litigation"]]
                              .sort_values("application_date").reset_index(drop=True))
    print(f"Evidence B | family {top_fam}: every member, parsed separately, gives the same root application number:")
else:
    print("Evidence B | no litigated family with two or more members in this run.")
evidence_b_family_root


# ── 2.12  List of the removed records with the reason for each ─────────────
# Patent number and decision columns only, no full text, so that users can check each removal or decide again with their own rule.
REMOVED_CSV = os.path.join(PROV_DIR, "removed_homonym.csv")
removed = df.loc[~df["is_hpc_sense"]].copy()

# the display name is "domain: English name"; the English name after the colon is the sense_label
SENSE_LABEL_EN = {g: v[0].split(": ", 1)[-1] for g, v in SENSE_GROUPS.items()}

def basis_code(b):
    if b.startswith("(1)"):
        return "definition_multi" if "several senses" in b else "definition"
    if b.startswith("(2)"):
        return "phrase"
    return "none"

removed_out = pd.DataFrame({
    "patent_id":        removed["patent_id"],
    "sense_group":      removed["sense"],
    "sense_label":      removed["sense"].map(SENSE_LABEL_EN),
    "evidence_basis":   removed["basis"].map(basis_code),
    "sh_long_forms":    removed["sh_defs"],
    "phrase_count":     removed.apply(lambda r: int(r[f"n_{r['sense']}"]), axis=1),
    "hpc_mentions":     removed["hpc_mentions"].astype(int),
    "ipc_section":      removed["ipc_section"],
    "main_ipc_4":       removed["main_ipc_4"],
    "application_year": removed["application_year"],
    "title":            removed["title"],
}).sort_values(["sense_group", "patent_id"]).reset_index(drop=True)
removed_out.to_csv(REMOVED_CSV, index=False)
print(f"List of homonym removals → {REMOVED_CSV} ({len(removed_out):,} records × {removed_out.shape[1]} columns)")
print(f"Check: removed {len(removed_out):,} + kept {N_CLEAN:,} = {len(removed_out) + N_CLEAN:,}; raw {N_RAW:,}")

# Sense group x decision basis
removed_tab = pd.crosstab(removed_out["sense_label"], removed_out["evidence_basis"], margins=True, margins_name="total")
removed_tab


# ── 2.13  Candidate identifiers and the hand-off to section 3 ──────────────
# Writes the identifiers of the candidates (no full text) and counts the records with all five required
# fields, the rule that section 3.3 applies. When an earlier run is reused, its released identifiers must
# all be in this five-field subset.
CANDIDATE_IDS_CSV = os.path.join(PROV_DIR, "candidate_ids.csv")
cand_ids = kept["patent_id"]
cand_ids.to_frame().to_csv(CANDIDATE_IDS_CSV, index=False)
print(f"Candidate identifiers -> {CANDIDATE_IDS_CSV} ({len(cand_ids):,} records)")

REQUIRED_5 = ["title", "abstract", "background", "detailed_description", "claims_text"]
complete_mask = np.logical_and.reduce([kept[c].fillna("").str.strip() != "" for c in REQUIRED_5])
complete_set  = set(kept.loc[complete_mask, "patent_id"])
print(f"Candidates with all five required fields: {len(complete_set):,}; without: {len(cand_ids) - len(complete_set):,}")

handoff_df = None
_rel_path = os.path.join(SAVE_DIR, "reused_released_corpus_ids.csv")
if os.path.exists(_rel_path):
    _rel = set(pd.read_csv(_rel_path, dtype=str)["patent_id"])
    handoff_df = pd.DataFrame([{
        "check": "released identifiers of the reused run", "records": len(_rel),
        "reference": "five-field subset of this run", "reference_records": len(complete_set),
        "only_in_reused": len(_rel - complete_set), "result": "subset" if _rel <= complete_set else "NOT a subset"}])
handoff_df

# Free the memory of the screening step: section 3 reads the candidate file again
for _n in ("txt", "pat", "combo", "present", "length", "best", "removed", "src_df", "dedup", "ann", "kept"):
    _ = globals().pop(_n, None)
del df
_ = gc.collect()
