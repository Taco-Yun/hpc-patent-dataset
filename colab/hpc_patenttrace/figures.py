"""Figures 1 and 2 of the data descriptor, drawn from the output folder of the pipeline notebook.

Every number in the figures is read from the files of that folder; nothing is typed in by hand. Each function
takes the folder opened with open_run(), returns the matplotlib figure and, when out_dir is given, also saves
it there as a 300 dpi PNG.
"""
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

INK, MUTE, ACC, LINE = "#1F2733", "#6B7584", "#24589E", "#9AA6B6"
OTHER_GREY = "#d3d2cc"
NOISE_LABELS = ["__noise__", "Off-topic / Non-HPC", "Unclassified", ""]
# The font list is set on the text itself, so that the figure looks the same when the notebook draws it again.
# Only installed fonts are listed (Google Colab has none of the first three; DejaVu Sans comes with matplotlib).
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONTS = [f for f in ("Helvetica", "Arial", "Liberation Sans") if f in _INSTALLED] + ["DejaVu Sans"]
STYLE = {"font.family": FONTS, "font.size": 8.5}


_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _word(n):
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def _save(fig, out_dir, name):
    """Save the figure (if out_dir is given) and close it; the notebook shows the returned figure once."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def construction_counts(run):
    """Records at each step of the corpus construction (table of section 3.10 of the pipeline)."""
    f = run.csv("release/tables/funnel_from_crawler_output.csv")   # crawler output, candidates, complete, released
    query, candidates, complete, released = (int(v) for v in f["records"])
    assert released == len(run.csv("release/released_corpus_ids.csv", dtype=str)), "funnel and released_corpus_ids.csv disagree"
    return {"query": query, "homonym_removed": query - candidates, "candidates": candidates,
            "incomplete_removed": candidates - complete, "complete": complete, "released": released,
            "excluded": complete - released}


def fig1_construction(run, out_dir=None):
    """Fig. 1: records at each construction step (a) and the data layers of the released records (b)."""
    c = construction_counts(run)
    clu = run.csv("release/cluster_results.csv", usecols=["cluster_id"])
    topics = run.csv("release/topic_summary.csv")
    emb = run.csv("release/embeddings/MANIFEST.csv")
    own = run.csv("release/ownership_events.csv", usecols=["conveyance_type"])
    n_clusters = int(clu.loc[clu["cluster_id"] != -1, "cluster_id"].nunique())
    n_topics = int((topics["topic_id"] != -1).sum())
    n_sets, dim = len(emb), int(emb["dim"].iloc[0])
    n_categories = int(own["conveyance_type"].nunique())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

        titles = []

        def box(x, y, w, h, title, sub, fc="#FFFFFF", ec=ACC):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
            titles.append((ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color=INK,
                                   fontweight="bold", fontsize=8), w))
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", color=MUTE, fontsize=7.2, linespacing=1.15)

        stages = [("Query result", f"{c['query']:,} records", "Google Patents, US granted,\nfiled 2015–2025"),
                  ("Homonym screening", f"{c['candidates']:,} records", f"{c['homonym_removed']:,} removed:\nother meanings of HPC"),
                  ("Completeness filter", f"{c['complete']:,} records", f"{c['incomplete_removed']:,} removed:\na required field is empty"),
                  ("Inclusion rule", f"{c['released']:,} records", f"{c['excluded']:,} excluded:\nno expanded phrase")]
        w, h, y0, gap = 20.5, 9, 41, 5.5
        xs = [1 + k * (w + gap) for k in range(4)]
        for k, (t, num, note) in enumerate(stages):
            box(xs[k], y0, w, h, t, num, fc="#EEF3FA" if k == 3 else "#FFFFFF")
            ax.text(xs[k] + w / 2, y0 - 2.2, note, ha="center", va="top", color=MUTE, fontsize=6.9, linespacing=1.15)
            if k < 3:
                ax.annotate("", xy=(xs[k + 1] - 0.6, y0 + h / 2), xytext=(xs[k] + w + 0.6, y0 + h / 2),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=9))
        ax.text(1, 57.5, "a", fontweight="bold", color=INK, fontsize=10)
        ax.text(4, 57.5, "Corpus boundary", color=INK, fontsize=8.5)
        ax.text(1, 28.5, "b", fontweight="bold", color=INK, fontsize=10)
        ax.text(4, 28.5, f"Layers for each of the {c['released']:,} released records, linked by patent_id", color=INK, fontsize=8.5)
        layers = [("Source fields", "parsed text, metadata,\nclassification"),
                  ("Section facets", "why, what, how\n(Llama-3.1-8B-Instruct)"),
                  ("Summary layers", "direct-brief, direct-full,\nsynthesized (Qwen2.5-14B)"),
                  ("Embeddings", f"{_word(n_sets)} sets, {dim} dimensions\n(SPECTER2)"),
                  ("Topic index", f"{n_clusters} clusters merged\ninto {n_topics} topics"),
                  ("Assignment evidence", f"legal events in\n{_word(n_categories)} categories")]
        lw, lh = 31, 9.5
        for k, (t, sub) in enumerate(layers):
            r, col = divmod(k, 3)
            box(1 + col * (lw + 2.5), 14.5 - r * (lh + 3), lw, lh, t, sub, fc="#F7F8FA", ec=LINE)
        # a wider fallback font can make a title longer than its box; such a title is set smaller
        fig.canvas.draw()
        to_data = ax.transData.inverted()
        for text, width in titles:
            bb = text.get_window_extent()
            span = to_data.transform((bb.x1, 0))[0] - to_data.transform((bb.x0, 0))[0]
            if span > width * 0.94:
                text.set_fontsize(8 * width * 0.94 / span)
    return _save(fig, out_dir, "Fig1_construction.png")


def fig2_two_views(run, out_dir=None, top_classes=10, top_topics=12):
    """Fig. 2: the two-dimensional UMAP map of the synthesized embeddings, coloured by class (a) and topic (b)."""
    clu = run.csv("release/cluster_results.csv", dtype=str, keep_default_na=False,
                  usecols=["patent_id", "main_ipc_3", "merged_topic"])
    xy = run.csv("umap2d_coords.csv", dtype={"patent_id": str})     # section 7.7 of the pipeline
    plot = clu.merge(xy, on="patent_id", how="inner").reset_index(drop=True)
    cmap = [plt.get_cmap("tab20")(i) for i in range(20)]
    ipc3, topic = plot["main_ipc_3"].astype(str), plot["merged_topic"].astype(str)
    n_ipc3, n_top = ipc3.nunique(), topic[~topic.isin(NOISE_LABELS)].nunique()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for ax, lab, top_n, title, other in (
                (axes[0], ipc3, top_classes, "a  Three-character CPC class", "other classes"),
                (axes[1], topic, top_topics, "b  Merged topic", "other or no topic")):
            lab = lab.where(~lab.isin(NOISE_LABELS), "")
            top = lab[lab != ""].value_counts().head(top_n).index.tolist()
            grey = ~lab.isin(top)
            ax.scatter(plot.loc[grey, "umap_2d_x"], plot.loc[grey, "umap_2d_y"], s=0.6, c=[OTHER_GREY],
                       linewidths=0, rasterized=True)
            handles = []
            for k, t in enumerate(top):
                sel = (lab == t).values
                ax.scatter(plot.loc[sel, "umap_2d_x"], plot.loc[sel, "umap_2d_y"], s=0.8, c=[cmap[k % 20]],
                           linewidths=0, rasterized=True)
                handles.append(Line2D([0], [0], marker="o", ls="", ms=3, color=cmap[k % 20],
                                      label=t if len(t) <= 44 else t[:43] + "…"))
            handles.append(Line2D([0], [0], marker="o", ls="", ms=3, color=OTHER_GREY, label=other))
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4)
            ax.legend(handles=handles, fontsize=4.6, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                      frameon=False, handletextpad=0.2, columnspacing=0.8)
        fig.suptitle(f"UMAP of the synthesized embeddings ({len(plot):,} records)", fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.n_classes, fig.n_topics = n_ipc3, n_top
    return _save(fig, out_dir, "Fig2_two_views.png")

