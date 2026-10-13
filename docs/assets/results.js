// Charts and the record viewer of the HPC-PatentTrace results pages.
// Data: window.RESULTS (assets/data.js, written by scripts/build_results_page.py from release/).
// Text: window.PAGE (set in each page, English or Traditional Chinese).
(function () {
  "use strict";
  const R = window.RESULTS;
  const P = window.PAGE;
  const T = P.t;
  const FONT = 'system-ui, -apple-system, "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif';
  const LAYERS = ["synthesized", "direct-brief", "direct-full"];
  const REPS = ["abstract", "fulltext-truncated", "direct-brief", "direct-full", "synthesized"];
  const nf = new Intl.NumberFormat(P.lang === "en" ? "en-US" : "zh-TW");

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const series = (i) => css("--s" + (i + 1));
  const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);

  function merge(a, b) {
    const out = Object.assign({}, a);
    for (const k of Object.keys(b)) out[k] = isObj(a[k]) && isObj(b[k]) ? merge(a[k], b[k]) : b[k];
    return out;
  }

  function layout(extra) {
    const axis = {
      gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, automargin: true,
      tickfont: { color: css("--ink-2") }, title: { font: { color: css("--ink-2") } },
    };
    return merge({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: FONT, size: 13, color: css("--ink-2") },
      margin: { l: 60, r: 24, t: 20, b: 52 },
      xaxis: axis, yaxis: axis, showlegend: false, bargap: 0.35, barcornerradius: 3,
      legend: { orientation: "h", x: 0, y: 1.14, font: { color: css("--ink-2") } },
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { family: FONT, color: css("--ink") } },
    }, extra || {});
  }

  const CONFIG = { displaylogo: false, responsive: true, displayModeBar: false };
  const draw = (id, traces, lay, config) => Plotly.react(id, traces, layout(lay), config || CONFIG);

  // ── corpus ──
  function funnel() {
    const x = R.funnel;
    draw("chart-funnel", [{
      type: "bar", orientation: "h", x: x, y: T.funnel, marker: { color: css("--s1") },
      text: x.map((v) => nf.format(v)), textposition: "outside", cliponaxis: false,
      textfont: { color: css("--ink") }, hovertemplate: "%{y}: %{x:,}<extra></extra>",
    }], {
      yaxis: { autorange: "reversed", showgrid: false },
      xaxis: { title: { text: T.records }, range: [0, Math.max(...x) * 1.18], tickformat: "," },
      margin: { l: 20 },
    });
  }

  function removal() {
    const r = R.removal;
    const tr = (key, name, color, base) => ({
      type: "bar", name: name, x: r.sections, y: r[key], marker: { color: color },
      customdata: r[base],
      hovertemplate: name + "<br>%{x}: %{y:.1f}% (" + T.records + " %{customdata:,})<extra></extra>",
    });
    draw("chart-removal", [
      tr("completeness", T.completeness, series(0), "candidates"),
      tr("inclusion", T.inclusion, series(1), "complete"),
    ], {
      barmode: "group", showlegend: true,
      yaxis: { title: { text: T.percent_removed }, range: [0, 100], ticksuffix: "" },
      xaxis: { title: { text: T.section } },
    });
  }

  // ── composition ──
  function sections() {
    const s = R.sections;
    draw("chart-sections", [{
      type: "bar", x: s.map((d) => d.s), y: s.map((d) => d.n), marker: { color: css("--s1") },
      text: s.map((d) => d.share.toFixed(1) + "%"), textposition: "outside", cliponaxis: false,
      textfont: { color: css("--ink") }, hovertemplate: "%{x}: %{y:,}<extra></extra>",
    }], { xaxis: { title: { text: T.section } }, yaxis: { title: { text: T.records }, tickformat: "," } });
  }

  function years() {
    const y = R.years;
    const first = y.find((d) => d.trunc);
    const x0 = first ? first.y - 0.5 : null;
    const last = y[y.length - 1].y;
    draw("chart-years", [{
      type: "bar", x: y.map((d) => d.y), y: y.map((d) => d.n),
      marker: { color: y.map((d) => (d.trunc ? css("--seq-light") : css("--s1"))) },
      hovertemplate: "%{x}: %{y:,}<extra></extra>",
    }], {
      // a fixed range: Plotly would otherwise widen the axis to fit the annotation text
      xaxis: { title: { text: T.year }, dtick: 2, range: [y[0].y - 0.6, last + 0.6], showgrid: false },
      yaxis: { title: { text: T.records }, tickformat: "," },
      shapes: x0 === null ? [] : [{ type: "line", xref: "x", yref: "paper", x0: x0, x1: x0, y0: 0, y1: 1,
                                    line: { color: css("--ink-2"), width: 1, dash: "dot" } }],
      annotations: x0 === null ? [] : [{
        x: last + 0.6, y: 1, xref: "x", yref: "paper", text: T.truncated, showarrow: false,
        xanchor: "right", yanchor: "bottom", font: { size: 12, color: css("--ink-2") },
      }],
      margin: { t: 34 },
    });
  }

  // ── summaries ──
  function ppl() {
    draw("chart-ppl", LAYERS.map((l, i) => ({
      type: "box", name: l, y: R.ppl[l], marker: { color: series(i) }, line: { width: 1.5 },
      fillcolor: "rgba(0,0,0,0)", boxmean: true, boxpoints: false,
      hovertemplate: l + ": %{y:.1f}<extra></extra>",
    })), { yaxis: { title: { text: T.ppl } } });
  }

  function examples() {
    const sel = document.getElementById("example-select");
    const body = document.getElementById("example-body");
    R.examples.forEach((e, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = e.id + " — " + e.title;
      sel.appendChild(o);
    });
    const show = () => {
      const e = R.examples[Number(sel.value)];
      body.textContent = "";
      const head = document.createElement("p");
      head.className = "example-title";
      const a = document.createElement("a");
      a.href = e.url; a.target = "_blank"; a.rel = "noopener";
      a.textContent = e.title + " (" + e.id + ", " + P.example.open + ")";
      head.appendChild(a);
      body.appendChild(head);
      const grid = document.createElement("div");
      grid.className = "example-grid";
      ["abstract", "synthesized", "direct-brief", "direct-full"].forEach((k) => {
        const card = document.createElement("div");
        card.className = "example-card";
        const h = document.createElement("h4");
        h.textContent = P.example[k];
        const p = document.createElement("p");
        p.textContent = e[k];
        card.appendChild(h); card.appendChild(p);
        grid.appendChild(card);
      });
      body.appendChild(grid);
    };
    sel.addEventListener("change", show);
    show();
  }

  // ── topics ──
  const M = R.map;
  const labelOf = Object.fromEntries(R.topics.map((t) => [t.id, t.label]));
  const HOVER = "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>%{customdata[3]}<extra></extra>";

  function points(keep, name, color, size, opacity) {
    const idx = [];
    for (let i = 0; i < M.x.length; i++) if (keep(i)) idx.push(i);
    return {
      type: "scatter3d", mode: "markers", name: name,
      x: idx.map((i) => M.x[i]), y: idx.map((i) => M.y[i]), z: idx.map((i) => M.z[i]),
      customdata: idx.map((i) => [M.title[i], M.who[i], M.cpc[i], M.topic[i] === -1 ? T.noise : labelOf[M.topic[i]]]),
      marker: { size: size, color: color, opacity: opacity, line: { width: 0 } },
      hovertemplate: HOVER,
    };
  }

  function map() {
    const mode = document.getElementById("map-mode").value;
    const topicId = Number(document.getElementById("map-topic").value);
    document.getElementById("map-topic-label").style.display = mode === "topic" ? "" : "none";
    let traces;
    if (mode === "topic") {
      traces = [
        points((i) => M.topic[i] === -1, T.noise, css("--noise"), 2, 0.6),
        points((i) => M.topic[i] !== -1 && M.topic[i] !== topicId, T.other_topics, css("--other"), 2, 0.7),
        points((i) => M.topic[i] === topicId, labelOf[topicId], css("--s1"), 3.4, 1),
      ];
    } else {
      traces = [
        points((i) => M.section[i] === 0, T.sec_G, series(0), 2.4, 0.85),
        points((i) => M.section[i] === 1, T.sec_H, series(1), 2.4, 0.85),
        points((i) => M.section[i] === 2, T.sec_other, series(2), 3, 0.95),
      ];
    }
    const ax = { showticklabels: false, title: { text: "" }, showgrid: true, gridcolor: css("--grid"),
                 zeroline: false, showbackground: false, showspikes: false };
    Plotly.react("chart-map", traces, layout({
      showlegend: true, uirevision: "map", margin: { l: 0, r: 0, t: 8, b: 0 },
      legend: { y: 1, x: 0, orientation: "h", itemsizing: "constant" },
      scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "cube", bgcolor: "rgba(0,0,0,0)",
               camera: { eye: { x: 1.25, y: 1.25, z: 0.75 } } },
    }), { displaylogo: false, responsive: true, displayModeBar: true,
          modeBarButtonsToRemove: ["toImage", "tableRotation", "orbitRotation"] });
  }

  function mapControls() {
    const sel = document.getElementById("map-topic");
    R.topics.forEach((t) => {
      const o = document.createElement("option");
      o.value = String(t.id);
      o.textContent = t.label + " (" + nf.format(t.n) + ")";
      sel.appendChild(o);
    });
    sel.addEventListener("change", map);
    document.getElementById("map-mode").addEventListener("change", map);
  }

  function trends() {
    const t = R.trends;
    const trunc = R.years.filter((d) => d.trunc).map((d) => d.y);
    const x0 = trunc.length ? Math.min(...trunc) - 0.5 : null;
    draw("chart-trends", t.topics.map((name, i) => ({
      type: "scatter", mode: "lines", stackgroup: "one", name: name, x: t.years, y: t.values[i],
      line: { width: 0.6, color: css("--surface") }, fillcolor: series(i),
      hovertemplate: name + "<br>%{x}: %{y:.1f}%<extra></extra>",
    })), {
      showlegend: true, legend: { orientation: "h", y: -0.18, x: 0 }, margin: { b: 40 },
      xaxis: { dtick: 1, range: [t.years[0], t.years[t.years.length - 1]], showgrid: false },
      yaxis: { title: { text: T.share_pct }, range: [0, 100] },
      shapes: x0 === null ? [] : [
        { type: "rect", xref: "x", yref: "paper", x0: x0, x1: t.years[t.years.length - 1], y0: 0, y1: 1,
          fillcolor: css("--shade"), line: { width: 0 }, layer: "above" },
        { type: "line", xref: "x", yref: "paper", x0: x0, x1: x0, y0: 0, y1: 1,
          line: { color: css("--ink-2"), width: 1, dash: "dot" }, layer: "above" },
      ],
      annotations: x0 === null ? [] : [{ x: t.years[t.years.length - 1], y: 1, xref: "x", yref: "paper",
                                         text: T.truncated, showarrow: false, xanchor: "right", yanchor: "bottom",
                                         font: { size: 12, color: css("--ink-2") } }],
      margin: { t: 34 },
    });
  }

  // ── ownership ──
  function conveyance() {
    const c = R.conveyance;
    draw("chart-conveyance", [{
      type: "bar", orientation: "h", y: P.conv, x: c.map((d) => d.n), marker: { color: css("--s1") },
      text: c.map((d) => nf.format(d.n)), textposition: "outside", cliponaxis: false,
      textfont: { color: css("--ink") }, hovertemplate: "%{y}: %{x:,}<extra></extra>",
    }], {
      yaxis: { autorange: "reversed", showgrid: false },
      xaxis: { title: { text: P.conv_label }, range: [0, Math.max(...c.map((d) => d.n)) * 1.2] },
      margin: { l: 20 },
    });
  }

  // ── validation ──
  function pairing() {
    const mk = (key, name, color) => ({
      type: "bar", name: name, x: LAYERS, y: LAYERS.map((l) => R.pairing[l][key]), marker: { color: color },
      hovertemplate: name + "<br>%{x}: %{y:.3f}<extra></extra>",
    });
    draw("chart-pairing", [mk("own", T.own, series(0)), mk("shift", T.shifted, series(1))], {
      barmode: "group", showlegend: true, yaxis: { title: { text: T.rouge1 }, range: [0, 0.5] },
    });
  }

  function dots(id, values, title, extra) {
    const y = REPS.map((r) => values[r]);
    draw(id, [{
      type: "scatter", mode: "markers+text", x: REPS, y: y, marker: { size: 11, color: css("--s1") },
      text: y.map((v) => v.toFixed(3)), textposition: "top center", textfont: { color: css("--ink") },
      cliponaxis: false, hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
    }], merge({ yaxis: { title: { text: title } }, xaxis: { showgrid: false } }, extra || {}));
  }

  function cpc() {
    const v = Object.values(R.cpc);
    dots("chart-cpc", R.cpc, T.microf1, { yaxis: { range: [Math.min(...v) - 0.02, Math.max(...v) + 0.02] } });
  }

  function boot() {
    const b = R.boot;
    draw("chart-boot", [{
      type: "scatter", mode: "markers", y: b.map((d) => d.pair), x: b.map((d) => d.est),
      marker: { size: 10, color: css("--s1") },
      error_x: { type: "data", symmetric: false, array: b.map((d) => d.hi - d.est),
                 arrayminus: b.map((d) => d.est - d.lo), color: css("--s1"), thickness: 2, width: 0 },
      customdata: b.map((d) => [d.lo, d.hi]),
      hovertemplate: "%{y}<br>%{x:.4f} [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>",
    }], {
      xaxis: { title: { text: T.difference }, zeroline: true, zerolinecolor: css("--ink-2"), zerolinewidth: 1 },
      yaxis: { autorange: "reversed", showgrid: false },
      margin: { l: 20 },
    });
  }

  function applicant() {
    const a = R.applicant;
    dots("chart-applicant", a.acc, T.accuracy, {
      yaxis: { range: [0, 0.7] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: a.majority, y1: a.majority,
                 line: { color: css("--ink-2"), width: 1, dash: "dash" } }],
      annotations: [{ xref: "paper", x: 1, y: a.majority, text: T.majority + " (" + (a.majority * 100).toFixed(1) + "%)",
                      showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 12, color: css("--ink-2") } }],
    });
  }

  function drawAll() {
    funnel(); removal(); sections(); years(); ppl(); map(); trends(); conveyance(); pairing(); cpc(); boot(); applicant();
  }

  function start() {
    examples();
    if (typeof window.Plotly === "undefined") {
      document.querySelectorAll(".chart").forEach((el) => {
        el.textContent = "plotly.js could not be loaded; the charts need an internet connection.";
        el.style.padding = "16px";
      });
      return;
    }
    mapControls();
    drawAll();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", drawAll);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
