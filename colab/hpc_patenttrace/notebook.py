"""Helpers that keep the notebooks short.

Each section of the pipeline notebook calls run_step("<file>"), which executes the matching file of steps/ in
the namespace of the notebook. The file is split at its "# ── <number>  <title>" lines; every part is timed
and recorded in cell_times.csv. Printed text goes to <output folder>/logs/<file>.log, so the notebook shows the
figures and the tables only (SHOW_LOG = True prints it as well). Progress bars are kept for the long steps.
"""
import contextlib
import functools
import io
import numbers
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
_HEADER = re.compile(r"^# ── (\d+\.\d+\S*)\s+(.*?)\s*[─ ]*$")
# parts that keep their progress bar: generation, encoding and the slow validation measures
LONG_PARTS = {"4.3", "5.6", "5.10", "5.11", "6.2", "6.7", "6.8", "6.12", "8.11", "8.12", "8.14"}
SKIP_PARTS = set()   # numbers of parts to leave out; used only by the test runs of the repository


def in_colab():
    return "google.colab" in sys.modules


def _namespace():
    from IPython import get_ipython
    ip = get_ipython()
    if ip is None:
        raise RuntimeError("run_step() runs inside a notebook (Jupyter or Google Colab).")
    return ip.user_ns


def _parts(path):
    """Split a step file at its section lines: [(number, title, first line, code)]."""
    lines = open(path, encoding="utf-8").read().splitlines()
    starts = [(i, _HEADER.match(l)) for i, l in enumerate(lines) if _HEADER.match(l)]
    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, None))
    out = []
    for k, (i, m) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        number, title = (m.group(1), m.group(2)) if m else ("", os.path.basename(path))
        out.append((number, title, i, "\n".join(lines[i:j])))
    return out


def _record(ns, number, title, seconds, ok):
    save_dir = ns.get("SAVE_DIR")
    if not save_dir or not number:
        return
    path = os.path.join(save_dir, "cell_times.csv")
    pd.DataFrame([{"cell": f"{number}  {title}".strip(), "seconds": round(seconds, 2), "ok": ok,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": ns.get("HARDWARE", "")}]
                 ).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_step(name, show_log=None):
    """Run steps/<name>.py in the notebook namespace; print nothing unless show_log is True."""
    ns = _namespace()
    path = os.path.join(STEPS_DIR, f"{name}.py")
    show_log = bool(ns.get("SHOW_LOG", False)) if show_log is None else show_log
    from tqdm.auto import tqdm as _tqdm
    quiet_tqdm = functools.partial(_tqdm, disable=True)

    def long_tqdm(iterable=None, *args, **kw):
        """A progress bar, except for an empty loop (a step that reuses its checkpoint has nothing to do)."""
        total = kw.get("total", len(iterable) if hasattr(iterable, "__len__") else None)
        if total == 0:
            kw["disable"] = True
        return _tqdm(iterable, *args, **kw)

    buffer = io.StringIO()            # holds the log until the output folder is known
    for number, title, first, code in _parts(path):
        if number in SKIP_PARTS:
            continue
        ns["tqdm"] = long_tqdm if (show_log or number in LONG_PARTS) else quiet_tqdm
        t0, ok = time.perf_counter(), False
        sink = sys.stdout if show_log else buffer
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile("\n" * first + code, path, "exec"), ns)
            ok = True
        except BaseException as err:
            log = _flush(ns, name, buffer)
            where = f"; log: {log}" if log else ""
            print(f"{name}: part {number} ({title}) stopped with {type(err).__name__}{where}", file=sys.stderr)
            raise
        finally:
            _record(ns, number, title, time.perf_counter() - t0, ok)
        _flush(ns, name, buffer)
    ns["tqdm"] = _tqdm


def _flush(ns, name, buffer):
    """Append the buffered log to <SAVE_DIR>/logs/<name>.log once SAVE_DIR exists."""
    save_dir = ns.get("SAVE_DIR")
    text = buffer.getvalue()
    if not save_dir or not text:
        return os.path.join(save_dir, "logs", f"{name}.log") if save_dir else ""
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    log = os.path.join(save_dir, "logs", f"{name}.log")
    with open(log, "a", encoding="utf-8") as f:
        f.write(text)
    buffer.seek(0)
    buffer.truncate()
    return log


def show(table, title=None, rows=30):
    """Show a table: a DataFrame, or a CSV file (relative to SAVE_DIR) shown exactly as it is stored."""
    from IPython.display import Markdown, display
    if isinstance(table, (str, os.PathLike)):
        path = str(table)
        if not os.path.isabs(path):
            path = os.path.join(_namespace().get("SAVE_DIR", ""), path)
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if title:
        display(Markdown(f"**{title}**"))
    view = table.head(rows)
    try:
        display(view.style.hide(axis="index").format(_number))
    except Exception:
        display(view)
    if len(table) > rows:
        display(Markdown(f"*{len(table) - rows:,} more rows in the file.*"))


def _number(v):
    """Thousands separators for whole numbers, at most four decimals otherwise; text stays as it is."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Number) or v != v:
        return v
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def resolve(base, path):
    """An absolute path stays as it is; a relative path is taken inside base."""
    path = os.path.expanduser(str(path).strip())
    return path if not path or os.path.isabs(path) else os.path.join(base, path)
