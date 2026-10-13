"""Release names: every "ipc" in a file or column name of release/ becomes "cpc".

The class codes come from the Cooperative Patent Classification (CPC) paths of Google Patents. The crawler output
and the working files of the pipeline keep their historical names (main_ipc, main_ipc_4, ipc_description, ...);
section 9.4 converts release/ with convert_release(). Only names change:
  - file names and column names: ipc -> cpc (lower case only)
  - the JSON entries of ipc_description: keys "ipc" and "ipc_description" -> "cpc" and "cpc_description"
  - three columns that list field or step names, and the file list of MANIFEST_release.csv (VALUE_COLUMNS)
No class code, no patent text and no result is changed. Every converted file is checked against the file it
replaces (rows, identifiers, class codes, all other values), and NAME_MAPPING_ipc_to_cpc.csv lists each old and
new name. The conversion can be run alone on the release/ folder of an earlier run:
    python -m hpc_patenttrace.names <release folder>
"""
import json
import os
import sys

import pandas as pd

VALUE_COLUMNS = {"t2_field_completeness.csv": "field", "e6_top_share_summary.csv": "ipc_level",
                 "runtime_log_release.csv": "step", "MANIFEST_release.csv": "file"}
JSON_COLUMN = "ipc_description"
MAPPING_FILE = "NAME_MAPPING_ipc_to_cpc.csv"
CHUNK = 2000


def cpc_name(name):
    """Release name of a file, column or listed field: ipc -> cpc (lower case only, so no other word changes).
    The mapping file keeps its name, so that a second run changes nothing."""
    if os.path.basename(name) == MAPPING_FILE:
        return name
    return name.replace("ipc", "cpc")


def _keys(value):
    """JSON list of {ipc, ipc_description} -> the same list with the keys cpc and cpc_description."""
    if not value:
        return value
    try:
        items = json.loads(value)
    except (TypeError, ValueError):
        return value
    if not isinstance(items, list):
        return value
    return json.dumps([{cpc_name(k): v for k, v in it.items()} if isinstance(it, dict) else it for it in items],
                      ensure_ascii=False)


def _codes(value, key):
    try:
        return [it.get(key) for it in json.loads(value) if isinstance(it, dict)] if value else []
    except (TypeError, ValueError):
        return value


def _needs(path):
    base = os.path.basename(path)
    if "ipc" in base or base in VALUE_COLUMNS:
        return True
    return any("ipc" in c for c in pd.read_csv(path, nrows=0).columns)


def _convert_chunk(frame, base):
    out = frame.copy()
    if JSON_COLUMN in out.columns:
        out[JSON_COLUMN] = out[JSON_COLUMN].map(_keys)
    col = VALUE_COLUMNS.get(base)
    if col in out.columns:
        out[col] = out[col].map(cpc_name)
    out = out.rename(columns=cpc_name)
    dup = out.columns[out.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"{base}: two columns get the same release name {dup}")
    return out


def _chunks(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=CHUNK)


def _convert_file(src, dest):
    base, first = os.path.basename(src), True
    for chunk in _chunks(src):
        _convert_chunk(chunk, base).to_csv(dest, mode="w" if first else "a", header=first, index=False)
        first = False
    if first:                                   # header only
        _convert_chunk(pd.read_csv(src, dtype=str, nrows=0), base).to_csv(dest, index=False)


def _check_file(src, dest):
    """Compare the converted file with the original: rows, column order, every value. Returns the counts."""
    base = os.path.basename(src)
    col = VALUE_COLUMNS.get(base)
    rows, values_renamed, codes_checked = 0, set(), 0
    for old, new in zip(_chunks(src), _chunks(dest)):
        assert list(new.columns) == [cpc_name(c) for c in old.columns], f"{base}: columns differ"
        assert len(old) == len(new), f"{base}: rows differ"
        for c in old.columns:
            o, n = old[c].reset_index(drop=True), new[cpc_name(c)].reset_index(drop=True)
            if c == JSON_COLUMN:
                assert (n == o.map(_keys)).all(), f"{base}: {c} differs beyond its keys"
                assert all(_codes(a, "ipc") == _codes(b, "cpc") for a, b in zip(o, n)), f"{base}: class codes differ"
                codes_checked += len(o)
            elif c == col:
                assert (n == o.map(cpc_name)).all(), f"{base}: {c} differs beyond the names"
                values_renamed |= {(a, b) for a, b in zip(o, n) if a != b}
            else:
                assert (n == o).all(), f"{base}: values of {c} differ"
        rows += len(old)
    total_new = sum(len(c) for c in _chunks(dest))
    assert total_new == rows, f"{base}: {rows} rows before, {total_new} after"
    return rows, values_renamed, codes_checked


def convert_release(folder):
    """Convert release/ in place to the CPC names; return one row per converted file (all checks passed)."""
    report, mapping = [], []
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                   if f.endswith(".csv") and f != MAPPING_FILE)
    for src in paths:
        if not _needs(src):
            continue
        rel = os.path.relpath(src, folder).replace(os.sep, "/")
        dest = os.path.join(os.path.dirname(src), cpc_name(os.path.basename(src)))
        tmp = dest + ".converting"
        _convert_file(src, tmp)
        rows, values, codes = _check_file(src, tmp)
        old_cols = list(pd.read_csv(src, nrows=0).columns)
        os.replace(tmp, dest)
        if dest != src:
            os.remove(src)
        new_rel = os.path.relpath(dest, folder).replace(os.sep, "/")
        if new_rel != rel:
            mapping.append(("file", new_rel, rel, new_rel))
        mapping += [("column", new_rel, c, cpc_name(c)) for c in old_cols if cpc_name(c) != c]
        if JSON_COLUMN in old_cols:
            mapping += [("json key", new_rel, "ipc", "cpc"), ("json key", new_rel, "ipc_description", "cpc_description")]
        mapping += [("value", new_rel, a, b) for a, b in sorted(values)]
        report.append({"file": new_rel, "was": rel if new_rel != rel else "", "rows": rows,
                       "columns renamed": sum(cpc_name(c) != c for c in old_cols),
                       "values renamed": len(values), "class codes checked": codes, "check": "same"})
    out = pd.DataFrame(mapping, columns=["kind", "file", "old_name", "new_name"])
    mpath = os.path.join(folder, MAPPING_FILE)
    if os.path.exists(mpath):                   # a second run keeps the names of the first one
        out = pd.concat([pd.read_csv(mpath, dtype=str, keep_default_na=False), out]).drop_duplicates()
    if len(out):
        out.to_csv(mpath, index=False)
        man = os.path.join(folder, "MANIFEST_release.csv")
        if os.path.exists(man):
            m = pd.read_csv(man, dtype=str, keep_default_na=False)
            if MAPPING_FILE not in set(m["file"]):
                row = pd.DataFrame([{"file": MAPPING_FILE, "rows": str(len(out)), "columns": "4",
                                     "MB": str(round(os.path.getsize(mpath) / 1e6, 2))}])
                pd.concat([m, row], ignore_index=True).to_csv(man, index=False)
    return pd.DataFrame(report, columns=["file", "was", "rows", "columns renamed", "values renamed",
                                         "class codes checked", "check"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        raise SystemExit("usage: python -m hpc_patenttrace.names <release folder>")
    print(convert_release(sys.argv[1]).to_string(index=False))
