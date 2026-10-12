"""Read the output folder of the pipeline notebook (OUTPUT_DIR/PROFILE of step6_patentsum_pipeline.ipynb).

Files are named by their path inside that folder, for example "release/facets.csv",
"release/tables/t1_funnel.csv" or "umap2d_coords.csv".
"""
import json
import os

import pandas as pd


class RunOutputs:
    """The output folder of one complete run of the pipeline notebook."""

    def __init__(self, path):
        path = os.path.expanduser(str(path))
        if not os.path.isfile(os.path.join(path, "release", "MANIFEST_release.csv")):
            raise FileNotFoundError(f"{path} is not the output folder of a complete run of the pipeline notebook: "
                                    "release/MANIFEST_release.csv is missing (section 9 writes it)")
        self.path = path

    def file(self, name):
        return os.path.join(self.path, name)

    def csv(self, name, **kw):
        return pd.read_csv(self.file(name), **kw)

    def parquet(self, name, **kw):
        return pd.read_parquet(self.file(name), **kw)

    def json(self, name):
        with open(self.file(name), encoding="utf-8") as f:
            return json.load(f)


def open_run(path):
    return RunOutputs(path)
