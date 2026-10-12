"""Code of the HPC-PatentTrace notebooks.

notebook  helpers of the notebooks: run_step() runs one pipeline step, show() shows a table
outputs   reads the output folder of the pipeline notebook
figures   Figures 1 and 2 of the data descriptor
tables    Tables of the data descriptor
steps/    the pipeline, one file per section of the pipeline notebook
"""
from .notebook import in_colab, resolve, run_step, show
from .outputs import RunOutputs, open_run

__version__ = "1.0.0"
__all__ = ["RunOutputs", "in_colab", "open_run", "resolve", "run_step", "show"]
