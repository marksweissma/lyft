# Lyft travel time takehome

## Directory Structure

* analysis: jupyter notebook - `lyft.ipynb` and support helper functions for analysis `notebook.py`
* confs: configuration files for training
* data: csvs - upstream source and tabular data written out during training
* models: model artifacts produced during training
* training: executor `run.py` and helpers in `utils.py` - note many of these are pulled in or adapted from another project
* result: scored csvs of the test set

## Key artifacts:
1. **The scored test set `duration.csv` with header `row_id` and `duration` is at `result/duration.csv` from the root**
2. **Discussion of data, model approach and performance is in Markdown at the top of the notebook `analysis/lyft.ipynb` from the root**


