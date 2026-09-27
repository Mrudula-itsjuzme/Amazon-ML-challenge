# Entity Resolution Submission

## Overview
This solution uses a fast Sparse TF-IDF candidate generation followed by a LightGBM classification model. It efficiently handles the scale of 1.7M test queries against 10M targets.

## Running the Code
Ensure you have the dependencies installed:
`pip install pandas numpy scipy scikit-learn lightgbm thefuzz unidecode`

Run the end-to-end pipeline (generates candidates, trains the model, and predicts on the test set):
`python3 src/hackathon_solution.py`

Outputs will be saved in the `output/` directory.
