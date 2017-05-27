#! /usr/bin/env python

# Script to evaluate prediction in phase 1.

import pandas as pd

# Load ground-truth and prediction.
file_label = "../dataSets/dataSet_phase2/training2_20min_avg_volume.csv"
file_pred = "../result/volume_prediction_phase1.csv"
df_label = pd.read_csv(file_label)
df_pred = pd.read_csv(file_pred)

# Evaluation.
df_eval = df_label.merge(df_pred, left_on=['time_window', 'tollgate_id', 'direction'],
right_on=['time_window', 'tollgate_id', 'direction'], suffixes=['_label', '_pred'], how='right')
print df_eval
mse = df_eval.apply(lambda x: x.volume_label - x.volume_pred, axis=1).mean()
mape = df_eval.apply(lambda x: abs(x.volume_label - x.volume_pred) / x.volume_label, axis=1).mean()
print("MSE: {}, MAPE: {}".format(mse, mape))

