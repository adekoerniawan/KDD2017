#! /usr/bin/env python

# Script to evaluate prediction in phase 1.

import pandas as pd

# Load ground-truth and prediction.
file_label = "../dataSets/dataSet_phase2/training2_20min_avg_volume.csv"
file_pred = "../result/volume_prediction_phase1.csv"
df_label = pd.read_csv(file_label)
df_pred = pd.read_csv(file_pred)

# Evaluation.
attr_index = ['time_window', 'tollgate_id', 'direction']
attr_eval = 'volume'
df_eval = df_label.merge(df_pred, left_on=attr_index,
right_on=attr_index, suffixes=['_label', '_pred'], how='right')
print df_eval
mse = df_eval.apply(lambda x: x[attr_eval + '_label'] - x[attr_eval + '_pred'], axis=1).mean()
mape = df_eval.apply(lambda x: abs(x[attr_eval + '_label'] - x[attr_eval + '_pred'])
/ x[attr_eval +'_label'], axis=1).mean()
print("MSE: {}, MAPE: {}".format(mse, mape))

