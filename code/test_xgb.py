#! /usr/bin/python

import xgboost as xgb
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

from tools import export_predict
from train_xgb import convert_dataframe
from config import config

if __name__ == "__main__":
	from load_data_all import *
	# Reading the csv file into pandas dataframe #
	test_df = avg_volume_test

	# Gernate dataframe for prediction.
	date_offset = DateOffset(hours=2)
	test_df['time_window'] = test_df.time_window.apply(lambda x: x + date_offset)
	test_df.drop('volume', axis=1, inplace=True)
	test_cond = test_df.copy()

	test_df = convert_dataframe(test_df, weather_data, mean_weather, volume_test)
	print("Preprocessing on test data done.")

	# Load model from file.
	model_path = config.xgb_model
	model = xgb.Booster(model_file=model_path)
	print("Load xgboost model from {}.".format(model_path))

	test_x = xgb.DMatrix(test_df)
	pred = model.predict(test_x)
	print("Prediction finished.")

	# Export prediction data.
	export_file = '../result/volume_prediction_phase2.csv'
	export_predict(test_cond, pred, export_file, 'volume')
	print("Export prediction to {}".format(export_file))

