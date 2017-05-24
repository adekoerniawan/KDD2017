#! /usr/bin/python

import xgboost as xgb
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

from tools import get_time_window, convert_time_window_by_delta, export_predict
from train_xgb import convert_dataframe


if __name__ == "__main__":
	from load_data import *
	# Reading the csv file into pandas dataframe #
	test_df = avg_volume_test

	# Gernate dataframe for prediction.
	date_offset = DateOffset(hours=2)
	test_df.drop('volume', axis=1, inplace=True)
	test_cond = test_df.copy()

	test_df = convert_dataframe(test_df, weather_test, mean_weather_test, volume_test)
	print("Preprocessing on test data done.")

	# Load model from file.
	model_path = '../model/model_xgb_reg:linear_0.8_9_700_history.bin'
	model = xgb.Booster()
	model.load_model(model_path)
	print("Load xgboost model from {}.".format(model_path))

	test_x = xgb.DMatrix(test_df)
	pred = model.predict(test_x)
	print("Prediction finished.")

	# Export prediction data.
	export_file = '../result/volume_prediction.csv'
	export_predict(test_cond, pred, export_file, 'volume')
	print("Export prediction to {}".format(export_file))

