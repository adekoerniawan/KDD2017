#! /usr/bin/python

import xgboost as xgb
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

from tools import get_time_window
from train_xgb import convert_dataframe, add_weather_data

def convert_time_window_by_delta(dataframe, date_offset = DateOffset(0)):
	""" Alter time_window of dataframe by offset.
	Args:
		dataframe: Original dataframe with column 'time_window'.
		date_offset: Offset of time_window, set to DateOffset(0) as default.
	Returns:
		df: Dataframe with time_window changed.
	"""

	df = dataframe.copy()
	df['time_begin'] = pd.to_datetime(df.time_window.apply(lambda x: x.split(',')[0].strip('[')))
	df['time_end'] = pd.to_datetime(df.time_window.apply(lambda x: x.split(',')[1].strip(')')))
	df['time_begin'] = df.time_begin.apply(lambda x: x + date_offset)
	df['time_end'] = df.time_end.apply(lambda x: x + date_offset)
	df['time_window'] = df.apply(lambda x:
	'[' + str(x['time_begin']) + ',' + str(x['time_end']) + ')', axis=1)
	df.drop(['time_begin', 'time_end'], axis=1, inplace=True)
	return df

if __name__ == "__main__":
	# Input data path #
	data_path = "../dataSets/"
	test_file = data_path + "testing_phase1/test1_20min_avg_volume.csv"

	# Reading the csv file into pandas dataframe #
	test_df = pd.read_csv(test_file)

	# Gernate dataframe for prediction.
	date_offset = DateOffset(hours=2)
	test_df = convert_time_window_by_delta(test_df, date_offset)
	test_df.drop('volume', axis=1, inplace=True)
	test_cond = test_df.copy()

	from load_data import *
	test_df = convert_dataframe(test_df, weather_test, mean_weather_test)
	print("Preprocessing on test data done.")

	# Load model from file.
	model_path = '../model/model_xgb.bin'
	model = xgb.Booster()
	model.load_model(model_path)
	print("Load xgboost model from {}.".format(model_path))

	test_x = xgb.DMatrix(test_df)
	pred = model.predict(test_x)
	print("Prediction finished.")

	# Export prediction data.
	export_file = '../result/volume_prediction.csv'
	test_cond['volume'] = pred
	test_cond.to_csv(export_file, index=False)
	print("Export prediction to {}".format(export_file))

