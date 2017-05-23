#! /usr/bin/env python

import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime

from tools import get_time_window, get_weather_data, export_predict

from load_data import *

def convert_dataframe(dataframe, weather_data, mean_weather_data):
	df = dataframe.copy()

	# Convert to date format.
	df['time'] = df['time_window'].apply(lambda x: x.split(',')[0].strip('['))
	df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")

	# Process date field.
	df['month'] = df['time'].apply(lambda x: x.month)
	df['day'] = df['time'].apply(lambda x: x.day)
	df['weekday'] = df['time'].apply(lambda x: x.dayofweek)
	df['hour'] = df['time'].apply(lambda x: x.hour)
	df['minute'] = df['time'].apply(lambda x: x.minute)

	# Add weather info to dataframe.
	df = add_weather_data(df, weather_data, mean_weather_data)

	# Drop unused field.
	df.drop(['time_window', 'time'], axis=1, inplace=True)
	return df

def add_weather_data(dataframe, weather_data, avg_weather_data):
	""" Add weather column to dataframe.
	Args:
		dataframe: Original dataframe.
		weather_data: A list containing dicts of weather data of each time_window.
		avg_weather_data: A dict with average weather data.
	Returns:
		df: Dataframe with additional weather info.
	"""

	df = dataframe.copy()
	df_time = df.time.values
	df_time = map(lambda x: datetime.utcfromtimestamp(x.tolist()/1e9), df_time)

	df_weather = []
	for item in df_time:
		datum = get_weather_data(item, weather_data, avg_weather_data)
		df_weather.append([datum['precipitation'], datum['rel_humidity'], datum['temperature'],
		datum['wind_direction'], datum['wind_speed']])
	df_weather = np.array(df_weather)
	df['precipitation'] = df_weather[:, 0]
	df['rel_humidity'] = df_weather[:, 1]
	df['temperature'] = df_weather[:, 2]
	df['wind_direction'] = df_weather[:, 3]
	df['wind_speed'] = df_weather[:, 4]
	return df

def rmse(label, pred):
	""" Root of mean square error.
	Args:
		label: One-dimension ndarray of groud truth.
		pred: One-dimension ndarray of prediction.
	Returns:
		error: Root of mean square error.
	"""
	return np.sqrt(((label - pred) ** 2).mean())

def mape(label, pred):
	""" Mean of absolute percentage error.
	Args:
		label: One-dimension ndarray of groud truth.
		pred: One-dimension ndarray of prediction.
	Returns:
		error: Mean of absolute percentage error.
	"""
	return np.mean(np.abs(label - pred).astype(np.float64) / label)


if __name__ == "__main__":
	# Input data path #
	data_path = "../dataSets/"
	train_file = data_path + "training/training_20min_avg_volume.csv"
	test_file = data_path + "testing_phase1/test1_20min_avg_volume.csv"

	# Reading the csv file into pandas dataframe #
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

	train_label = train_df['volume'].values
	test_label = test_df['volume'].values
	train_df.drop(['volume'], axis=1, inplace=True)
	test_df.drop(['volume'], axis=1, inplace=True)
	train_cond = train_df.copy()
	test_cond = test_df.copy()

	print "Converting dateframe..."
	train_df = convert_dataframe(train_df, weather_train, mean_weather_train)
	test_df = convert_dataframe(test_df, weather_test, mean_weather_test)

	# Config XGBoost model.
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.02
	params["min_child_weight"] = 8
	params["subsample"] = 0.9
	params["colsample_bytree"] = 0.8
	params["max_depth"] = 9
	params["seed"] = 1
	params["silent"] = 1
	num_round = 100
	k_fold = 5
	print("Params of xgboost model: {}".format(params))
	print("Number of rounds: {}, Number of fold: {}".format(num_round, k_fold))
 
	# Convert data format.
	train_x, test_x = np.array(train_df), np.array(test_df)
	train_y, test_y = train_label, test_label
	xgtrain = xgb.DMatrix(train_x, label=train_y)
	xgtest = xgb.DMatrix(test_x, label=test_y)
	watch_list = [(xgtrain, 'train'), (xgtest, 'eval')]
	
	# Train and test with xgboost.
	model = xgb.train(params, xgtrain, num_round, watch_list, verbose_eval=100)
	print("Training complete.")

#	model_path = '../model/model_xgb.bin'
#	model = xgb.Booster(model_file=model_path)
	train_pred = model.predict(xgtrain)
	test_pred = model.predict(xgtest)

	# Show model error.
	train_err = rmse(train_y, train_pred)
	test_err = rmse(test_y, test_pred)
	print("RMSE of train: {}".format(train_err))
	print("RMSE of test: {}".format(test_err))
	train_err = mape(train_y, train_pred)
	test_err = mape(test_y, test_pred)
	print "MAPE of train: {}".format(train_err)
	print "MAPE of test: {}".format(test_err)

# 	# Export predictions.
# 	print("Export predictioin to csv file...")
# 	export_predict(train_cond, train_pred, '../result/export_train.csv', 'volume')
# 	export_predict(test_cond, test_pred, '../result/export_test.csv', 'volume')
# 
# 	# Save model.
# 	params_str = '_'.join(map(str, [params['objective'], params['colsample_bytree'],
# 	params['max_depth'], num_round]))
# 	model_path = '../model/model_xgb_{}.bin'.format(params_str)
# 	model.save_model(model_path)
# 	print("Model saved to {}.".format(model_path))

