#! /usr/bin/env python

import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime
import time

from tools import get_time_window, get_weather_data, export_predict, get_history_volume
from load_data import *
from config import config

def convert_dataframe(dataframe, weather_data, mean_weather_data, volume_data):
	df = dataframe.copy()

	# Process date field.
	df['month'] = df['time_window'].apply(lambda x: x.month)
	df['day'] = df['time_window'].apply(lambda x: x.day)
	df['weekday'] = df['time_window'].apply(lambda x: x.dayofweek)
	df['hour'] = df['time_window'].apply(lambda x: x.hour)
	df['minute'] = df['time_window'].apply(lambda x: x.minute)

	# Add extra info to dataframe.
	if config.add_weather:
		df = add_weather_data(df, weather_data, mean_weather_data)
	if config.add_history:
		window_num = config.window_num
		df = add_history_volume(df, volume_data, window_num)

	# Drop unused field.
	df.drop(['time_window'], axis=1, inplace=True)
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
	df_time = df.time_window.values
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

def add_history_volume(dataframe, volume_data, window_num):
	""" Add history volume info into dataframe.
	Args:
		dataframe: Original dataframe.
		volume_data: A dataframe with history volume data.
		window_num: Number of window involved.
	Returns:
		df: Dataframe with additional history info.
	"""

	df_history = dataframe.apply(lambda x: pd.Series(get_history_volume(volume_data, x.time_window,
	window_num, x.tollgate_id, x.direction)), axis=1)
	df = pd.concat([avg_volume_test, df_history], axis=1)
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
	# Reading the csv file into pandas dataframe #
	train_df = avg_volume_train
	test_df = avg_volume_test

	# Divide dataset into training part and validation part.
	num_train = len(train_df)
	idx_train = np.arange(num_train)
	num_val = int(num_train * config.val_ratio)
	idx_val = np.random.randint(0, num_train, num_val)
	idx_traini_part = np.delete(idx_train, idx_val)
	num_train = num_train - num_val
	print("Divide dataset into training set with {} samples and validation set with {} \
	samples.".format(num_train, num_val))

	train_label = train_df['volume'].values
	test_label = test_df['volume'].values
	train_df.drop(['volume'], axis=1, inplace=True)
	test_df.drop(['volume'], axis=1, inplace=True)
	train_cond = train_df.copy()
	test_cond = test_df.copy()

	print "Converting dateframe..."
	t1 = time.time()
	train_df = convert_dataframe(train_df, weather_train, mean_weather_train, volume_train)
	t2 = time.time()
	print("Convert train dataframe in {} seconds.".format(t2 - t1))
	t1 = time.time()
	test_df = convert_dataframe(test_df, weather_test, mean_weather_test, volume_test)
	t2 = time.time()
	print("Convert test dataframe in {} seconds.".format(t2 - t1))

	# Config XGBoost model.
	params = config.xgb_params
	num_round = config.xgb_num_round
	k_fold = config.xgb_nfold
	print("Params of xgboost model: {}".format(params))
	print("Number of rounds: {}, Number of fold: {}".format(num_round, k_fold))
 
	# Convert data format.
	train_x, test_x = train_df, test_df
	train_y, test_y = train_label, test_label

	train_part_x, train_part_y = train_df.iloc[idx_train], train_label[idx_train]
	val_x, val_y = train_df.iloc[idx_val], train_label[idx_val]

	# Construct DMatrix for xgb model.
	xgb_train = xgb.DMatrix(train_x, label=train_y)
	xgb_train_part = xgb.DMatrix(train_part_x, label=train_part_y)
	xgb_val = xgb.DMatrix(val_x, label=val_y)
	xgb_test = xgb.DMatrix(test_x, label=test_y)
	watch_list = [(xgb_train_part, 'train'), (xgb_val, 'eval')]
	
	# Train and test with xgboost.
	model = xgb.train(params, xgb_train, num_round, watch_list, verbose_eval=100)
	print("Training complete.")

#	model_path = '../model/model_xgb.bin'
#	model = xgb.Booster(model_file=model_path)
	train_pred = model.predict(xgb_train)
	test_pred = model.predict(xgb_test)

	# Show model error.
	train_err = rmse(train_y, train_pred)
	test_err = rmse(test_y, test_pred)
	print("RMSE of train: {}".format(train_err))
	print("RMSE of test: {}".format(test_err))
	train_err = mape(train_y, train_pred)
	test_err = mape(test_y, test_pred)
	print "MAPE of train: {}".format(train_err)
	print "MAPE of test: {}".format(test_err)

 	# Export predictions.
 	print("Export predictioin to csv file...")
 	export_predict(train_cond, train_pred, '../result/export_train.csv', 'volume')
 	export_predict(test_cond, test_pred, '../result/export_test.csv', 'volume')
 
 	# Save model.
 	params_str = '_'.join(map(str, [params['objective'], params['colsample_bytree'],
 	params['max_depth'], num_round]))
	if config.add_history:
		params_str += '_history{}'.format(config.window_num)

 	model_path = '../model/model_xgb_{}.bin'.format(params_str)
 	model.save_model(model_path)
 	print("Model saved to {}.".format(model_path))

