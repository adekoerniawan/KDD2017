#! /usr/bin/env python

from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import DateOffset

# Declare global variables.
intersections = range(1, 4)
tollgates = range(1, 4)
directions = range(2)

# Find time window of a particular time.
def get_time_window(date_time):
	minute, second = date_time.minute, date_time.second
	time_delta = timedelta(minutes=minute%20, seconds=second)
	time_window = date_time - time_delta
	return time_window

# Calculate average of weather data.
def get_mean_data(weather_data):
	mean_data = {}
	keys = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
		'temperature', 'rel_humidity', 'precipitation']
	for key in keys:
		mean_data[key] = []
	for item in weather_data.itervalues():
		for key in keys:
			mean_data[key].append(item[key])
	for key in keys:
		mean_data[key] = np.mean(mean_data[key])
	return mean_data

# Get weather data at a specific time_window.
def get_weather_data(date_time, weather_data, mean_weather_data=None):
	if not mean_weather_data:
		mean_weather_data = get_mean_data(weather_data)

	hour, minute, second = date_time.hour, date_time.minute, date_time.second
	time_delta = timedelta(hours=hour%3, minutes=minute, seconds=second)
	time_window = date_time - time_delta
	if time_window in weather_data:
		return weather_data[time_window]
	else:
		return mean_weather_data

# Query all volume within a time-window.
def get_volume_by_time(data, time_window, tollgate, direction):
	results = []
	for item in data:
		if get_time_window(item['time']) != time_window:
			continue
		if item['tollgate_id'] != tollgate or item['direction'] != direction:
			continue
		results.append(item)
	return results

# Query all trajectories within a time-window.
def get_trajectory_by_time(data, time_window, intersection, tollgate):
	results = []
	for item in data:
		if get_time_window(item['start_time']) != time_window:
			continue
		if item['tollgate_id'] != tollgate or item['intersection_id'] != intersection:
			continue
		results.append(item)
	return results

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

def export_predict(cond, pred, filename, pred_attr):
	""" Export prediction to csv file.
	Args:
		cond: Dataframe with prediction conditions.
		pred: Prediction of conditions.
		filename: Path to csv file to be export.
		pred_attr: Name of prediction attribute.
	"""
	dataframe = cond.copy()
	dataframe[pred_attr] = pred
	dataframe.to_csv(filename, index=False)
	return

if __name__ == '__main__':
	from parse_data import *

	data_root = '../dataSets/'
	weather_file = data_root + 'training/weather (table 7)_training_update.csv'
	weather_data = parse_weather_data(weather_file)
	mean_weather_data = get_mean_data(weather_data)

	time_window = "2016-08-30 12:41:23"
	time_start = datetime.strptime(time_window, "%Y-%m-%d %H:%M:%S")
	time_end = time_start + timedelta(minutes=20)

	print time_start
	print time_end

	print get_time_window(time_end)
	print get_weather_data(time_end, weather_data, mean_weather_data)
