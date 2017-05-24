#! /usr/bin/env python

from datetime import datetime, timedelta
import numpy as np
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
		mean_data[key] = weather_data[key].mean()
	return mean_data

# Get weather data at a specific time_window.
def get_weather_data(date_time, weather_data, mean_weather_data=None):
	if not mean_weather_data:
		mean_weather_data = get_mean_data(weather_data)

	hour, minute, second = date_time.hour, date_time.minute, date_time.second
	time_delta = timedelta(hours=hour%3, minutes=minute, seconds=second)
	time_window = pd.to_datetime(date_time - time_delta)

	if time_window in weather_data.index:
		return weather_data.loc[time_window].to_dict()
	else:
		return mean_weather_data

def get_history_volume(volume_data, time_window, window_num, tollgate_id, direction):
	""" Find volume info in previous time window.
	Args:
		volume_data: Original volume data to be searched.
		time_window: Current time-window.
		window_num: Number of previous time-windows.
		tollgate_id: ID of tollgate in the volume to be queried.
		direction: Direcetion of volume to be queried.
	Returns:
		history_data: A two-dimension list with history volume data.
	"""

	volume_select = volume_data[(volume_data.tollgate_id == tollgate_id) & (volume_data.direction ==
	direction)]
	window_size = timedelta(minutes=20)
	history_data = []
	time_begin = time_window
	for i in range(window_num):
		time_end = time_begin
		time_begin -= window_size
		sub_volume = volume_select[(volume_select.time >= time_begin) & (volume_select.time <
		time_end)]
#		print("{} records found from {} to {} at Tollgate {} in direction {}.".format(
#		sub_volume.size, time_begin, time_end, tollgate_id, direction))

		# Case of missin data.
		if len(sub_volume) == 0:
			history_data.append([np.nan] * 13)
			continue

		# Capicity of the vehicle, ranging from 0 to 7, bigger the higher.
		count_model = sub_volume.vehicle_model.value_counts()
		num_model = [0 for x in xrange(8)]
		for key, value in count_model.iteritems():
			num_model[key] = value
		
		# 0-passenger vehicle, 1-cargo vehicle, 2-nan.
		count_type = sub_volume.vehicle_type.value_counts(dropna=False)
		num_type = [0 for x in xrange(3)]
		for key, value in count_type.iteritems():
			try:
				num_type[int(key)] = value
			except:
				num_type[2] = value

		# 0-without ETC, 1-with ETC.
		count_etc = sub_volume.has_etc.value_counts()
		num_etc = [0, 0]
		for key, value in count_etc.iteritems():
			num_etc[key] = value

		# Merge data of vehicle_vehicle, vehicle_type and etc.
		num_history = reduce(lambda x, y: x + y, (num_model, num_type, num_etc))
		history_data.append(num_history)
	return np.array(history_data).flatten()


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
	dataframe['time_window'] = dataframe.time_window.apply(lambda x: "[{},{})".\
	format(x, x + DateOffset(minutes=20)))
	dataframe[pred_attr] = pred
	dataframe.to_csv(filename, index=False)
	return

if __name__ == '__main__':
	from load_data import *

	time_window = "2016-08-30 12:41:23"
	time_start = datetime.strptime(time_window, "%Y-%m-%d %H:%M:%S")
	time_end = time_start + timedelta(minutes=20)

	print time_start
	print time_end

	print get_time_window(time_end)
	print get_weather_data(time_end, weather_train, mean_weather_train)
