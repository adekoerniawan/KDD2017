#! /usr/bin/env python

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

# Declare global variables.
intersections = range(1, 4)
tollgates = range(1, 4)
directions = range(2)

# Calculate average of weather data.
def get_mean_data(weather_data):
	mean_data = {}
	keys = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
		'temperature', 'rel_humidity', 'precipitation']
	for key in keys:
		mean_data[key] = weather_data[key].mean()
	return mean_data

# Find time window of a particular time.
def get_time_window(date_time):
	minute, second = date_time.minute, date_time.second
	time_delta = timedelta(minutes=minute%20, seconds=second)
	time_window = date_time - time_delta
	return time_window

def get_history_volume(volume_data, time_window, window_num, tollgate_id, direction):
	""" Find volume info in previous time window.
	Args:
		volume_data: Original average volume data to be searched.
		time_window: Current time-window.
		window_num: Number of previous time-windows.
		tollgate_id: ID of tollgate in the volume to be queried.
		direction: Direcetion of volume to be queried.
	Returns:
		history_data: A two-dimension list with history volume data.
	"""

	volume_select = volume_data[(volume_data.tollgate_id == tollgate_id) &
	(volume_data.direction == direction)]
	window_size = timedelta(minutes=20)
	history_data = [np.nan] * window_num
	time_begin = time_window
	for i in range(window_num):
		time_end = time_begin
		time_begin -= window_size
		history_volume = volume_select[volume_select.time_window == time_begin].volume.values
		if len(history_volume) > 0:
			history_data[i] = history_volume[0]
	history_data = np.array(history_data).flatten()
	return history_data

def is_holiday(date_time):
	""" Function to determine whether the date is holiday or not.
	Args:
		date_time: Pandas datetime object for judgement.
	Returns:
		0 - Not a holiday. 1 - Holiday.
	"""

	# Mid-Autumn Festival.
	if date_time.month == 9 and date_time.day >= 15 and date_time.day <= 17:
		return 1
	# National Day.
	elif date_time.month == 10 and date_time.day >= 1 and date_time.day <= 7:
		return 1
	return 0

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
