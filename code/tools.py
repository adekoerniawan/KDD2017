#! /usr/bin/env python

from datetime import datetime, timedelta

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

def get_volume_by_time(data, time_window, tollgate, direction):
	results = []
	for item in data:
		if get_time_window(item['time']) != time_window:
			continue
		if item['tollgate_id'] != tollgate or item['direction'] != direction:
			continue
		results.append(item)
	return results

def get_trajectory_by_time(data, time_window, intersection, tollgate):
	results = []
	for item in data:
		if get_time_window(item['start_time']) != time_window:
			continue
		if item['tollgate_id'] != tollgate or item['intersection_id'] != intersection:
			continue
		results.append(item)
	return results

def testing():
	time_begin = "2016-07-19 00:00:00"
	time_begin = datetime.strptime(time_begin, "%Y-%m-%d %H:%M:%S")
	time_end = "2016-10-11 23:40:00"
	time_end = datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S")
	window_size = timedelta(minutes=20)
	window_num = 6

	time_query = "2016-08-01 00:00:00"
	time_query = datetime.strptime(time_query, "%Y-%m-%d %H:%M:%S")


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
