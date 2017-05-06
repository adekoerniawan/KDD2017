#! /usr/bin/env python

# from load_data import *
from datetime import datetime, timedelta

# Find time window of a particular time.
def get_time_window(date_time):
	minute, second = date_time.minute, date_time.second
	time_delta = timedelta(minutes=minute%20, seconds=second)
	time_window = date_time - time_delta
	return time_window

def get_weather_data(date_time, weather_data, mean_weather_data):
	hour, minute, second = date_time.hour, date_time.minute, date_time.second
	time_delta = timedelta(hours=hour%3, minutes=minute, seconds=second)
	time_window = date_time - time_delta
	if time_window in weather_data:
		return weather_data[time_window]
	else:
		return mean_weather_data

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