#! /usr/bin/env python

import numpy as np
import pandas as pd

PRINT_INFO = False
data_root = '../dataSets/'

def parse_link_data(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()

	if PRINT_INFO:
		num_datum = len(lines)
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	link_data = {}
	for line in lines:
		# "link_id","length","width","lanes","in_top","out_top","lane_width"
		datum = line.strip('\n').split('"')
		link_dict = {}

		link_id = int(datum[1])
		link_dict["length"] = int(datum[3])
		link_dict["lanes"] = int(datum[7])
		link_dict["in_top"] = map(int, datum[9].split(',')) if len(datum[9]) > 0 else []
		link_dict["out_top"] = map(int, datum[11].split(',')) if len(datum[11]) > 0 else []
		link_data[link_id] = link_dict
	return link_data

def parse_route_data(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()

	if PRINT_INFO:
		num_datum = len(lines)
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	route_data = {}
	for line in lines:
		# "intersection_id","tollgate_id","link_seq"
		datum = line.strip('\n').split('"')

		intersection_id = ord(datum[1]) - ord('A') + 1
		tollgate_id = int(datum[3])
		link_seq = map(int, datum[5].split(',')) if len(datum[5]) > 0 else []
		route_data[(intersection_id, tollgate_id)] = link_seq
	return route_data

def parse_weather_data(filename):
	df = pd.read_csv(filename)

	if PRINT_INFO:
		# "date","hour","pressure","sea_pressure","wind_direction","wind_speed","temperature","rel_humidity","precipitation"
		attr_names = list(df.columns)
		num_datum = df.shape[0]
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	df['time_window'] = pd.to_datetime(df.apply(lambda x: "{} {:02}".format(x.date, x.hour), axis=1))
	# Fix errors at wind_direction.
	df['wind_direction'] = df['wind_direction'].apply(lambda x: x % 360)
	df.drop(['date', 'hour'], axis=1, inplace=True)

	return df

# Calculate average of weather data.
def get_mean_data(weather_data):
	mean_data = {}
	keys = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
		'temperature', 'rel_humidity', 'precipitation']
	for key in keys:
		mean_data[key] = weather_data[key].mean()
	return mean_data

def parse_volume_data(filename):
	df = pd.read_csv(filename)

	if PRINT_INFO:
		# "time","tollgate_id","direction","vehicle_model","has_etc","vehicle_type"
		attr_names = list(df.columns)
		num_datum = df.shape[0]
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	df['time'] = pd.to_datetime(df['time'])
	return df

def parse_travel_seq(seq):
	data = []
	for item in seq.split(';'):
		link_id, enter_time, travel_time = item.split('#')
		link_id = int(link_id)
		enter_time = datetime.strptime(enter_time, "%Y-%m-%d %H:%M:%S")
		travel_time = float(travel_time)
		datum = {'link_id': link_id, 'enter_time': enter_time, 
			'travel_time': travel_time}
		data.append(datum)
	return data

def parse_trajectory_data(filename):
	df = pd.read_csv(filename)

	if PRINT_INFO:
		# "intersection_id","tollgate_id","vehicle_id","starting_time","travel_seq","travel_time"
		attr_names = list(df.columns)
		num_datum = df.shape[0]
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	# Convert data format.
	df['starting_time'] = pd.to_datetime(df['starting_time'])
	df['intersection_id'] = df['intersection_id'].apply(lambda x: ord(x) - ord('A') + 1)

	return df

def parse_avg_time(filename):
	df = pd.read_csv(filename)

	if PRINT_INFO:
		# "intersection_id","tollgate_id","time_window","avg_travel_time"
		attr_names = list(df.columns)
		num_datum = df.shape[0]
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	# Convert data format.
	df['time_window'] = df['time_window'].apply(lambda x: x.split(',')[0].strip('['))
	df['time_window'] = pd.to_datetime(df['time_window'], format="%Y-%m-%d %H:%M:%S")
	df['intersection_id'] = df['intersection_id'].apply(lambda x: ord(x) - ord('A') + 1)

	return df

def parse_avg_volume(filename):
	df = pd.read_csv(filename)

	if PRINT_INFO:
		# "tollgate_id","time_window","direction","volume"
		attr_names = list(df.columns)
		num_datum = df.shape[0]
		print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
		print "{} items in the list.".format(num_datum)

	# Convert data format.
	df['time_window'] = df['time_window'].apply(lambda x: x.split(',')[0].strip('['))
	df['time_window'] = pd.to_datetime(df['time_window'], format="%Y-%m-%d %H:%M:%S")

	return df

def load_data(filepaths):
	weather_file, volume_file, trajectory_file, avg_time_file, avg_volume_file = filepaths

	weather_data = parse_weather_data(weather_file)
	mean_weather_data = get_mean_data(weather_data)
	volume_data = parse_volume_data(volume_file)
	trajectory_data = parse_trajectory_data(trajectory_file)
	avg_time_data = parse_avg_time(avg_time_file)
	avg_volume_data = parse_avg_volume(avg_volume_file)

	return (weather_data, mean_weather_data, volume_data, trajectory_data, avg_time_data,
	avg_volume_data)

# Load data from csv files.

# Load link and route data for both training and test.
link_file = data_root + 'training/links (table 3).csv'
route_file = data_root + 'training/routes (table 4).csv'

link_data = parse_link_data(link_file)
route_data = parse_route_data(route_file)

weather_file = data_root + 'training/weather (table 7)_training_update.csv'
volume_file = data_root + 'training/volume(table 6)_training.csv'
trajectory_file = data_root + 'training/trajectories(table 5)_training.csv'
avg_time_file = data_root + 'training/training_20min_avg_travel_time.csv'
avg_volume_file = data_root + 'training/training_20min_avg_volume.csv'
filepaths = [weather_file, volume_file, trajectory_file, avg_time_file, avg_volume_file]

train_data = load_data(filepaths)
weather_train, mean_weather_train, volume_train, trajectory_train, avg_time_train, avg_volume_train = train_data

weather_file = data_root + 'testing_phase1/weather (table 7)_test1.csv'
volume_file = data_root + 'testing_phase1/volume(table 6)_test1.csv'
trajectory_file = data_root + 'testing_phase1/trajectories(table 5)_test1.csv'
avg_time_file = data_root + 'testing_phase1/test1_20min_avg_travel_time.csv'
avg_volume_file = data_root + 'testing_phase1/test1_20min_avg_volume.csv'
filepaths = [weather_file, volume_file, trajectory_file, avg_time_file, avg_volume_file]

test_data = load_data(filepaths)
weather_test, mean_weather_test, volume_test, trajectory_test, avg_time_test, avg_volume_test = test_data
