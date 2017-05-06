#! /usr/bin/env python

import numpy as np
from datetime import datetime
import cPickle as pickle

data_root = '../dataSets/'

def parse_link_data(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()

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
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()
	
	num_datum = len(lines)
	print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
	print "{} items in the list.".format(num_datum)

	weather_data = {}
	for line in lines:
		# "date","hour","pressure","sea_pressure","wind_direction","wind_speed","temperature","rel_humidity","precipitation"
		datum = map(lambda x: x.strip('"'), line.strip('\n').split(','))
		weather_dict = {}

		date = datetime.strptime(datum[0], "%Y-%m-%d")
		hour = int(datum[1])
		date_hour = datetime(date.year, date.month,
			date.day, hour, 0, 0)
		weather_dict['pressure'] = float(datum[2])
		weather_dict['sea_pressure'] = float(datum[3])
		# Fix errors at wind_direction.
		weather_dict['wind_direction'] = float(datum[4]) % 360
		weather_dict['wind_speed'] = float(datum[5])
		weather_dict['temperature'] = float(datum[6])
		weather_dict['rel_humidity'] = float(datum[7])
		weather_dict['precipitation'] = float(datum[8])
		# print weather_dict
		weather_data[date_hour] = weather_dict
	return weather_data

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

def parse_volume_data(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()
	
	num_datum = len(lines)
	print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
	print "{} items in the list.".format(num_datum)

	volume_data = []
	for line in lines:
		# "time","tollgate_id","direction","vehicle_model","has_etc","vehicle_type"
		datum = map(lambda x: x.strip('"'), line.strip('\n').split(','))
		volume_dict = {}

		volume_dict["time"] = datetime.strptime(datum[0], "%Y-%m-%d %H:%M:%S")
		volume_dict["tollgate_id"] = int(datum[1])
		volume_dict["direction"] = int(datum[2])
		volume_dict["vehicle_model"] = int(datum[3])
		volume_dict["has_etc"] = int(datum[4])
		volume_dict["vehicle_type"] = int(datum[5]) if (datum[5].isdigit()) else -1

		volume_data.append(volume_dict)
	return volume_data

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
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()
	
	num_datum = len(lines)
	print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
	print "{} items in the list.".format(num_datum)

	trajectory_data = []
	for line in lines:
		# "intersection_id","tollgate_id","vehicle_id","starting_time","travel_seq","travel_time"
		datum = map(lambda x: x.strip('"'), line.strip('\n').split(','))
		trajectory_dict = {}

		trajectory_dict['intersection_id'] = ord(datum[0]) - ord('A') + 1
		trajectory_dict['tollgate_id'] = int(datum[1])
		trajectory_dict['vehicle_id'] = int(datum[2])
		trajectory_dict['starting_time'] = datetime.strptime(datum[3], "%Y-%m-%d %H:%M:%S")
		trajectory_dict['travel_seq'] = parse_travel_seq(datum[4])
		trajectory_dict['travel_time'] = float(datum[5])

		trajectory_data.append(trajectory_dict)
	return trajectory_data

def parse_avg_time(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()
	
	num_datum = len(lines)
	print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
	print "{} items in the list.".format(num_datum)

	avg_time_data = {}
	for line in lines:
		# "intersection_id","tollgate_id","time_window","avg_travel_time"
		datum = map(lambda x: x.strip('"'), line.strip('\n').split(','))

		intersection_id = ord(datum[0]) - ord('A') + 1
		tollgate_id = int(datum[1])
		time_window = datetime.strptime(datum[2].strip('['), "%Y-%m-%d %H:%M:%S")
		avg_travel_time = float(datum[4])

		avg_time_data[(intersection_id, tollgate_id, time_window)] = avg_travel_time
	return avg_time_data

def parse_avg_volume(filename):
	with open(filename, 'r') as f:
		attr_line = f.readline().strip('\n')  # Skip the header.
		attr_names = map(lambda x: x.strip('"'), attr_line.split(','))
		lines = f.readlines()
	
	num_datum = len(lines)
	print "Found the following {} attributes: {}.".format(len(attr_names), attr_names)
	print "{} items in the list.".format(num_datum)

	avg_volume_time = {}
	for line in lines:
		# "tollgate_id","time_window","direction","volume"
		datum = map(lambda x: x.strip('"'), line.strip('\n').split(','))

		tollgate_id = int(datum[0])
		time_window = datetime.strptime(datum[1].strip('['), "%Y-%m-%d %H:%M:%S")
		direction = int(datum[3])
		volume = float(datum[4])

		avg_volume_time[(tollgate_id, direction, time_window)] = volume
	return avg_volume_time

def generate_link_route_pkl(pickle_name):
	# Save link and route info used for both training and test.
	link_file = data_root + 'training/links (table 3).csv'
	route_file = data_root + 'training/routes (table 4).csv'

	link_data = parse_link_data(link_file)
	route_data = parse_route_data(route_file)

	f = open(pickle_name, 'wb')
	pickle.dump(link_data, f)
	pickle.dump(route_data, f)
	f.close()

def generate_train_pkl(pickle_name):
	# Save weather, volume, trajectory info for training.
	weather_file = data_root + 'training/weather (table 7)_training_update.csv'
	volume_file = data_root + 'training/volume(table 6)_training.csv'
	trajectory_file = data_root + 'training/trajectories(table 5)_training.csv'
	avg_time_file = data_root + 'training/training_20min_avg_travel_time.csv'
	avg_volume_file = data_root + 'training/training_20min_avg_volume.csv'

	weather_data = parse_weather_data(weather_file)
	mean_weather_data = get_mean_data(weather_data)
	volume_data = parse_volume_data(volume_file)
	trajectory_data = parse_trajectory_data(trajectory_file)
	avg_time_data = parse_avg_time(avg_time_file)
	avg_volume_data = parse_avg_volume(avg_volume_file)

	f = open(pickle_name, 'wb')
	pickle.dump(weather_data, f)
	pickle.dump(mean_weather_data, f)
	pickle.dump(volume_data, f)
	pickle.dump(trajectory_data, f)
	pickle.dump(avg_time_data, f)
	pickle.dump(avg_volume_data, f)
	f.close()

def generate_test_pkl(pickle_name):
	# Save weather, volume, trajectory info for testing.
	weather_file = data_root + 'testing_phase1/weather (table 7)_test1.csv'
	volume_file = data_root + 'testing_phase1/volume(table 6)_test1.csv'
	trajectory_file = data_root + 'testing_phase1/trajectories(table 5)_test1.csv'
	avg_time_file = data_root + 'testing_phase1/test1_20min_avg_travel_time.csv'
	avg_volume_file = data_root + 'testing_phase1/test1_20min_avg_volume.csv'

	weather_data = parse_weather_data(weather_file)
	mean_weather_data = get_mean_data(weather_data)
	volume_data = parse_volume_data(volume_file)
	trajectory_data = parse_trajectory_data(trajectory_file)
	avg_time_data = parse_avg_time(avg_time_file)
	avg_volume_data = parse_avg_volume(avg_volume_file)

	f = open(pickle_name, 'wb')
	pickle.dump(weather_data, f)
	pickle.dump(mean_weather_data, f)
	pickle.dump(volume_data, f)
	pickle.dump(trajectory_data, f)
	pickle.dump(avg_time_data, f)
	pickle.dump(avg_volume_data, f)
	f.close()

if __name__ == '__main__':
	pickle_file = 'data.pkl'

	pickle_link_route = 'link_route.pkl'
	pickle_train_data = 'train_data.pkl'
	pickle_test_data = 'test_data.pkl'

	generate_link_route_pkl(pickle_link_route)
	generate_train_pkl(pickle_train_data)
	generate_test_pkl(pickle_test_data)

	print "Link and route data loaded into {}.".format(pickle_link_route)
	print "Train data loaded into {}.".format(pickle_train_data)
	print "Test data loaded into {}.".format(pickle_test_data)
