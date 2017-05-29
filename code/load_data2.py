#! /usr/bin/env python

# Script to load data in phase 2.

import pandas as pd
from config import config
from load_data import parse_link_data, parse_route_data, parse_weather_data, parse_volume_data, \
parse_trajectory_data, parse_avg_time, parse_avg_volume, get_mean_data

data_root = config.data_root

# Load data from csv files.

# Load link and route data for both training and test.
link_file = data_root + 'training/links (table 3).csv'
route_file = data_root + 'training/routes (table 4).csv'

link_data = parse_link_data(link_file)
route_data = parse_route_data(route_file)

# Load weather data.
weather_file = data_root + 'dataSet_phase2/weather (table 7)_2.csv'
weather_train = parse_weather_data(weather_file)
weather_file = data_root + 'testing_phase1/weather (table 7)_test1.csv'
weather_test = parse_weather_data(weather_file)
weather_data = pd.concat([weather_train, weather_test])
mean_weather = get_mean_data(weather_data)

# Load data in train phase.
volume_file = data_root + 'dataSet_phase2/volume(table 6)_training2.csv'
trajectory_file = data_root + 'dataSet_phase2/trajectories(table 5)_training2.csv'
avg_time_file = data_root + 'dataSet_phase2/training2_20min_avg_travel_time.csv'
avg_volume_file = data_root + 'dataSet_phase2/training2_20min_avg_volume.csv'

volume_train = parse_volume_data(volume_file)
trajectory_train = parse_trajectory_data(trajectory_file)
avg_time_train = parse_avg_time(avg_time_file)
avg_volume_train = parse_avg_volume(avg_volume_file)

# Load data in test phase.
volume_file = data_root + 'dataSet_phase2/volume(table 6)_test2.csv'
trajectory_file = data_root + 'dataSet_phase2/trajectories(table 5)_test2.csv'
avg_time_file = data_root + 'dataSet_phase2/test2_20min_avg_travel_time.csv'
avg_volume_file = data_root + 'dataSet_phase2/test2_20min_avg_volume.csv'

volume_test = parse_volume_data(volume_file)
trajectory_test = parse_trajectory_data(trajectory_file)
avg_time_test = parse_avg_time(avg_time_file)
avg_volume_test = parse_avg_volume(avg_volume_file)

