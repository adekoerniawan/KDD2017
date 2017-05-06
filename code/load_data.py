#! /usr/bin/env python
import cPickle as pickle

pickle_root = './'

# import link and route data
pkl_link_route = pickle_root + 'link_route.pkl'
f = open(pkl_link_route, 'rb')
link_data = pickle.load(f)
route_data = pickle.load(f)
f.close()

# import train data
pkl_train = pickle_root + 'train_data.pkl'
f = open(pkl_train, 'rb')
weather_data_train = pickle.load(f)
mean_weather_data_train = pickle.load(f)
volume_data_train = pickle.load(f)
trajectory_data_train = pickle.load(f)
avg_time_data_train = pickle.load(f)
avg_volume_data_train = pickle.load(f)
f.close()

# import test data
pkl_test = pickle_root + 'test_data.pkl'
f = open(pkl_test, 'rb')
weather_data_test = pickle.load(f)
mean_weather_data_test = pickle.load(f)
volume_data_test = pickle.load(f)
trajectory_data_test = pickle.load(f)
avg_time_data_test = pickle.load(f)
avg_volume_data_test = pickle.load(f)
f.close()