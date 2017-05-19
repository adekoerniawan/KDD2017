#! /usr/bin/env python
import cPickle as pickle

pickle_root = './'

# import link and route data
pkl_link_route = pickle_root + 'link_route.pkl'
f = open(pkl_link_route, 'rb')
link_data = pickle.load(f)
route_data = pickle.load(f)
f.close()
print("Link and route data loaded.")

# import train data
pkl_train = pickle_root + 'train_data.pkl'
f = open(pkl_train, 'rb')
weather_train = pickle.load(f)
mean_weather_train = pickle.load(f)
volume_train = pickle.load(f)
trajectory_train = pickle.load(f)
avg_time_train = pickle.load(f)
avg_volume_train = pickle.load(f)
f.close()
print("Train data loaded.")

# import test data
pkl_test = pickle_root + 'test_data.pkl'
f = open(pkl_test, 'rb')
weather_test = pickle.load(f)
mean_weather_test = pickle.load(f)
volume_test = pickle.load(f)
trajectory_test = pickle.load(f)
avg_time_test = pickle.load(f)
avg_volume_test = pickle.load(f)
f.close()
print("Test data loaded.")

