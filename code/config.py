#! /usr/bin/env python

# Configuration for the project.
class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

config = AttrDict()
config.print_info = False
config.data_root = '../dataSets/'
config.val_ratio = 0.2

# Configuration for training feature.
config.add_weather = True
config.weather_attr = ['']
config.add_history = True
config.window_num = 6

# Parameters for xgb model.
xgb_params = {}
xgb_params["objective"] = "reg:linear"
xgb_params["eta"] = 0.02
xgb_params["min_child_weight"] = 8
xgb_params["subsample"] = 0.9
xgb_params["colsample_bytree"] = 0.8
xgb_params["max_depth"] = 9
xgb_params["seed"] = 1
xgb_params["silent"] = 1
config.xgb_params = xgb_params
config.xgb_num_round = 700
config.xgb_nfold = 5

