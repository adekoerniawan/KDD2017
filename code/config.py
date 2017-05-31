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
config.add_history = False
config.window_num = 6

# Parameters for xgb model.
xgb_params = {}
xgb_params["objective"] = "reg:linear"
xgb_params["eta"] = 0.02
xgb_params["min_child_weight"] = 8
xgb_params["subsample"] = 0.9
xgb_params["colsample_bytree"] = 0.8
xgb_params["max_depth"] = 6
xgb_params["seed"] = 1
xgb_params["silent"] = 1

use_pretrained = False
config.xgb_model = '../model/model_xgb_reg:linear_0.8_6_iter3000.bin'
if use_pretrained:
	xgb_params['pretrained_model'] = config.xgb_model
config.xgb_params = xgb_params
config.xgb_num_round = 3000

