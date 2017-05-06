#! /usr/bin/env python

file_name = '../dataSets/training/weather (table 7)_training_update.csv'
# file_name = '../dataSets/testing_phase1/weather (table 7)_test1.csv'

with open(file_name, 'r') as f:
	lines = f.readlines()

attr = map(lambda x: x.strip('\"'), lines[0].rstrip('\n').split(','))
print attr
lines = lines[1:]
print "Find", len(lines), "records."

diffs = set()
date_hours = set()
for line in lines:
	datum = map(lambda x: x.strip('\"'), line.rstrip('\n').split(','))
	date, hour = datum[0], datum[1]
	pres = float(datum[2])  # Pressure
	slp = float(datum[3])  # Sea level pressure
	wind_dir, wind_sp = float(datum[4]), float(datum[5])
	temp = datum[6]  # Temperature
	rh, pcpn = float(datum[7]), float(datum[8])  # Relative-humidity and precipitation
	
	if (wind_dir >= 360.0 or wind_dir < 0.0):
		print "Wind_dir:", wind_dir
	if wind_sp < 0.0:
		print "Wind_sp:", wind_sp
	if not (rh >= 0.0 and rh <= 100):
		print "RH:", rh
	if pcpn < 0.0:
		print "PCPN:", pcpn

	date_hour = (date, hour)
	date_hours.add(date_hour)

# Search for missing weather statistic.
day_num = [('07', 31), ('08', 31), ('09', 30), ('10', 17)]
hours = map(lambda x: str(x), range(0, 24, 3))
weather_missing = 0
for month, days_in_month in day_num:
	for day in range(days_in_month):
		date_str = "2016-{}-{:0>2}".format(month, day + 1)
		for hour in hours:
			if (date_str, hour) not in date_hours:
				weather_missing += 1
				print (date_str, hour)
print "{} items missing in weather table.".format(weather_missing)
