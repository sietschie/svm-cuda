#!/usr/bin/python
import csv
import sys


reader = csv.DictReader(open(sys.argv[1], 'r'), delimiter=';')

times = {}
quotients = {}

for row in reader:
	#print row
	#if time is not present, skip line
	if row['time'] == None:
		continue
	#if cache not present, use default value
	if 'cache' not in row.keys():
		row['cache'] = '10' 

		#if 'kernel' not in row.keys():
	#	row['kernel'] = '2' 
	key = (row['cache'], row['kernel'], row['binary'], row['Filename'])
	if  key not in times.keys():
		times[ key ] = []
		quotients[ key ] = []
	times[ key ].append(float(row['time']))
	quotients[ key ].append(float(row['i/t']))
	
time = {}
quotient = {}
variance = {}

for k in times.keys():
	time[k] = sum(times[k]) / len(times[k])
	quotient[k] = sum(quotients[k]) / len(quotients[k])
	squared_sum = 0
	for t in times[k]:
		squared_sum += (t - time[k]) * (t - time[k])
	variance[k] = squared_sum / len(times[k])
	
	

binary_names = []
binary_names.append('./svm-train')
binary_names.append([ b for (c,k,b,f) in time.keys() if b != binary_names[0] ][0])

print 'kernel, file, cache, ' + binary_names[0] +', ' +binary_names[1] + ', speedup'
	
keys = time.keys()
keys.sort()
for k in keys:
	(cache, kernel, binary, file) = k
	if binary ==  binary_names[0] and (cache, kernel,  binary_names[1], file) in time.keys():
		print kernel,
		print file,
		print cache, 
		print time[k],
		print time[(cache, kernel, binary_names[1], file)],
		print time[k] / time[(cache, kernel, binary_names[1], file)] ,
		print variance[k],
		print variance[(cache, kernel, binary_names[1], file)]

#print time