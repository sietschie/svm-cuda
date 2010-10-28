#!/usr/bin/python
import csv
import sys


reader = csv.DictReader(open(sys.argv[1], 'r'), delimiter=';')

times = {}
quotients = {}

for row in reader:
	if (row['cache'], row['kernel'], row['binary'], row['Filename']) not in times.keys():
		times[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ] = []
		quotients[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ] = []
	print row
	times[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ].append(float(row['time']))
	quotients[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ].append(float(row['i/t']))
	
time = {}
quotient = {}

for k in times.keys():
	time[k] = sum(times[k]) / len(times[k])
	quotient[k] = sum(quotients[k]) / len(quotients[k])

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
		print time[k] / time[(cache, kernel, binary_names[1], file)] 

#print time