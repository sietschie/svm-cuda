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
	#print row
	times[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ].append(float(row['time']))
	quotients[( row['cache'], row['kernel'], row['binary'], row['Filename'] ) ].append(float(row['i/t']))
	
time = {}
quotient = {}
	
for k in times.keys():
	time[k] = sum(times[k]) / len(times[k])
	quotient[k] = sum(quotients[k]) / len(quotients[k])
	
print 'kernel, file, cache, seriell, cuda, speedup'
keys = time.keys()
keys.sort()
for k in keys:
	(cache, kernel, binary, file) = k
	if binary == './svm-train' and (cache, kernel, './svm-cuda-train', file) in time.keys():
		print kernel,
		print file,
		print cache, 
		print time[k],
		print time[(cache, kernel, './svm-cuda-train', file)],
		print time[k] / time[(cache, kernel, './svm-cuda-train', file)] 

#print time