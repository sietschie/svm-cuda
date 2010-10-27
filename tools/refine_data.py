import csv

reader = csv.DictReader(open('results/results.csv', 'r'), delimiter=';')

times = {}
quotients = {}

for row in reader:
	if ( row['kernel'], row['seriell'], row['Filename'] ) not in times.keys():
		times[( row['kernel'], row['seriell'], row['Filename'] ) ] = []
		quotients[( row['kernel'], row['seriell'], row['Filename'] ) ] = []
	
	times[( row['kernel'], row['seriell'], row['Filename'] ) ].append(float(row['time']))
	quotients[( row['kernel'], row['seriell'], row['Filename'] ) ].append(float(row['i/t']))
	
time = {}
quotient = {}
	
for k in times.keys():
	time[k] = sum(times[k]) / len(times[k])
	quotient[k] = sum(quotients[k]) / len(quotients[k])
	
for k in time.keys():
	(kernel, seriell, file) = k
	print seriell
	if seriell == 'True' and (kernel, 'False', file) in time.keys():
		print time[(kernel, 'False', file)],
		print time[k]
	
print time