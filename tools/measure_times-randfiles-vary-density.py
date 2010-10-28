#!/usr/bin/python
import subprocess
import csv
import datetime
import sys
import os.path


p = subprocess.Popen(["./svm-train", "-v",str(0), "data/a1a"], stdout = subprocess.PIPE)

def run(datafile, binary='./svm-cuda-train', kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000, cache=10):
	args = [binary, 
			"-t", str(kernel), 
			"-v", "0", 
			"-c", str(c), 
#			"-g", str(gamma), 
			"-e", str(e), 
			"-i", str(i), 
			"-m", str(cache),
			datafile]
	
	#print 'seriell: ', seriell, 'args: ', args
	
	p = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	
	return p.communicate()

def runandwrite(datafile, outfile, binary='./svm-cuda-train', kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000, cache=10, density=999):
	(output, error) = run(datafile, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i, cache=cache)
	#print output
	if error:
		print error
		time = '1'
		iters = '1'
	else:
		try:
			ol = output.split(',')
			print ol
			time = ol[0].split()[1]
			iters = ol[1].split()[1]

		except IndexError:
			time = '123456789.0123455678'
			iters = '1234567890'
		
	row = [datafile, binary, time, iters, float(iters)/float(time), kernel, c, gamma, e, i, cache, density]
	outfile.writerow(row)

	
#datafiles = "a1a a2a a3a a4a breast-cancer_scale ionosphere_scale"

datafiles = 'rand-e1024-v32768-1.0.data  rand-e2048-v16384-1.0.data  rand-e4096-v8192-1.0.data rand-e16384-v2048-1.0.data  rand-e32768-v1024-1.0.data  rand-e8192-v4096-1.0.data'
datafiles = '''rand-e1024-v32768-0.0001.data  rand-e1024-v32768-0.04.data  rand-e1024-v32768-0.1.data  rand-e1024-v32768-0.75.data
rand-e1024-v32768-0.001.data   rand-e1024-v32768-0.05.data  rand-e1024-v32768-0.2.data  rand-e1024-v32768-0.7.data
rand-e1024-v32768-0.005.data   rand-e1024-v32768-0.06.data  rand-e1024-v32768-0.3.data  rand-e1024-v32768-0.8.data
rand-e1024-v32768-0.01.data    rand-e1024-v32768-0.07.data  rand-e1024-v32768-0.4.data  rand-e1024-v32768-0.9.data
rand-e1024-v32768-0.02.data    rand-e1024-v32768-0.08.data  rand-e1024-v32768-0.5.data  rand-e1024-v32768-1.data
rand-e1024-v32768-0.03.data    rand-e1024-v32768-0.09.data  rand-e1024-v32768-0.6.data
'''

df_pathes = [ ''.join(["tools/", file]) for file in datafiles.split()  ]



#seriell = False
kernel = 2
c = 1
gamma = 0.5
e=0.01
i=100
cache=10

basename = os.path.basename(sys.argv[0])

w = csv.writer(open('results' + basename + '.csv', 'w'), delimiter=';')
w.writerow(['Filename', 'binary', 'time', 'iters', 'i/t', 'kernel', 'c', 'gamma', 'e', 'i', 'cache', 'density'])
w.writerow([])
w.writerow([datetime.datetime.now()])

for repetition in range(3):
	for datafile in df_pathes:
		density = datafile.split('-')[-1].rsplit('.',1)[0]
		#for kernel in [2,0]:
		for binary in ['./svm-train', './svm-cuda-train-noeps']:
			print 'run binary %s, kernel %d, datafile %s, repetition %d, density %s' % (binary, kernel, datafile, repetition, density)
			runandwrite(datafile, w, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i, cache=cache, density=density)

w.writerow([datetime.datetime.now()])
