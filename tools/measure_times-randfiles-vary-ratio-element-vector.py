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

def runandwrite(datafile, outfile, binary='./svm-cuda-train', kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000, cache=10):
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
		
	row = [datafile, binary, time, iters, float(iters)/float(time), kernel, c, gamma, e, i, cache]
	outfile.writerow(row)

	
#datafiles = "a1a a2a a3a a4a breast-cancer_scale ionosphere_scale"

datafiles = 'rand-e512-v65536-1.data  rand-e1024-v32768-1.data  rand-e2048-v16384-1.data  rand-e4096-v8192-1.data rand-e16384-v2048-1.data  rand-e32768-v1024-1.data  rand-e8192-v4096-1.data'


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
w.writerow(['Filename', 'binary', 'time', 'iters', 'i/t', 'kernel', 'c', 'gamma', 'e', 'i', 'cache'])
w.writerow([])
w.writerow([datetime.datetime.now()])

for repetition in range(3):
	for datafile in df_pathes:
		for kernel in [2,0]:
			for binary in ['./svm-train', './svm-cuda-train-noeps']:
				print 'run binary %s, kernel %d, datafile %s, cache %d, repetition %d' % (binary, kernel, datafile, cache, repetition)
				runandwrite(datafile, w, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i, cache=cache)

w.writerow([datetime.datetime.now()])
