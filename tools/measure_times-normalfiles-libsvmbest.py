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
			"-g", str(gamma), 
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

	
datafiles = "a1a a2a a3a a4a breast-cancer_scale ionosphere_scale"
df_pathes = [ ''.join(["data/", file]) for file in datafiles.split()  ]

input_list = []
input_list[0] = ( df_pathes[0], 8192.0, 3.0517578125e-05 )
input_list[1] = ( df_pathes[1], 8.0, 0.03125 )
input_list[2] = ( df_pathes[2], 512.0, 0.00048828125 )
input_list[3] = ( df_pathes[3], 8192.0, 3.0517578125e-05 )
input_list[4] = ( df_pathes[4], 512.0, 0.0001220703125 )
input_list[5] = ( df_pathes[5], 2.0, 0.5 )

'''8192.0 3.0517578125e-05 83.8629
==> a2a.out <==
8.0 0.03125 82.8256
==> a3a.out <==
512.0 0.00048828125 83.8305
==> a4a.out <==
8192.0 3.0517578125e-05 84.1037
==> breast-cancer_scale.out <==
512.0 0.0001220703125 97.2182
==> ionosphere_scale.out <==
2.0 0.5 95.1567'''

#seriell = False
kernel = 2
#c = 1
#gamma = 0.5
e=0.01
i=1000000

basename = os.path.basename(sys.argv[0])

w = csv.writer(open('results' + basename + '.csv', 'w'), delimiter=';')
w.writerow(['Filename', 'binary', 'time', 'iters', 'i/t', 'kernel', 'c', 'gamma', 'e', 'i', 'cache'])
w.writerow([])
w.writerow([datetime.datetime.now()])

for repetition in range(3):
	for cache in [10, 1000]:
		for (datafile, c, gamma) in input_list:
			for binary in ['./svm-train', './svm-cuda-train']:
				print 'run binary %s, kernel %d, datafile %s, cache %d, repetition %d' % (binary, kernel, datafile, cache, repetition)
				runandwrite(datafile, w, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i, cache=cache)

w.writerow([datetime.datetime.now()])
