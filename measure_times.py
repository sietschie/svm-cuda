import subprocess
import csv
import datetime


p = subprocess.Popen(["./svm-train", "-v",str(0), "data/a1a"], stdout = subprocess.PIPE)

def run(datafile, binary='./svm-cuda-train', kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000):
	args = [binary, 
			"-t", str(kernel), 
			"-v", "0", 
			"-c", str(c), 
			"-g", str(gamma), 
			"-e", str(e), 
			"-i", str(i), 
			datafile]
	
	#print 'seriell: ', seriell, 'args: ', args
	
	p = subprocess.Popen(args, stdout = subprocess.PIPE)
	
	return p.communicate()[0]

def runandwrite(datafile, outfile, binary='./svm-cuda-train', kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000):
	output = run(datafile, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i)
	#print output
	ol = output.split(',')
	print ol
	time = ol[0].split()[1]
	iters = ol[1].split()[1]

	row = [datafile, seriell, time, iters, float(iters)/float(time), kernel, c, gamma, e, i]
	outfile.writerow(row)

	
datafiles = "a1a a2a a3a a4a breast-cancer_scale ionosphere_scale"
df_pathes = [ ''.join(["data/", file]) for file in datafiles.split()  ]

#seriell = False
kernel = 2
c = 1
gamma = 0.5
e=0.01
i=1000000

w = csv.writer(open('results.csv', 'w'), delimiter=';')
w.writerow([])
w.writerow([datetime.datetime.now()])
w.writerow(['Filename', 'seriell', 'time', 'iters', 'i/t', 'kernel', 'c', 'gamma', 'e', 'i'])

for repetition in range(10):
	for datafile in df_pathes:
		for kernel in [2,0]:
			for binary in ['./svm-train', './svm-cuda-train', './svm-cuda-train-float', './svm-cuda-train-doubles']:
				print 'run binary %s, kernel %d, datafile %s, repetition %d' % (binary, kernel, datafile, repetition)
				runandwrite(datafile, w, binary = binary, kernel = kernel, c=c, gamma=gamma, e=e, i=i)

w.writerow([datetime.datetime.now()])
