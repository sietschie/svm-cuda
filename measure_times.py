import subprocess
import csv
import datetime


p = subprocess.Popen(["./svm-train", "-v",str(0), "data/a1a"], stdout = subprocess.PIPE)

def run(datafile, seriell=False, kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000):
	args = ["./svm-cuda-train", 
			"-t", str(kernel), 
			"-v", "0", 
			"-c", str(c), 
			"-g", str(gamma), 
			"-e", str(e), 
			"-i", str(i), 
			datafile]
	
	if seriell:
		args[0] = "./svm-train"
	
	#print 'seriell: ', seriell, 'args: ', args
	
	p = subprocess.Popen(args, stdout = subprocess.PIPE)
	
	return p.communicate()[0]

def runandwrite(datafile, outfile, seriell=False, kernel = 2, c=1, gamma=0.5, e=0.01, i=1000000):
	output = run(datafile, seriell = seriell, kernel = kernel, c=c, gamma=gamma, e=e, i=i)
	print output
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
	for kernel in [0,2]:
		for datafile in df_pathes:
			for seriell in [True, False]:
				print 'run kernel %d, datafile %s, repetition %d, seriell %s' % (kernel, datafile, repetition, seriell)
				runandwrite(datafile, w, seriell = seriell, kernel = kernel, c=c, gamma=gamma, e=e, i=i)

w.writerow([datetime.datetime.now()])
