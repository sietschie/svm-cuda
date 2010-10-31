#!/usr/bin/python
import random
import sys

def getRandomVector(dim):
    v = []
    for i in xrange(dim):
        v.append((random.random(), random.random()))

    return v

def generateRandomSample(r):
    res = [random.gauss(x[0], x[1]) for x in r ]
    return res

def generateRandomSamples(r, number):
    samples = []
    for i in xrange(number):
        samples.append(generateRandomSample(r))
    return samples


def printSample(s, label, prob):
    stringlist = [label]
    for i in range(len(s)):
        if random.random() < prob:
            stringlist.append("%d:%f" % (i+1,s[i]))
    stringlist.append('\n')
    return " ".join(stringlist)

elements = int( sys.argv[1] )
vectors = int( int( sys.argv[2] ) / 2 )
	

filename = 'rand-e' + str(elements) + '-v' + str(2*vectors)

if len(sys.argv) >= 4 and sys.argv[3] == 'varydensity':
	prob_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,0.3,0.2, 0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001]
else:
	prob_list = [1]
	
print 'generating random data. Elements %d, Vectors: %d, list of probabilities: ' % (elements, vectors), prob_list

root1 = getRandomVector(elements)
samples1 = generateRandomSamples(root1,vectors)

for prob in prob_list:
	f = open(filename + '-' + str(prob) + '.data', 'w')
	for s in samples1:
		f.write(printSample(s, '-1', prob))
	f.close()
		
root2 = getRandomVector(elements)
samples2 = generateRandomSamples(root2,vectors)

for prob in prob_list:
	f = open(filename + '-' + str(prob) + '.data', 'a')
	for s in samples1:
		f.write(printSample(s, '+1', prob))
	f.close()

