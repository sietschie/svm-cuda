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
	
root1 = getRandomVector(elements)
samples1 = generateRandomSamples(root1,vectors)

filename = 'rand-e' + str(elements) + '-v' + str(2*vectors)

f10 = open(filename + '-1.0.data','w')
f01 = open(filename + '-0.1.data','w')
f001 = open(filename + '-0.01.data','w')
for s in samples1:
	f10.write(printSample(s, '-1', 1))
for s in samples1:
    f01.write(printSample(s, '-1', 0.1))
for s in samples1:
    f001.write(printSample(s, '-1', 0.01))

root2 = getRandomVector(elements)
samples2 = generateRandomSamples(root2,vectors)
	
for s in samples2:
    f10.write(printSample(s, '+1', 1))
for s in samples2:
    f01.write(printSample(s, '+1', 0.1))
for s in samples2:
    f001.write(printSample(s, '+1', 0.01))

f10.close()
f01.close()
f001.close()

