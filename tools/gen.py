import random

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

root1 = getRandomVector(10)
samples1 = generateRandomSamples(root1,1000)

root2 = getRandomVector(10)
samples2 = generateRandomSamples(root2,1000)

filename = 'tmpfile5'

f = open(filename + '-1.0','w')
for s in samples1:
    f.write(printSample(s, '-1', 1))
for s in samples2:
    f.write(printSample(s, '+1', 1))
f.close()


f = open(filename + '-0.1','w')
for s in samples1:
    f.write(printSample(s, '-1', 0.1))
for s in samples2:
    f.write(printSample(s, '+1', 0.1))
f.close()


f = open(filename + '-0.01','w')
for s in samples1:
    f.write(printSample(s, '-1', 0.01))
for s in samples2:
    f.write(printSample(s, '+1', 0.01))
f.close()

