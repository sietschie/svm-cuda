#!/usr/bin/python
import sys
import csv

kernel = {'0':'linear', '2':'gauss'}

def getFileName(s):
	res1 = s.rsplit('/',1)[-1]
	res2 = res1.split('_')[0]
	return res2
	
def decimals(s,i):
	f = float(s)
	return '%.2f' % (f)

def createTable(data):
	print '\\begin{table}'
	print '\\begin{center}'
	print '\\begin{tabular}{|c|c|c|c|c|}'
	print '\\hline'
	print 'Dimension & Vectors & seriell & cuda & speedup \\\\'
	print '\\hline'
	for row in data:
		#print row
		(elem, vect) = getElementsNVectors(row)
		print elem, '&',
		print vect, '&',
		print decimals(row['time_seriell'], 2), '&',
		print decimals(row['time_cuda'], 2), '&',
		print decimals(row['speedup'], 2), '\\\\'
	print '\\hline'
	print '\\end{tabular}'
	print '\\end{center}'
	print '\\caption{ c=1, Kernel: ' + kernel[data[0]['kernel']] + ', cachesize: ' + data[0]['cache'] + ', 100 Iterationen}'
	print '\\end{table}'

reader = csv.DictReader(open(sys.argv[1], 'r'), delimiter=';')

#table = []

#for row in reader:
	#print row
#	key = ( row['cache'], row['kernel'])
#	table.append(row)
	
	#print k
	#print tables[k]

def getDensity(l):
	res1 = l['Filename'].rsplit('-', 1)[-1]
	res2 = res1.rsplit('.', 1)[0]
	return float(res2)

def getDensityMinus(l):
	return -getDensity(l)
	
def getElementsNVectors(l):
	res1 = l['Filename'].split('-')
	selem = res1[1]
	elem = selem[1:len(selem)]
	svect = res1[2]
	vect = svect[1:len(svect)]
	return (int(elem), int(vect))
	
tables = {}

for row in reader:
	#print row
	key = ( row['kernel'])
	if key not in tables.keys():
		tables[key] = []
	tables[key].append(row)

for k in tables:
	table = tables[k]
	#print k
	#print k
	#print tables[k]
	#createTable(tables[k])

	
	#for row in table:
	#	print row['Filename']
	#	print getElementsNVectors(row)
	#	print getDensity(row)
		
	sortedTable = sorted(table, key=getElementsNVectors)
	#createTable(table)
	#print 'sorted'
	#for row in sortedTable:
	#	print row['Filename']
	#	print getElementsNVectors(row)
		#print row
	#	print getDensity(row)

	createTable(sortedTable)