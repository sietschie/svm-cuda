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
	print '\\begin{tabular}{|l|c|c|c|}'
	print '\\hline'
	print 'Dichte & Host & Cuda & Speedup \\\\'
	print '\\hline'
	for row in data:
		#print row
		print getDensity(row), '&',
		print decimals(row['time_seriell'], 2), '&',
		print decimals(row['time_cuda'], 2), '&',
		print decimals(row['speedup'], 2), '\\\\'
	print '\\hline'
	print '\\end{tabular}'
	print '\\end{center}'
	print '\\caption{ 32768 Vektoren der Dimension 1024, c=1, Kernel: ' + kernel[data[0]['kernel']] + ', cachesize: ' + data[0]['cache'] + ', 100 Iterationen}'
	print '\\label{tbl:density-' + kernel[data[0]['kernel']] + '-' + data[0]['cache'] + '}'
	print '\\end{table}'

reader = csv.DictReader(open(sys.argv[1], 'r'), delimiter=';')

table = []

for row in reader:
	#print row
	key = ( row['cache'], row['kernel'])
	table.append(row)
	
	#print k
	#print tables[k]

def getDensity(l):
	res1 = l['Filename'].rsplit('-', 1)[-1]
	res2 = res1.rsplit('.', 1)[0]
	return float(res2)

def getDensityMinus(l):
	return -getDensity(l)
	
#for row in table:
	#print row
#	print getDensity(row)
	
sortedTable = sorted(table, key=getDensityMinus)
#createTable(table)

#for row in sortedTable:
	#print row
#	print getDensity(row)

createTable(sortedTable)