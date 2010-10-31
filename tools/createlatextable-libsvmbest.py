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
	print '\\begin{tabular}{|l|c|c|c|c|c|}'
	print '\\hline'
	print 'dataset & C & $\gamma$ &seriell & cuda & speedup \\\\'
	print '\\hline'
	for row in data:
		#print row
		print getFileName(row['Filename']), '&',
		print row['c'], '&',
		print row['gamma'], '&',
		print decimals(row['time_seriell'], 2), '&',
		print decimals(row['time_cuda'], 2), '&',
		print decimals(row['speedup'], 2), '\\\\'
	print '\\hline'
	print '\\end{tabular}'
	print '\\end{center}'
	print '\\caption{ Kernel: ' + kernel[data[0]['kernel']] + ', cachesize: ' + data[0]['cache'] + '}'
	print '\\end{table}'

reader = csv.DictReader(open(sys.argv[1], 'r'), delimiter=';')

tables = {}

for row in reader:
	#print row
	key = ( row['cache'], row['kernel'])
	if key not in tables.keys():
		tables[key] = []
	tables[key].append(row)
	
for k in tables:
	print
	#print k
	#print tables[k]
	createTable(tables[k])
