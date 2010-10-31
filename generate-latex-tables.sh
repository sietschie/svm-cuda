#!/bin/bash

#TODO: generate nicer filenames, "-d" = don't run the actual tests, just generate the tables from the results file


if [ "$1" != "-d" ]
then

	#generate random data
	cd tools
	./gen.py 1024 32768 varydensity
	./gen.py 2048 16384
	./gen.py 4096 8192
	./gen.py 8192 4096
	./gen.py 16384 2048
	./gen.py 32768 1024
	cd ..

	#run the experiments
	./tools/measure_times-normalfiles-c1-g05.py
	./tools/measure_times-normalfiles-libsvmbest.py
	./tools/measure_times-randfiles-vary-density.py
	./tools/measure_times-randfiles-vary-ratio-element-vector.py

fi

#condense data
./tools/refine_data.py resultsmeasure_times-normalfiles-c1-g05.py.csv
./tools/refine_data.py resultsmeasure_times-normalfiles-libsvmbest.py.csv
./tools/refine_data.py resultsmeasure_times-randfiles-vary-density.py.csv
./tools/refine_data.py resultsmeasure_times-randfiles-vary-ratio-element-vector.py.csv

#create latex tables
echo 'creating latextable: random data, varying density'
./tools/createlatextable-rand-density.py resultsmeasure_times-randfiles-vary-density.py.csv.refined.csv > tables.latex
echo >> tables.latex
echo >> tables.latex
echo >> tables.latex
echo 'creating latextable: random data, varying ratio'
./tools/createlatextable-rand-ratio.py resultsmeasure_times-randfiles-vary-ratio-element-vector.py.csv.refined.csv >> tables.latex
echo >> tables.latex
echo >> tables.latex
echo >> tables.latex
echo 'creating latextable: libsvm data, fixed parameters'
./tools/createlatextable-c1-g05.py resultsmeasure_times-normalfiles-c1-g05.py.csv.refined.csv >> tables.latex
echo >> tables.latex
echo >> tables.latex
echo >> tables.latex
echo 'creating latextable: libsvm data, libsvms best parameters'
./tools/createlatextable-libsvmbest.py resultsmeasure_times-normalfiles-libsvmbest.py.csv.refined.csv >> tables.latex



