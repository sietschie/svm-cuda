# Makefile im Top-Level-Verzeichnis

DIRS = cuda 

compile:
	for i in $(DIRS); do make -C $$i; done;

