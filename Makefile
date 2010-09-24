# Makefile im Top-Level-Verzeichnis

DIRS = cuda efficient predict

compile:
	for i in $(DIRS); do make -C $$i; done;

