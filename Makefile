# Makefile im Top-Level-Verzeichnis

DIRS = cuda common

compile:
	for i in $(DIRS); do make -C $$i; done;

