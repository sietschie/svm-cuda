# Makefile im Top-Level-Verzeichnis

DIRS = cuda efficient

compile:
	for i in $(DIRS); do make -C $$i; done;

