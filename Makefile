# Makefile im Top-Level-Verzeichnis

DIRS = efficient

compile:
	for i in $(DIRS); do make -C $$i; done;

