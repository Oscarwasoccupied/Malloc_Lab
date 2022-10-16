LAB = malloclab
LAB_CHECKPOINT = malloclabcheckpoint
LAB_FINAL = malloclab
COURSECODE = 18213-f22
SAN_LIBRARY_PATH = /afs/cs.cmu.edu/academic/class/15213/lib/
ifneq (,$(wildcard autograde-lib/))
  SAN_LIBRARY_PATH = autograde-lib/
endif
