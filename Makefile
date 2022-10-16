#
# Makefile for the malloc lab driver
#
SHELL = /bin/bash
DOC = doxygen

# C Compiler
CLANG = clang
LLVM_PATH = /usr/local/depot/llvm-7.0/bin/
CC = $(LLVM_PATH)$(CLANG)

ifneq (,$(wildcard /usr/lib/llvm-7/bin/))
  LLVM_PATH = /usr/lib/llvm-7/bin/
endif

# Additional flags used to compile mdriver-dbg
# You can edit these freely to change how your debug binary compiles.
COPT_DBG = -O0
CFLAGS_DBG = -DDEBUG=1

# Flags used to compile normally
COPT = -O3
CFLAGS = $(COPT) -g \
         -Wall -Wextra -Werror -Wshorten-64-to-32 \
         -Wno-unused-function -Wno-unused-parameter

# Build configuration
FILES = mdriver mdriver-dbg mdriver-emulate mdriver-uninit
LDLIBS = -lm -lrt

MC = ./macro-check.pl
MCHECK = $(MC) -i dbg_

# Default rule
.PHONY: all
all: inst $(FILES)

objs:
	mkdir -p $@

###########################################################
# Driver programs
###########################################################

# General rules
DRIVERS = mdriver mdriver-dbg mdriver-emulate mdriver-uninit
$(DRIVERS):
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

REF_DRIVERS = mdriver-ref mdriver-cp-ref
$(REF_DRIVERS):
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	strip $@

# Object files
mdriver:         objs/mdriver.o        objs/mm-native.o     objs/memlib.o
mdriver-dbg:     objs/mdriver.o        objs/mm-native-dbg.o objs/memlib-asan.o
mdriver-emulate: objs/mdriver-sparse.o objs/mm-emulate.o    objs/memlib.o
mdriver-uninit:  objs/mdriver-msan.o   objs/mm-msan.o       objs/memlib-msan.o
mdriver-ref:     objs/mdriver-ref.o    objs/mm-ref.o        objs/memlib.o
mdriver-cp-ref:  objs/mdriver-ref.o    objs/mm-cp-ref.o     objs/memlib.o
$(DRIVERS) $(REF_DRIVERS): objs/fcyc.o objs/clock.o objs/stree.o

###########################################################
# Macro check script
###########################################################

.PHONY: mm-check
mm-check: mm.c $(MC)
	$(MCHECK) -f $<

###########################################################
# mm.c object files
###########################################################

# General rule
MM_OBJS = objs/mm-native.o objs/mm-native-dbg.o \
          objs/mm-ref.o objs/mm-cp-ref.o
$(MM_OBJS):
	$(CC) $(CFLAGS) -c -o $@ $<

# Rules for instrumented emulate driver
# Note: -O3 is necessary for the final step.
MM_EMULATE_OBJS = objs/mm-emulate.o objs/mm-msan.o

objs/mm-emulate.o:
	$(LLVM_PATH)$(CLANG) $(CFLAGS) -emit-llvm -S -o objs/mm.ll $<
	$(LLVM_PATH)opt -load=inst/MLabInst.so -MLabInst -o objs/mm_ct.bc objs/mm.ll
	$(CC) -O3 -c -o $@ objs/mm_ct.bc

objs/mm-msan.o:
	$(LLVM_PATH)$(CLANG) $(CFLAGS) -emit-llvm -S -o objs/mm-msan.ll $<
	$(LLVM_PATH)opt -load=inst/MLabInst2.so -MLabInst -o objs/mm_ct-msan.bc objs/mm-msan.ll
	$(CC) $(COPT) -c -o $@ objs/mm_ct-msan.bc

# Source files
objs/mm-native.o: mm.c
objs/mm-native-dbg.o: mm.c
objs/mm-emulate.o: mm.c | inst
objs/mm-msan.o: mm.c | inst
objs/mm-ref.o: $(MM-REF)
objs/mm-cp-ref.o: $(MM-CP-REF)

# Header files
$(MM_OBJS) $(MM_EMULATE_OBJS): mm.h memlib.h | objs mm-check

# Updated flags
$(MM_OBJS) $(MM_EMULATE_OBJS): CFLAGS += -DDRIVER
objs/mm-native-dbg.o: COPT = $(COPT_DBG)
objs/mm-native-dbg.o: CFLAGS += $(CFLAGS_DBG)
objs/mm-emulate.o: CFLAGS += -fno-vectorize
objs/mm-msan.o: COPT = -Og
objs/mm-msan.o: CFLAGS += -fno-inline -fno-optimize-sibling-calls -fno-omit-frame-pointer

###########################################################
# mdriver.c object files
###########################################################

# General rule
MDRIVER_OBJS = objs/mdriver.o objs/mdriver-sparse.o objs/mdriver-msan.o \
               objs/mdriver-ref.o
$(MDRIVER_OBJS):
	$(CC) $(CFLAGS) -o $@ -c $<

# Source files
$(MDRIVER_OBJS): mdriver.c

# Header files
$(MDRIVER_OBJS): fcyc.h clock.h memlib.h config.h mm.h stree.h | objs

# Updated flags
$(MDRIVER_OBJS): CFLAGS += -DDRIVER
objs/mdriver-sparse.o: CFLAGS += -DSPARSE_MODE
objs/mdriver-ref.o: CFLAGS += -DREF_ONLY

###########################################################
# memlib.c object files
###########################################################

# General rule
MEMLIB_OBJS = objs/memlib.o objs/memlib-asan.o objs/memlib-msan.o
$(MEMLIB_OBJS):
	$(CC) $(CFLAGS) -o $@ -c $<

# Source files
$(MEMLIB_OBJS): memlib.c

# Header files
$(MEMLIB_OBJS): memlib.h | objs

# Updated flags
$(MEMLIB_OBJS): CFLAGS += -DNO_CHECK_UB

###########################################################
# Other object files
###########################################################

# General rule
OTHER_OBJS = objs/fcyc.o objs/clock.o objs/stree.o
$(OTHER_OBJS):
	$(CC) $(CFLAGS) -o $@ -c $<

# Source files
objs/fcyc.o: fcyc.c
objs/clock.o: clock.c
objs/stree.o: stree.c

# Header files
objs/fcyc.o: fcyc.h
objs/clock.o: clock.h
objs/stree.o: stree.h
$(OTHER_OBJS): | objs

###########################################################
# Interpositioning library
###########################################################

mm.so: mm.c memlib-passthrough.c
	$(CC) -O2 -fPIC -shared -o $@ $^

###########################################################
# Other rules
###########################################################

.PHONY: clean
clean:
	rm -f *~
	rm -f $(FILES)
	rm -rf objs/


.PHONY: doc
doc: doxygen.conf mm.c mm.h memlib.h
	$(DOC) $<


# Include rules for submit, format, etc
FORMAT_FILES = mm.c
HANDIN_FILES = mm.c
include helper.mk


# Compile certain targets with sanitizers
SAN_TARGETS = mdriver-dbg
SAN_OBJS = objs/mm-native-dbg.o objs/memlib-asan.o
SAN_FLAGS = -fsanitize=address,undefined -DUSE_ASAN

# Add compiler flags
$(SAN_OBJS): CFLAGS += $(SAN_FLAGS)
$(SAN_OBJS): CFLAGS += -I$(SAN_LIBRARY_PATH)clang/7.0.0/include

# Hacks for loading archive properly
ifneq (,$(wildcard $(SAN_LIBRARY_PATH)))
  $(SAN_TARGETS): LDFLAGS += \
    -Wl,--whole-archive,$(SAN_LIBRARY_PATH)clang/7.0.0/lib/linux/libclang_rt.asan-x86_64.a \
    -Wl,--no-whole-archive \
    -Wl,--export-dynamic \
    -Wl,--no-as-needed
  $(SAN_TARGETS): LDLIBS += -lpthread -lrt -lm -ldl
else
  $(SAN_TARGETS): LDLIBS += $(SAN_FLAGS)
endif


# Compile certain targets with MemorySanitizer
MSAN_TARGETS = mdriver-uninit
MSAN_OBJS = objs/mm-msan.o objs/mdriver-msan.o objs/memlib-msan.o
MSAN_FLAGS = -fsanitize=memory -fsanitize-memory-track-origins -DUSE_MSAN

# Add compiler flags
$(MSAN_OBJS): CFLAGS += $(MSAN_FLAGS)
$(MSAN_OBJS): CFLAGS += -I$(SAN_LIBRARY_PATH)clang/7.0.0/include

# Linking memsan
ifneq (,$(wildcard $(SAN_LIBRARY_PATH)))
  $(MSAN_TARGETS): LDFLAGS += \
    -Wl,--whole-archive,$(SAN_LIBRARY_PATH)clang/7.0.0/lib/linux/libclang_rt.msan-x86_64.a \
    -Wl,--no-whole-archive \
    -Wl,--export-dynamic \
    -Wl,--no-as-needed
  $(MSAN_TARGETS): LDLIBS += -lpthread -lrt -lm -ldl
else
  $(MSAN_TARGETS): LDLIBS += $(MSAN_FLAGS)
endif


# Add check-format dependencies
submit: | check-format
$(FILES): | check-format
