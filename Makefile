# Copyright (c) 2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
#
# bpnn -- a backpropagation feed-forward network in C99, and the tools that measure
# how much of a result taken from it is real.
#
# Forked from SMBPANN (github.com/Anode1/SMBPANN) with the evolutionary search removed.
# The engine, its unit suite and the statistics tooling carried over; the genetic
# algorithm, the co-evolved genome and the architecture-emergence probes did not.
# SMBPANN itself is untouched and remains the reference implementation for the paper
# that cites it.
#
#   make | ut | ut-asan | ut-ubsan | pedantic | tools | clean
SHELL = /bin/sh

CC      = cc
STD     = -std=c99
WARN    = -W -Wall
OPT     = -O2
CFLAGS  = $(STD) $(WARN) $(OPT)
CPPFLAGS = -Ic
LDLIBS  = -lm

ENGINE  = c/rng.o c/act.o c/net.o c/train.o c/arena.o c/data.o c/conv2f.o c/ckpt.o
SRC     = $(ENGINE:.o=.c)
PROG    = bpnn
WORKER  = bpnn_worker

.PHONY: all ut ut-asan ut-ubsan pedantic tools clean

all: $(PROG) $(WORKER)

# bpnn: the tabular predictor. Reads linearr's CSV, fits one network per group, writes a
# model, scores a case against it. This is the program the README is about.
$(PROG): c/bpnn.o $(ENGINE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# bpnn_worker: the older single-topology worker (c/main.c). It trains one named topology
# and prints a machine-readable RESULT line, which is what the removed search consumed.
# Kept because the XOR demonstration and the checkpoint paths go through it.
$(WORKER): c/main.o $(ENGINE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

c/%.o: c/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard c/*.d)

# ---- the commit gate -------------------------------------------------------
ut: bpnn_ut
	./bpnn_ut
bpnn_ut: tests/tests.c $(SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DUNIT_TEST -o $@ $^ $(LDLIBS)

ut-asan:
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=address,undefined -DUNIT_TEST \
	  -o bpnn_ut_asan tests/tests.c $(SRC) $(LDLIBS) && ./bpnn_ut_asan
ut-ubsan:
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=undefined -DUNIT_TEST \
	  -o bpnn_ut_ubsan tests/tests.c $(SRC) $(LDLIBS) && ./bpnn_ut_ubsan

pedantic:
	$(CC) $(CPPFLAGS) $(STD) -pedantic $(WARN) -Wextra -Wshadow -Wconversion -O2 \
	  -fsyntax-only $(SRC) c/main.c c/bpnn.c

# ---- the measurement tools -------------------------------------------------
# Each is one self-contained .c with no engine dependency. resolve and pairstat refuse
# to be trusted until their self-tests pass, so the target runs both.
TOOLS = resolve pairstat nb101_trials nb101_budget nb101_flip nb101_signal nb_ceiling nb201_extract

tools: $(TOOLS)
	./resolve selftest
	./pairstat --selftest

$(TOOLS): %: validation/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# NAS-Bench-101's archive table carries the graph as well; the flip and ceiling probes need
# only the three per-run accuracies, so project them rather than teaching each probe a
# second format. The input is re-fetched per validation/PROVENANCE_nas.md.
validation/nb101_triples.txt: validation/nasbench101_trials.txt
	awk '!/^#/{n=$$1; b=2+n+1; print $$(b+1), $$(b+3), $$(b+5)}' $< > $@

clean:
	rm -f c/*.o c/*.d $(PROG) $(WORKER) $(TOOLS) \
	      bpnn_ut bpnn_ut_asan bpnn_ut_ubsan
