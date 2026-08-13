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
#   make | check | ut | cliut | ut-asan | ut-ubsan | pedantic | tools | clean
SHELL = /bin/sh

# The version comes from the git tag, so there is one source for it. Override for a build from
# a tarball with no history: make BPNN_VERSION=x.y.z
ifeq ($(origin BPNN_VERSION), undefined)
BPNN_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//')
endif
ifeq ($(strip $(BPNN_VERSION)),)
BPNN_VERSION := 0.0.0-dev
endif

PREFIX  ?= /usr/local
DESTDIR ?=

CC      = cc
STD     = -std=c99
WARN    = -W -Wall
OPT     = -O2
CFLAGS  = $(STD) $(WARN) $(OPT)
CPPFLAGS = -Ic -DBPNN_VERSION='"$(BPNN_VERSION)"'
LDLIBS  = -lm

ENGINE  = c/rng.o c/act.o c/net.o c/train.o c/arena.o c/data.o c/conv2f.o c/ckpt.o
# The tabular program: one concept per file, sharing the table through tab.h.
TAB     = c/tab.o c/csvread.o c/fit.o c/stream.o c/model.o c/report.o
SRC     = $(ENGINE:.o=.c)
PROG    = bpnn
WORKER  = bpnn_worker

.PHONY: all check ut cliut ut-asan ut-ubsan pedantic tools install uninstall clean

all: $(PROG) $(WORKER)

# What must pass before a commit.
check: ut cliut

# bpnn: the tabular predictor. Reads linearr's CSV, fits one network per group, writes a
# model, scores a case against it. This is the program the README is about.
$(PROG): c/bpnn.o $(TAB) $(ENGINE)
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

# cliut: black-box tests of the built binary.
cliut: $(PROG)
	sh tests/cli.sh

# The suite AND the binary, because the CSV reader is only reached through the binary and it is
# the part that touches attacker-shaped input.
ut-asan:
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=address,undefined -DUNIT_TEST \
	  -o bpnn_ut_asan tests/tests.c $(SRC) $(LDLIBS) && ./bpnn_ut_asan
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=address,undefined \
	  -o bpnn_asan c/bpnn.c $(TAB:.o=.c) $(SRC) $(LDLIBS) && BPNN=$(PWD)/bpnn_asan sh tests/cli.sh
ut-ubsan:
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=undefined -DUNIT_TEST \
	  -o bpnn_ut_ubsan tests/tests.c $(SRC) $(LDLIBS) && ./bpnn_ut_ubsan
	$(CC) $(CPPFLAGS) $(STD) $(WARN) -O1 -g -fsanitize=undefined \
	  -o bpnn_ubsan c/bpnn.c $(TAB:.o=.c) $(SRC) $(LDLIBS) && BPNN=$(PWD)/bpnn_ubsan sh tests/cli.sh

pedantic:
	$(CC) $(CPPFLAGS) $(STD) -pedantic $(WARN) -Wextra -Wshadow -Wconversion -O2 \
	  -fsyntax-only $(SRC) $(TAB:.o=.c) c/main.c c/bpnn.c

# ---- the measurement tools -------------------------------------------------
# One self-contained .c each, no engine dependency. Both self-tests run before the tools are
# considered usable.
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

# The binary, and the example data it names in its own messages.
install: $(PROG)
	mkdir -p $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/share/bpnn/example
	cp $(PROG) $(DESTDIR)$(PREFIX)/bin/
	cp example/*.csv $(DESTDIR)$(PREFIX)/share/bpnn/example/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(PROG)
	rm -rf $(DESTDIR)$(PREFIX)/share/bpnn

clean:
	rm -f c/*.o c/*.d $(PROG) $(WORKER) $(TOOLS) \
	      bpnn_ut bpnn_ut_asan bpnn_ut_ubsan bpnn_asan bpnn_ubsan
