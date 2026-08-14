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

# The standard variables are yours; the project's own flags are appended, never substituted for
# them. Setting CFLAGS used to drop -std=c99, every warning, and the version stamp.
CC       ?= cc
CFLAGS   ?= -O2
CPPFLAGS ?=
LDLIBS   ?=

# No fused multiply-add contraction. a*b+c computed as one FMA is more accurate than the two
# roundings the source asks for, and it is not what another compiler or another architecture
# does. This program publishes model files and claims they reproduce byte for byte; that is only
# true if the arithmetic is the arithmetic in the source. gcc's default is `fast` even under
# -std=c99, and x86-64 baseline hides it only because SSE2 has no FMA: build with -march=native
# on a recent chip, or anywhere on arm64, and the weights move in the ninth digit.
STD  = -std=c99 -ffp-contract=off
WARN = -W -Wall -Wshadow -Wconversion
PROJ = $(STD) $(WARN) -Ic -DBPNN_VERSION='"$(BPNN_VERSION)"' -DSMB_VERSION='"$(BPNN_VERSION)"'
LIBM = -lm

# Every header is a prerequisite of every target that compiles sources in one shot: those have no
# .d files, so a header edit otherwise leaves the suite testing a stale binary. A deliberately
# broken c/net.h passed `make ut` before this line existed.
HEADERS = $(wildcard c/*.h)

# Defined here rather than beside its rules: a prerequisite is expanded where it is written, so
# `cliut: $(TOOLS)` silently expanded to nothing while the variable was declared further down,
# and the suite skipped seven checks it claimed to run.
TOOLS = resolve pairstat

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
	$(CC) $(PROJ) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBM) $(LDLIBS)

# bpnn_worker: the older single-topology worker (c/main.c). It trains one named topology
# and prints a machine-readable RESULT line, which is what the removed search consumed.
# Kept because the XOR demonstration and the checkpoint paths go through it.
$(WORKER): c/main.o $(ENGINE)
	$(CC) $(PROJ) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBM) $(LDLIBS)

c/%.o: c/%.c
	$(CC) $(PROJ) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard c/*.d)

# ---- the commit gate -------------------------------------------------------
ut: bpnn_ut
	./bpnn_ut
bpnn_ut: tests/tests.c $(SRC) $(HEADERS)
	$(CC) $(PROJ) $(CPPFLAGS) $(CFLAGS) -DUNIT_TEST -o $@ tests/tests.c $(SRC) $(LIBM) $(LDLIBS)

# cliut: black-box tests of the built binary.
cliut: $(PROG) $(TOOLS)
	sh tests/cli.sh

# The suite AND the binary, because the CSV reader is only reached through the binary and it is
# the part that touches attacker-shaped input.
# -fno-sanitize-recover is what makes these gates. Without it UBSan prints its finding and exits
# 0, so a live report scrolled past and the target announced success.
SAN = -O1 -g -fno-sanitize-recover=undefined

ut-asan: $(HEADERS)
	$(CC) $(PROJ) $(CPPFLAGS) $(SAN) -fsanitize=address,undefined -DUNIT_TEST \
	  -o bpnn_ut_asan tests/tests.c $(SRC) $(LIBM) && ./bpnn_ut_asan
	$(CC) $(PROJ) $(CPPFLAGS) $(SAN) -fsanitize=address,undefined \
	  -o bpnn_asan c/bpnn.c $(TAB:.o=.c) $(SRC) $(LIBM) && BPNN=$(CURDIR)/bpnn_asan sh tests/cli.sh
ut-ubsan: $(HEADERS)
	$(CC) $(PROJ) $(CPPFLAGS) $(SAN) -fsanitize=undefined -DUNIT_TEST \
	  -o bpnn_ut_ubsan tests/tests.c $(SRC) $(LIBM) && ./bpnn_ut_ubsan
	$(CC) $(PROJ) $(CPPFLAGS) $(SAN) -fsanitize=undefined \
	  -o bpnn_ubsan c/bpnn.c $(TAB:.o=.c) $(SRC) $(LIBM) && BPNN=$(CURDIR)/bpnn_ubsan sh tests/cli.sh

# Compiled, not -fsyntax-only: -Wmaybe-uninitialized, -Wformat-truncation and -Warray-bounds
# exist only after the optimiser runs, and three gcc versions had something to say that this
# target could not see. Every source in the tree, including the tools and the suite.
pedantic: $(HEADERS)
	@rc=0; tmp=`mktemp -d`; \
	for f in $(SRC) $(TAB:.o=.c) c/main.c c/bpnn.c tests/tests.c validation/*.c; do \
	    $(CC) $(PROJ) -pedantic -Wextra -O2 -c "$$f" -o "$$tmp/p.o" || rc=1; \
	done; \
	rm -rf "$$tmp"; \
	test $$rc -eq 0 && echo "pedantic: clean"; exit $$rc

# ---- the measurement tools -------------------------------------------------
# One self-contained .c each, no engine dependency. Both self-tests run before the tools are
# considered usable.

tools: $(TOOLS)
	./resolve selftest
	./pairstat --selftest

$(TOOLS): %: validation/%.c
	$(CC) $(PROJ) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBM) $(LDLIBS)

# The binary, and the example data it names in its own messages.
# Quoted, and refusing an empty PREFIX: `make uninstall PREFIX=` printed `rm -rf /share/bpnn`.
# The example files are enumerated rather than globbed, so a model written into example/ by
# scripts/escalate.sh is not installed as if it were example data.
EXAMPLES = example/nonlinear.csv example/interaction.csv \
           example/elos-linear.csv example/elos-saturating.csv example/elos-interaction.csv

install: $(PROG)
	@test -n "$(PREFIX)" || { echo "PREFIX is empty; refusing to install into /bin" >&2; exit 1; }
	mkdir -p "$(DESTDIR)$(PREFIX)/bin" "$(DESTDIR)$(PREFIX)/share/bpnn/example"
	cp $(PROG) "$(DESTDIR)$(PREFIX)/bin/"
	cp $(EXAMPLES) README.md LICENSE "$(DESTDIR)$(PREFIX)/share/bpnn/"
	cp $(EXAMPLES) "$(DESTDIR)$(PREFIX)/share/bpnn/example/"

uninstall:
	@test -n "$(PREFIX)" || { echo "PREFIX is empty; refusing to remove /share" >&2; exit 1; }
	rm -f "$(DESTDIR)$(PREFIX)/bin/$(PROG)"
	rm -rf "$(DESTDIR)$(PREFIX)/share/bpnn"

clean:
	rm -f c/*.o c/*.d $(PROG) $(WORKER) $(TOOLS) \
	      bpnn_ut bpnn_ut_asan bpnn_ut_ubsan bpnn_asan bpnn_ubsan
