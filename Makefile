# bpnn -- a backpropagation feed-forward network in C99, and tools for measuring
# how much of a neural-network experiment's result is real.
#
# Forked from SMBPANN (github.com/Anode1/SMBPANN) with the evolutionary search removed.
# The engine, its unit suite and the statistics tooling are carried over unchanged; the
# genetic algorithm, the co-evolved genome and the architecture-emergence probes are not.
# SMBPANN itself is untouched and remains the reference implementation for the paper that
# cites it.

CC      = cc
STD     = -std=c99
WARN    = -W -Wall
OPT     = -O2
CFLAGS  = $(STD) $(WARN) $(OPT)
LDLIBS  = -lm

ENGINE  = rng.o act.o net.o train.o arena.o data.o conv2f.o ckpt.o
PROG    = bpnn
WORKER  = bpnn_worker

.PHONY: all ut ut-asan ut-ubsan pedantic tools clean

all: $(PROG) $(WORKER)

# bpnn: the tabular predictor. Reads linearr's CSV, fits one network per group, writes a
# model, scores a case against it. This is the program the README is about.
$(PROG): bpnn.o $(ENGINE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# bpnn_worker: the older single-topology worker (main.c). It trains one named topology and
# prints a machine-readable RESULT line, which is what the removed search used to consume.
# Kept because the XOR demonstration and the checkpoint paths go through it.
$(WORKER): main.o $(ENGINE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard *.d)

# ---- the commit gate -------------------------------------------------------
ut: bpnn_ut
	./bpnn_ut
bpnn_ut: tests.c $(ENGINE:.o=.c)
	$(CC) $(CFLAGS) -DUNIT_TEST -o $@ $^ $(LDLIBS)

ut-asan:
	$(CC) $(STD) $(WARN) -O1 -g -fsanitize=address,undefined -DUNIT_TEST \
	  -o bpnn_ut_asan tests.c $(ENGINE:.o=.c) $(LDLIBS) && ./bpnn_ut_asan
ut-ubsan:
	$(CC) $(STD) $(WARN) -O1 -g -fsanitize=undefined -DUNIT_TEST \
	  -o bpnn_ut_ubsan tests.c $(ENGINE:.o=.c) $(LDLIBS) && ./bpnn_ut_ubsan
pedantic:
	$(CC) $(STD) -pedantic $(WARN) -Wextra -Wshadow -Wconversion -O2 -fsyntax-only \
	  $(ENGINE:.o=.c) main.c bpnn.c

# ---- measurement tools -----------------------------------------------------
# pairstat is the paired-statistics tool: Wilcoxon signed-rank (exact where ties permit),
# exact sign test, Hodges-Lehmann with distribution-free intervals, paired t, Holm across a
# declared family, and minimum detectable effect. It self-tests before it is trusted.
tools: resolve pairstat nb101_trials nb101_budget nb201_extract nb101_flip nb101_signal nb_ceiling validation/nb101_triples.txt
	./resolve selftest
	./pairstat --selftest

pairstat: validation/pairstat.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
nb101_trials: validation/nb101_trials.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f *.o *.d $(PROG) bpnn_ut bpnn_ut_asan bpnn_ut_ubsan \
	      pairstat nb101_trials nb101_signal nb101_flip conv2d fsdd_frame
nb101_flip: validation/nb101_flip.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ---- benchmark tables projected to the triples the flip probe reads --------
# NAS-Bench-101's archive table carries the graph as well; the measurement needs only the three
# per-run accuracies, so project them rather than teaching the probe a second format.
validation/nb101_triples.txt: validation/nasbench101_trials.txt
	awk '!/^#/{n=$$1; b=2+n+1; print $$(b+1), $$(b+3), $$(b+5)}' $< > $@
nb101_signal: validation/nb101_signal.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
nb201_extract: validation/nb201_extract.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
nb_ceiling: validation/nb_ceiling.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
nb101_budget: validation/nb101_budget.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
resolve: validation/resolve.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
