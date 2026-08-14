/* tab.h -- the fitted table: the groups, the rows they came from, and the scaling.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 *
 * One Group per group code in the input. The row store holds every row when the default path is
 * fitting; --stream leaves it empty and works from a cache instead. Everything here is shared
 * state, which is why it has a header of its own: the reader fills it, the fitters consume it,
 * the model writer serialises it and the report prints it.
 */
#ifndef BPNN_TAB_H
#define BPNN_TAB_H

#include <stdio.h>
#include <stdint.h>
#include "common.h"
#include "net.h"

#define MAXTERM   64
#define MAXGROUP  512
#define NAMELEN   64
#define LINELEN   SMB_LINE_MAX
#define Z95       1.959964

/* Checked every epoch. At 25 the first look already came after the optimum on the example data
 * -- every reported epoch count was a multiple of 25 and none was below 225 -- so the interval
 * was deciding the answer. The check reads the stopping rows only, an eighth of the file, and
 * early stopping saves far more than it costs. */
#define CHECK 1

/* Target band for the sigmoid output. 0 and 1 are unreachable and just saturate. */
#define TLO       0.1
#define THI       0.9

typedef struct {
    char   name[NAMELEN];
    long   first;                    /* index of its first row in the row store */
    long   n;
    double lo[MAXTERM], hi[MAXTERM]; /* per-term training range */
    double tlo, thi;                 /* response range */
    double ymean, ym2, ysd;          /* Welford state, then the response's own spread */
    double gysd;                     /* ysd over the WHOLE group, which the advisories quote */
    double hsd;                      /* spread of the response over the reported rows only */
    Net   *net;
    double train_rmse, held_rmse, run_sd, best_held, shipped_held;
    long   nseed, nheld, epochs_ran, nseen;
    double tail;                     /* how far the response reaches past its own quartiles */
    int    flat;                     /* the response never varied in this group */
} Group;

/* The row store, when there is one. Stride is the file's term count, not MAXTERM. */
#define ROW(i)  (xs + (size_t)(i) * (size_t)nterm)

extern Group  gp[MAXGROUP];
extern long   ngroup;
extern char   term[MAXTERM][NAMELEN];
extern char   response[NAMELEN];
extern long   nterm;
extern double *xs, *ys;
extern long   *rowgrp;
extern long    nrow, caprow;

/* Set once from the command line, read everywhere. */
extern long   hidden, epochs, nseed, patience, minrows, bufrows;
extern double rate, momentum, holdout, decay;
extern int    activation, streaming;
extern const char *cachepath;
extern const char *refitpath;
extern const char *ycol;        /* -y: the column to predict, when it is not column 2 */   /* --per-refit: where the per-refit errors are written */
extern const char *inpath;      /* what a message names; set when a file is opened */

long   group_of(const char *name);
int    grow(long want);
double rowcost(void);
void   oom_rows(const char *doing);
int    cmpd(const void *a, const void *b);
int    sigdigits(double v, double span);
double explained(const Group *g);

void     range_init(Group *g);
void     range_add(Group *g, const double *x, double y);
void     range_done(Group *g);
void     ranges(Group *g);
/* The ranges of the rows at positions [a, b) of ORD: the scaling of a fit that must not see the
 * rows it will be reported on. */
void     ranges_of(Group *g, const long *ord, long a, long b);
void     scale_in(const Group *g, const double *raw, smb_real *out);
smb_real scale_out(const Group *g, double t);
double   unscale_out(const Group *g, double a);

/* csvread.c: one parsed row at a time, so both fitting paths read the same file the same way. */
typedef struct { FILE *f; const char *path; long lineno; int header; } Reader;
int  reader_open(Reader *rd, const char *path);
void reader_close(Reader *rd);
int  reader_row(Reader *rd, long *grp, double *y, double *x);
int  number(const char *s, const char *what, long line, long col, double *out);
int  split_csv(char *line, char **f, int maxf);
int  read_csv(const char *path);
int  regroup(void);

/* fit.c. train_one and rmse are here because the streaming fit and the self-test both need
 * them; keeping one copy is what stops the two fitting paths drifting apart again. */
int  fit_group(Group *g);
Net *train_one(Group *g, long *ord, long ntr, long nstop, uint32_t seed, long *epochs_out);
double rmse(Group *g, Net *net, const long *ord, long a, long b);
void net_copy_weights(Net *dst, const Net *src);
int  fit_stream(const char *path);  /* stream.c */
void write_model(void);         /* model.c  */
int  read_model(const char *path);
void reading_line(const char *path);
void refit_open(void);          /* report.c */
void refit_group(const Group *g, const double *held, const double *train, const long *ran);
void refit_close(void);
void report(void);
int  footprint(long terms, long groups);

#endif /* BPNN_TAB_H */
