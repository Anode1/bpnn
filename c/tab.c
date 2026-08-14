/* tab.c -- the group table, the row store, and the scaling they share.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "act.h"
#include "tab.h"

Group  gp[MAXGROUP];
long   ngroup;
char   term[MAXTERM][NAMELEN];
char   response[NAMELEN];
long   nterm;

double *xs, *ys;
long   *rowgrp;
long    nrow, caprow;

long   hidden = 6, epochs = 3000, nseed = 5, patience = 50, minrows = 24, bufrows = 65536;
double rate = 0.3, momentum = 0.9, holdout = 0.25, decay;
int    activation = ACT_TANH, streaming;
const char *cachepath;
const char *refitpath;
const char *ycol;
const char *inpath = "-";

long group_of(const char *name)
{
    long i;
    for (i = 0; i < ngroup; i++)
        if (!strcmp(gp[i].name, name)) return i;
    if (ngroup >= MAXGROUP) return -1;
    memset(&gp[ngroup], 0, sizeof gp[ngroup]);
    strncpy(gp[ngroup].name, name, NAMELEN - 1);
    return ngroup++;
}

int grow(long want)
{
    double *nx, *ny; long *ng;
    if (want <= caprow) return 0;
    caprow = caprow ? caprow * 2 : 1024;
    while (caprow < want) caprow *= 2;
    nx = realloc(xs, (size_t)caprow * (size_t)nterm * sizeof *xs);
    ny = realloc(ys, (size_t)caprow * sizeof *ys);
    ng = realloc(rowgrp, (size_t)caprow * sizeof *ng);
    if (!nx || !ny || !ng) { xs = nx ? nx : xs; ys = ny ? ny : ys; rowgrp = ng ? ng : rowgrp; return -1; }
    xs = nx; ys = ny; rowgrp = ng;
    return 0;
}

/* Bytes a row costs the default path: the terms, the response, and the group index. */
double rowcost(void)
{
    return ((double)nterm + 1.0) * (double)sizeof(double) + (double)sizeof(long);
}

/* On running out, print the row count, the per-row cost, and mention --stream. */
void oom_rows(const char *doing)
{
    fprintf(stderr, "bpnn: out of memory %s: %ld rows at %.0f bytes each is %.3g GB.\n"
                    "--stream fits the same model without holding the rows, and\n"
                    "./bpnn --footprint %ld %ld prints both figures for this shape.\n",
            doing, nrow, rowcost(), rowcost() * (double)nrow / 1073741824.0,
            nterm, ngroup > 0 ? ngroup : 1);
}

/* ------------------------------------------------------------------ scaling */

/* The range of every term and of the response, which is all the scaling needs. Split into three
 * so that the streaming path can accumulate it in one pass without holding a row: this is the
 * one part of a network's fit that has a fixed-size sufficient statistic, exactly as a least
 * squares fit does, and it is the reason the first pass can be a pass and not a load. */
void range_init(Group *g)
{
    long j;
    for (j = 0; j < nterm; j++) { g->lo[j] = 1e300; g->hi[j] = -1e300; }
    g->tlo = 1e300; g->thi = -1e300;
    g->ymean = g->ym2 = 0;
    g->nseen = 0;
}

void range_add(Group *g, const double *x, double y)
{
    long j;
    for (j = 0; j < nterm; j++) {
        if (x[j] < g->lo[j]) g->lo[j] = x[j];
        if (x[j] > g->hi[j]) g->hi[j] = x[j];
    }
    if (y < g->tlo) g->tlo = y;
    if (y > g->thi) g->thi = y;
    /* Welford, not sum-of-squares minus square-of-sum: the response here is expected to carry
     * a large offset, and the textbook form loses every digit of the variance at 1e10. */
    {
        double d = y - g->ymean;
        g->ymean += d / (double)(g->nseen + 1);
        g->ym2   += d * (y - g->ymean);
        g->nseen++;
    }
}

void range_done(Group *g)
{
    long j;
    for (j = 0; j < nterm; j++) if (g->hi[j] <= g->lo[j]) g->hi[j] = g->lo[j] + 1.0; /* constant term */
    /* A response that never varied would divide by zero here. Widening the range keeps the
     * arithmetic finite and makes the fit look perfect, so the fact is recorded and reported. */
    g->flat = g->thi <= g->tlo;
    if (g->flat) g->thi = g->tlo + 1.0;
    /* The spread of the response is the yardstick a fitted error is worth reading against:
     * an error the size of it means the model does no better than the group's own mean. */
    if (g->nseen > 1) {
        double var = g->ym2 / (double)(g->nseen - 1);
        g->ysd = var > 0 ? sqrt(var) : 0.0;
    }
}

void ranges(Group *g)
{
    long i;
    range_init(g);
    for (i = g->first; i < g->first + g->n; i++)
        range_add(g, ROW(i), ys[i]);
    range_done(g);
}

void ranges_of(Group *g, const long *ord, long a, long b)
{
    long i;
    range_init(g);
    for (i = a; i < b; i++)
        range_add(g, ROW(g->first + ord[i]), ys[g->first + ord[i]]);
    range_done(g);
}

void scale_in(const Group *g, const double *raw, smb_real *out)
{
    long j;
    for (j = 0; j < nterm; j++)
        out[j] = (smb_real)((raw[j] - g->lo[j]) / (g->hi[j] - g->lo[j]));
}

smb_real scale_out(const Group *g, double t)
{
    return (smb_real)(TLO + (THI - TLO) * (t - g->tlo) / (g->thi - g->tlo));
}

double unscale_out(const Group *g, double a)
{
    return g->tlo + (a - TLO) / (THI - TLO) * (g->thi - g->tlo);
}

/* Digits enough to show the prediction move across its fitted range. At %g's default six, a
 * response near 1e8 varying over 60 prints as 1e+08 for every case. */
int sigdigits(double v, double span)
{
    int d = 6;
    if (span > 0 && fabs(v) > span)
        d = (int)floor(log10(fabs(v) / span)) + 5;
    if (d < 6)  d = 6;
    if (d > 15) d = 15;
    return d;
}


double explained(const Group *g)
{
    if (g->hsd <= 0) return 0.0;
    return 1.0 - (g->held_rmse * g->held_rmse) / (g->hsd * g->hsd);
}



int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
