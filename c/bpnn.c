/* bpnn.c -- a backpropagation network for tabular data, in the shape of linearr.
 *
 * linearr fits straight lines and says so: given a parabola it reports that the residuals are not
 * random, and stops, because a line is all it has. This is the program you run next. Same CSV layout,
 * same per-group fitting, same fit-then-score split, and a network instead of a line.
 *
 *     ./bpnn -t example/curve.csv > model.txt        fit one network per group
 *     ./bpnn -c model.txt A x=3                      score one case
 *     ./bpnn --selftest                              check the arithmetic
 *
 * INPUT is linearr's: a header, then GROUP, the response, and one column per term.
 *
 *     group,value,x
 *     A,46,-6
 *     A,35,-5
 *
 * IT REPORTS THREE WAYS THE FIT CAN BE WRONG. A regression fails differently, so these are not
 * linearr's three.
 *
 *   1. THE FIT MEMORISED THE ROWS. Two coefficients cannot memorise thirteen points; thirty weights
 *      can, and the training error then measures recall rather than accuracy. So the rows are split,
 *      the fit sees one part, and both errors are reported side by side. When the held-out error is
 *      much the larger, the model learned this table and not the relation behind it.
 *
 *   2. REFITTING GIVES A DIFFERENT ANSWER. Least squares has one solution. A network starts from
 *      random weights and shuffles its examples, so the same rows fitted twice give two models and
 *      two scores. That spread is the resolution of the tool: two configurations closer together
 *      than 2.77 times it cannot be told apart, whichever way the comparison came out. So the fit is
 *      repeated over several seeds and the spread and that limit are both reported.
 *
 *      The model written out is the MEDIAN of the seeds, not the best of them. Selecting the best by
 *      held-out error makes that error optimistic by however much was selected for (Spearman 1904 for
 *      the arithmetic, Cawley & Talbot 2010 for the modern form). Both numbers are printed.
 *
 *   3. A CASE CAN FALL OUTSIDE THE DATA. linearr lists this as something a regression cannot check.
 *      This program can, because scaling an input needs the range it was trained on. Outside that
 *      range the units saturate and the prediction is flat, with nothing in the number to show it,
 *      whereas a line keeps going in the direction the data suggested. So scoring reports every input
 *      outside its training range and how far outside it is.
 *
 * Three it cannot check: a term missing from the table, rows that are not independent of each other,
 * and a relation that changed after the training rows were collected.
 *
 * WHEN NOT TO USE IT. If the relation is close to linear, use linearr: it is more accurate, it
 * returns coefficients you can read, and it has one answer rather than a distribution of them. Use
 * this program where a line leaves structure in the residuals. bench/compare.sh measures both on the
 * same files, so the crossover is measured rather than asserted.
 *
 * PRECISION. The engine computes in `smb_real`, which is float, so about seven digits. linearr
 * validates against NIST reference values to eleven. Nothing here should be trusted past six.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>   /* the row cache is keyed on the input's size and mtime */

#include "common.h"
#include "net.h"
#include "train.h"
#include "act.h"
#include "rng.h"

/* Below this a group cannot be split into a fit, a stopping check and a reported sample of a
 * size worth printing: at 4 rows the reported error was one row and the floor beside it was ten
 * times the error it bounded. */
#define MINROWS   24
#define MAXTERM   64
#define MAXGROUP  512
#define NAMELEN   64
#define LINELEN   SMB_LINE_MAX
#define Z95       1.959964
#define TLO       0.1   /* the target is mapped into [TLO, THI]: a sigmoid output */
#define THI       0.9   /* cannot reach 0 or 1, and asking it to only saturates it */

typedef struct {
    char   name[NAMELEN];
    long   first;                    /* index of its first row in the row store */
    long   n;
    double lo[MAXTERM], hi[MAXTERM]; /* per-term training range */
    double tlo, thi;                 /* response range */
    double ymean, ym2, ysd;          /* Welford state, then the response's own spread */
    double hsd;                      /* spread of the response over the reported rows only */
    Net   *net;
    double train_rmse, held_rmse, run_sd, best_held;
    long   nseed, nheld, epochs_ran, nseen;
    double tail;                     /* how far the response reaches past its own quartiles */
    int    flat;                     /* the response never varied in this group */
} Group;

static Group  gp[MAXGROUP];
static long   ngroup;
static char   term[MAXTERM][NAMELEN];
static char   response[NAMELEN];
static long   nterm;

/* The row store: rows are grouped together, so a group is a contiguous span. The stride is the
 * file's own term count, not MAXTERM: at the ceiling a two-term file paid 520 bytes a row for
 * 16 bytes of data, and the row store is the whole of what the default path costs. */
#define ROW(i)  (xs + (size_t)(i) * (size_t)nterm)

static double *xs, *ys;
static long   *rowgrp;
static long    nrow, caprow;

static long   hidden = 6, epochs = 3000, nseed = 5;
static double rate = 0.3, momentum = 0.9, holdout = 0.25;
static int    activation = ACT_TANH;
static long   patience = 8;         /* --patience: checks without improvement before stopping */
static int    streaming;            /* --stream: fit without holding the rows */
static long   bufrows = 65536;      /* --buffer: the shuffle window, in rows */
static const char *cachepath;       /* --cache: keep the scaled rows between runs */
static double decay;                /* --decay: weight decay lambda, 0 for none */

/* ---------------------------------------------------------------- CSV input */

static int split_csv(char *line, char **f, int maxf)
{
    int n = 0, i;
    char *p = line;
    while (n < maxf) {
        f[n++] = p;
        p = strchr(p, ',');
        if (!p) break;
        *p++ = '\0';
    }
    for (i = 0; i < n; i++) {
        char *a = f[i], *b = a + strlen(a);
        while (b > a && (b[-1]=='\n'||b[-1]=='\r'||b[-1]==' '||b[-1]=='\t')) *--b = '\0';
        while (*a==' '||*a=='\t') a++;
        f[i] = a;
    }
    return n;
}

static long group_of(const char *name)
{
    long i;
    for (i = 0; i < ngroup; i++)
        if (!strcmp(gp[i].name, name)) return i;
    if (ngroup >= MAXGROUP) return -1;
    memset(&gp[ngroup], 0, sizeof gp[ngroup]);
    strncpy(gp[ngroup].name, name, NAMELEN - 1);
    return ngroup++;
}

static int grow(long want)
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
static double rowcost(void)
{
    return ((double)nterm + 1.0) * (double)sizeof(double) + (double)sizeof(long);
}

/* The row store is the whole of what the default path spends. When it runs out, say what was
 * being held and name the option that does not hold it. */
static void oom_rows(const char *doing)
{
    fprintf(stderr, "bpnn: out of memory %s: %ld rows at %.0f bytes each is %.3g GB.\n"
                    "--stream fits the same model without holding the rows, and\n"
                    "./bpnn --footprint %ld %ld prints both figures for this shape.\n",
            doing, nrow, rowcost(), rowcost() * (double)nrow / 1073741824.0,
            nterm, ngroup > 0 ? ngroup : 1);
}

/* A field that must be a finite number. WHAT names it for the message, LINE locates it.
 *
 * Each refusal here is a value that would otherwise reach a weight. An empty field read as zero,
 * a nan multiplied through the layers, or a row dropped without a message all produce the same
 * result: a model that trained, printed a number and exited 0. Nothing later in the program can
 * detect that, so the check is here. */
static int number(const char *s, const char *what, long line, double *out)
{
    char *end;
    double v;

    if (*s == '\0') {
        fprintf(stderr, "bpnn: %s is empty on line %ld. An empty field is not a zero, and\n"
                        "choosing what to put there is a modelling decision, not a parse.\n",
                what, line);
        return -1;
    }
    v = strtod(s, &end);
    while (*end == ' ' || *end == '\t') end++;
    if (end == s || *end != '\0') {
        fprintf(stderr, "bpnn: %s is '%s' on line %ld, which is not a number.\n", what, s, line);
        return -1;
    }
    if (!isfinite(v)) {
        fprintf(stderr, "bpnn: %s is '%s' on line %ld. A nan or an infinity reaches every\n"
                        "weight in the group and every number the model then prints.\n",
                what, s, line);
        return -1;
    }
    *out = v;
    return 0;
}

/* A name from the header or the group column: non-empty, and short enough to store whole. A
 * truncated group name is the one that matters, because two distinct groups sharing a prefix
 * would then be fitted as one, with no message. */
static int checkname(const char *s, const char *what, long line)
{
    if (*s == '\0') {
        fprintf(stderr, "bpnn: %s is empty on line %ld.\n", what, line);
        return -1;
    }
    if (strlen(s) >= NAMELEN) {
        fprintf(stderr, "bpnn: %s on line %ld is longer than %d characters: '%s'.\n",
                what, line, NAMELEN - 1, s);
        return -1;
    }
    return 0;
}

/* One parsed row at a time. Both fitting paths read through this, so they cannot come to
 * disagree about what a file said or about which rows are refused. */
typedef struct {
    FILE       *f;
    const char *path;
    long        lineno;
    int         header;
} Reader;

static int reader_open(Reader *rd, const char *path)
{
    rd->f = strcmp(path, "-") ? fopen(path, "r") : stdin;
    rd->path = path;
    rd->lineno = 0;
    rd->header = 0;
    if (!rd->f) { fprintf(stderr, "bpnn: cannot open %s\n", path); return -1; }
    return 0;
}

static void reader_close(Reader *rd)
{
    if (rd->f && rd->f != stdin) fclose(rd->f);
    rd->f = NULL;
}

/* Returns 1 with a row in GRP/Y/X, 0 at end of file, -1 refused with the reason printed.
 * The header is consumed on the first non-comment line and names the response and the terms. */
static int reader_row(Reader *rd, long *grp, double *y, double *x)
{
    char line[LINELEN], *fld[MAXTERM + 8];
    char what[NAMELEN + 32];
    int nf, i;

    while (fgets(line, sizeof line, rd->f)) {
        rd->lineno++;
        /* fgets splits an over-long line in two, and the tail would then be read as a row of
         * its own. Refuse it rather than fit half a row. */
        if (strchr(line, '\n') == NULL && !feof(rd->f)) {
            fprintf(stderr, "bpnn: line %ld is longer than %d characters.\n",
                    rd->lineno, LINELEN - 1);
            return -1;
        }
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        nf = split_csv(line, fld, MAXTERM + 8);
        if (!rd->header) {
            if (nf < 3) {
                fprintf(stderr, "bpnn: the header on line %ld needs a group column, the value\n"
                                "being predicted, and at least one term.\n", rd->lineno);
                return -1;
            }
            nterm = nf - 2;
            if (nterm > MAXTERM) {
                fprintf(stderr, "bpnn: %ld terms on line %ld; this build holds %d.\n",
                        nterm, rd->lineno, MAXTERM);
                return -1;
            }
            if (checkname(fld[1], "the name of the value being predicted", rd->lineno) != 0)
                return -1;
            strncpy(response, fld[1], NAMELEN - 1);
            for (i = 0; i < nterm; i++) {
                int j;
                if (checkname(fld[i + 2], "a term name", rd->lineno) != 0) return -1;
                for (j = 0; j < i; j++)
                    if (!strcmp(term[j], fld[i + 2])) {
                        fprintf(stderr, "bpnn: the header names the term '%s' twice, so a case\n"
                                        "naming it could mean either column.\n", fld[i + 2]);
                        return -1;
                    }
                strncpy(term[i], fld[i + 2], NAMELEN - 1);
            }
            rd->header = 1;
            continue;
        }
        if (nf != nterm + 2) {
            fprintf(stderr, "bpnn: line %ld has %d field%s; the header declared %ld (the group,\n"
                            "%s, and %ld term%s).\n", rd->lineno, nf, nf == 1 ? "" : "s",
                    nterm + 2, response, nterm, nterm == 1 ? "" : "s");
            return -1;
        }
        if (checkname(fld[0], "the group", rd->lineno) != 0) return -1;
        *grp = group_of(fld[0]);
        if (*grp < 0) { fprintf(stderr, "bpnn: more than %d groups.\n", MAXGROUP); return -1; }
        snprintf(what, sizeof what, "'%s', the value being predicted,", response);
        if (number(fld[1], what, rd->lineno, y) != 0) return -1;
        for (i = 0; i < nterm; i++) {
            snprintf(what, sizeof what, "the term '%s'", term[i]);
            if (number(fld[i + 2], what, rd->lineno, &x[i]) != 0) return -1;
        }
        return 1;
    }
    if (ferror(rd->f)) { fprintf(stderr, "bpnn: error reading %s\n", rd->path); return -1; }
    if (!rd->header) { fprintf(stderr, "bpnn: %s has no header line.\n", rd->path); return -1; }
    return 0;
}

/* Read PATH into the row store: every row resident, which is what the default path fits from. */
static int read_csv(const char *path)
{
    Reader rd;
    double x[MAXTERM], y;
    long g;
    int r;

    if (reader_open(&rd, path) != 0) return -1;
    while ((r = reader_row(&rd, &g, &y, x)) == 1) {
        if (grow(nrow + 1) != 0) { oom_rows("reading the file"); r = -1; break; }
        memcpy(ROW(nrow), x, (size_t)nterm * sizeof *xs);
        ys[nrow] = y;
        rowgrp[nrow] = g;
        gp[g].n++;
        nrow++;
    }
    reader_close(&rd);
    return r == 0 ? 0 : -1;
}

/* Sort the rows by group, so that a group is one contiguous span.
 *
 * In place, by swaps. Copying into a second array is simpler, but it holds both arrays at once
 * and the row store is the largest thing this program allocates, so that copy doubled the peak.
 * Here the extra memory is one row plus one long per group. rowgrp is reused to hold each row's
 * destination: the group number is not needed once the destination is known. */
static int regroup(void)
{
    double *tmp = malloc((size_t)nterm * sizeof *tmp);
    long   *at = malloc((size_t)ngroup * sizeof *at), i, run = 0;

    if (!tmp || !at) {
        free(tmp); free(at);
        oom_rows("sorting the rows by group");
        return -1;
    }
    for (i = 0; i < ngroup; i++) { gp[i].first = run; at[i] = run; run += gp[i].n; }
    for (i = 0; i < nrow; i++) rowgrp[i] = at[rowgrp[i]]++;
    free(at);

    /* rowgrp[i] is where the row now at i belongs. Swapping it there also puts the row that was
     * in the way into a position whose destination is known, so each swap places one row. */
    for (i = 0; i < nrow; i++)
        while (rowgrp[i] != i) {
            long d = rowgrp[i];
            double y;
            memcpy(tmp,    ROW(i), (size_t)nterm * sizeof *tmp);
            memcpy(ROW(i), ROW(d), (size_t)nterm * sizeof *tmp);
            memcpy(ROW(d), tmp,    (size_t)nterm * sizeof *tmp);
            y = ys[i]; ys[i] = ys[d]; ys[d] = y;
            rowgrp[i] = rowgrp[d]; rowgrp[d] = d;
        }
    free(tmp);
    return 0;
}

/* ------------------------------------------------------------------ scaling */

/* The range of every term and of the response, which is all the scaling needs. Split into three
 * so that the streaming path can accumulate it in one pass without holding a row: this is the
 * one part of a network's fit that has a fixed-size sufficient statistic, exactly as a least
 * squares fit does, and it is the reason the first pass can be a pass and not a load. */
static void range_init(Group *g)
{
    long j;
    for (j = 0; j < nterm; j++) { g->lo[j] = 1e300; g->hi[j] = -1e300; }
    g->tlo = 1e300; g->thi = -1e300;
    g->ymean = g->ym2 = 0;
    g->nseen = 0;
}

static void range_add(Group *g, const double *x, double y)
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

static void range_done(Group *g)
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

static void ranges(Group *g)
{
    long i;
    range_init(g);
    for (i = g->first; i < g->first + g->n; i++)
        range_add(g, ROW(i), ys[i]);
    range_done(g);
}

static void scale_in(const Group *g, const double *raw, smb_real *out)
{
    long j;
    for (j = 0; j < nterm; j++)
        out[j] = (smb_real)((raw[j] - g->lo[j]) / (g->hi[j] - g->lo[j]));
}

static smb_real scale_out(const Group *g, double t)
{
    return (smb_real)(TLO + (THI - TLO) * (t - g->tlo) / (g->thi - g->tlo));
}

static double unscale_out(const Group *g, double a)
{
    return g->tlo + (a - TLO) / (THI - TLO) * (g->thi - g->tlo);
}

/* Enough significant digits to show a prediction move across the range it was fitted on. A
 * response near 1e8 that varies over 60 prints as 1e+08 at %g's six digits, for every case in
 * the group: six digits are all spent on the offset. */
static int sigdigits(double v, double span)
{
    int d = 6;
    if (span > 0 && fabs(v) > span)
        d = (int)floor(log10(fabs(v) / span)) + 5;
    if (d < 6)  d = 6;
    if (d > 15) d = 15;
    return d;
}

/* --------------------------------------------------------------------- fit  */

/* Copy one net's weights into another of the same shape. */
static void net_copy_weights(Net *dst, const Net *src)
{
    size_t l;
    for (l = 1; l < src->nlayers; l++) {
        memcpy(dst->w[l], src->w[l], net_layer_wsize(src, l) * sizeof *src->w[l]);
        memcpy(dst->b[l], src->b[l], net_layer_bsize(src, l) * sizeof *src->b[l]);
    }
}

static double rmse(Group *g, Net *net, const long *ord, long a, long b);

/* Train one net on the rows at positions [0, ntr) of `ord`.
 *
 * With --patience 0 it runs the full -e epochs. Otherwise it checks the error on the stop rows,
 * positions [ntr, nstop), every CHECK epochs, keeps the weights from the best check, and gives up
 * after `patience` checks with no improvement. Measured on the example files, the held-out error
 * stops improving one to two orders of magnitude before 3000 epochs, and everything after that is
 * time spent inside the refit spread.
 *
 * The stop rows are not the rows the error is reported on. Choosing when to stop by a number makes
 * that number optimistic by however much was selected for, which is the same mistake as shipping
 * the best seed instead of the median, so the held-out rows are split: one half decides when to
 * stop, the other half is reported and is never looked at during the fit.
 *
 * EPOCHS_OUT, when not NULL, receives the number of epochs actually run. */
#define CHECK 25

static Net *train_one(Group *g, long *ord, long ntr, long nstop, uint32_t seed, long *epochs_out)
{
    size_t dims[3];
    Net *net, *best = NULL;
    Trainer *t;
    Rng rng;
    long e, i, since = 0, best_epoch = 0;
    double bestv = 0;
    int stopping = patience > 0 && nstop > ntr, have_best = 0;

    dims[0] = (size_t)nterm; dims[1] = (size_t)hidden; dims[2] = 1;
    net = net_new(dims, 3);
    if (!net) return NULL;
    net->activation = activation;
    rng_seed(&rng, seed);
    net_init(net, &rng);
    t = trainer_new(net, (smb_real)rate, (smb_real)momentum);
    if (!t) { net_free(net); return NULL; }
    trainer_decay(t, (smb_real)decay);
    if (stopping) {
        best = net_new(dims, 3);
        if (!best) { trainer_free(t); net_free(net); return NULL; }
        best->activation = activation;
    }
    for (e = 0; e < epochs; e++) {
        for (i = ntr - 1; i > 0; i--) {          /* shuffle the training part in place */
            long k = (long)(rng_u32(&rng) % (uint32_t)(i + 1));
            long tmp = ord[i]; ord[i] = ord[k]; ord[k] = tmp;
        }
        for (i = 0; i < ntr; i++) {
            smb_real xn[MAXTERM], d;
            long r = g->first + ord[i];
            scale_in(g, ROW(r), xn);
            d = scale_out(g, ys[r]);
            (void)trainer_learn(t, xn, &d);
        }
        if (stopping && (e + 1) % CHECK == 0) {
            double v = rmse(g, net, ord, ntr, nstop);
            if (!have_best || v < bestv) {
                bestv = v; have_best = 1; since = 0; best_epoch = e + 1;
                net_copy_weights(best, net);
            } else if (++since >= patience) {
                e++;
                break;
            }
        }
    }
    /* The epoch reported is the one whose weights are being kept, not the one the run gave up
     * at: those differ by patience*CHECK and only the first describes the model. */
    if (epochs_out) *epochs_out = have_best ? best_epoch : (e < epochs ? e : epochs);
    trainer_free(t);
    if (stopping) {
        if (have_best) net_copy_weights(net, best);   /* never copy a net that was not filled */
        net_free(best);
    }
    return net;
}

/* RMSE in the response's own units over positions [a, b) of ord. */
static double rmse(Group *g, Net *net, const long *ord, long a, long b)
{
    double se = 0; long i;
    if (b <= a) return -1;
    for (i = a; i < b; i++) {
        smb_real xn[MAXTERM];
        long r = g->first + ord[i];
        double p;
        scale_in(g, ROW(r), xn);
        p = unscale_out(g, (double)net_forward(net, xn)[0]);
        se += (p - ys[r]) * (p - ys[r]);
    }
    return sqrt(se / (double)(b - a));
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Fit one group over `nseed` seeds; keep the median-held-out-error net. */
static int fit_group(Group *g)
{
    long *ord = malloc((size_t)g->n * sizeof *ord);
    double *held = malloc((size_t)nseed * sizeof *held);
    double *srt = malloc((size_t)nseed * sizeof *srt);
    long *ran = calloc((size_t)nseed, sizeof *ran);
    Net **nets = calloc((size_t)nseed, sizeof *nets);
    long i, s, ntr, nstop, med = 0;
    double m = 0, v = 0, mid;
    int rc = -1;
    if (!ord || !held || !srt || !nets || !ran) goto done;
    ranges(g);
    /* The target is mapped onto the sigmoid's band by its MINIMUM and MAXIMUM, so one extreme
     * row compresses every other row into a sliver of that band and the network can no longer
     * tell them apart. Measured against the quartiles, which one row cannot move. */
    {
        double *sorted = malloc((size_t)g->n * sizeof *sorted);
        if (sorted) {
            long i3;
            double q1, q3, iqr;
            for (i3 = 0; i3 < g->n; i3++) sorted[i3] = ys[g->first + i3];
            qsort(sorted, (size_t)g->n, sizeof *sorted, cmpd);
            q1 = sorted[g->n / 4]; q3 = sorted[(3 * g->n) / 4];
            iqr = q3 - q1;
            g->tail = iqr > 0 ? ((sorted[g->n - 1] - q3) > (q1 - sorted[0])
                                 ? (sorted[g->n - 1] - q3) / iqr : (q1 - sorted[0]) / iqr) : 0.0;
            free(sorted);
        }
    }
    /* Named here rather than at the option, because it depends on the group's row count. */
    if (decay > 0) {
        double eff = decay * rate * (double)g->n / (1.0 - momentum);
        if (eff > 0.5)
            fprintf(stderr, "bpnn: --decay %g over %ld rows at -r %g and -m %g shrinks a weight by\n"
                            "  a factor of %.3g per epoch. Above about 0.5 the fit collapses onto\n"
                            "  the response's mean. Try %.1g.\n",
                    decay, g->n, rate, momentum, eff, 0.1 * decay / eff);
    }
    ntr = holdout > 0 ? (long)((1.0 - holdout) * (double)g->n + 0.5) : g->n;
    if (ntr < 1) ntr = 1;
    if (ntr > g->n) ntr = g->n;
    /* Half the held-out rows decide when to stop, the other half are reported. Below four rows
     * either way the split is too small to mean anything, so the fit runs its full epochs. */
    nstop = ntr + (g->n - ntr) / 2;
    if (patience <= 0 || nstop - ntr < 4 || g->n - nstop < 4) nstop = ntr;
    g->nheld = g->n - nstop;
    for (s = 0; s < nseed; s++) {
        Rng r;
        /* a fresh, seed-dependent split AND init per seed, so the reported spread is the
         * spread of the whole procedure and not of the initial weights alone */
        rng_seed(&r, (uint32_t)(1000u + 7919u * (unsigned)s));
        for (i = 0; i < g->n; i++) ord[i] = i;
        for (i = g->n - 1; i > 0; i--) {
            long k = (long)(rng_u32(&r) % (uint32_t)(i + 1));
            long t = ord[i]; ord[i] = ord[k]; ord[k] = t;
        }
        nets[s] = train_one(g, ord, ntr, nstop, (uint32_t)(1u + 104729u * (unsigned)s), &ran[s]);
        if (!nets[s]) goto done;
        held[s] = g->nheld > 0 ? rmse(g, nets[s], ord, nstop, g->n) : rmse(g, nets[s], ord, 0, ntr);
        if (held[s] < 0) goto done;
        if (s == 0 || held[s] < g->best_held) g->best_held = held[s];
        if (s == 0) g->train_rmse = rmse(g, nets[s], ord, 0, ntr);
    }
    for (s = 0; s < nseed; s++) { m += held[s]; srt[s] = held[s]; }
    m /= (double)nseed;
    for (s = 0; s < nseed; s++) v += (held[s] - m) * (held[s] - m);
    g->run_sd = nseed > 1 ? sqrt(v / (double)(nseed - 1)) : 0.0;
    qsort(srt, (size_t)nseed, sizeof *srt, cmpd);
    /* The LOWER middle. srt[nseed/2] is the upper one, which at two refits is the maximum:
     * the model shipped was the worse of the two while the header claimed the median. */
    mid = srt[(nseed - 1) / 2];
    for (s = 0; s < nseed; s++) if (held[s] == mid) { med = s; break; }
    g->net = nets[med];
    nets[med] = NULL;
    g->held_rmse = held[med];
    g->nseed = nseed;
    g->epochs_ran = ran[med];
    /* The denominator of the variance explained has to be the spread of the same rows the
     * numerator was measured on. Over all the group's rows instead, one extreme value inflates
     * it and the ratio certifies a destroyed fit as perfect. */
    {
        long a = g->nheld > 0 ? nstop : 0, b = g->nheld > 0 ? g->n : ntr, i2;
        double hm = 0, q = 0;
        for (i2 = a; i2 < b; i2++) hm += ys[g->first + ord[i2]];
        hm /= (double)(b - a);
        for (i2 = a; i2 < b; i2++) {
            double d = ys[g->first + ord[i2]] - hm;
            q += d * d;
        }
        g->hsd = b - a > 1 ? sqrt(q / (double)(b - a - 1)) : 0.0;
    }
    /* the shipped model's training error, recomputed on its own split */
    {
        Rng r;
        rng_seed(&r, (uint32_t)(1000u + 7919u * (unsigned)med));
        for (i = 0; i < g->n; i++) ord[i] = i;
        for (i = g->n - 1; i > 0; i--) {
            long k = (long)(rng_u32(&r) % (uint32_t)(i + 1));
            long t = ord[i]; ord[i] = ord[k]; ord[k] = t;
        }
        g->train_rmse = rmse(g, g->net, ord, 0, ntr);
    }
    rc = 0;
done:
    for (s = 0; s < nseed; s++) if (nets && nets[s]) net_free(nets[s]);
    free(ord); free(held); free(srt); free(nets); free(ran);
    return rc;
}

/* ---------------------------------------------------- fitting from a stream */

/* The default path holds every row in memory. This one does not. It reads the file twice, once
 * for the ranges and once to write a cache of the scaled rows, then makes one pass over that
 * cache per epoch. Memory is the networks plus the shuffle windows. Neither depends on the
 * number of rows.
 *
 * Least squares needs one pass because its objective has a fixed-size sufficient statistic.
 * Backpropagation has none: the gradient depends on the current weights, so every epoch has to
 * see the rows again. Streaming removes the storage, not the passes.
 *
 * The two paths give different numbers. The default shuffles each group's rows completely every
 * epoch; this one shuffles through a window of --buffer rows, so the training order differs.
 * Both are deterministic. The default is the one the archived numbers come from.
 *
 * A cache record holds the group, the row's ordinal within that group, and the scaled terms and
 * target. Scaling here means an epoch is a sequential read with no arithmetic per row except the
 * network's own. */

#define REC_HEAD  (2 * sizeof(int32_t))

static size_t recsize(void)
{
    return REC_HEAD + (size_t)(nterm + 1) * sizeof(float);
}

/* Which rows are held out, from a hash of the row's position rather than a draw from the
 * training PRNG. A draw would move the split whenever the training consumed a different number
 * of random numbers, which is a change in the split for an unrelated reason. The seed is part of
 * the hash so that each refit holds out a different quarter, as the default path does; the
 * reported spread is over splits as well as over starting weights. */
static uint32_t mix32(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t h = a * 0x9E3779B1u + b * 0x85EBCA6Bu + c * 0xC2B2AE35u;
    h ^= h >> 15; h *= 0x2C1B3C6Du;
    h ^= h >> 13; h *= 0x297A2D39u;
    h ^= h >> 16;
    return h;
}

/* What a row is for, in one refit: 0 trains on it, 1 uses it to decide when to stop, 2 has it
 * reported. The stop half and the reported half are drawn by a second, independent hash, for the
 * same reason the default path splits its held-out rows: a stopping point chosen on a number
 * makes that number optimistic by however much was selected for. */
#define ROLE_TRAIN  0
#define ROLE_STOP   1
#define ROLE_REPORT 2

static int rowrole(long g, long ord, long seed)
{
    uint32_t h;
    if (holdout <= 0) return ROLE_TRAIN;
    h = mix32((uint32_t)g, (uint32_t)ord, (uint32_t)seed);
    if ((double)h >= holdout * 4294967296.0) return ROLE_TRAIN;
    if (patience <= 0) return ROLE_REPORT;
    return (mix32((uint32_t)ord, (uint32_t)seed, (uint32_t)g) & 1u) ? ROLE_STOP : ROLE_REPORT;
}

/* A window of records in random order: push one in, take one out. Memory is the window size,
 * where a full permutation would need eight bytes per row of the file. */
typedef struct {
    unsigned char *buf;
    long           fill;
    Rng            rng;
} Shuf;

static int shuf_push(Shuf *s, const unsigned char *in, unsigned char *out, size_t rs, long cap)
{
    long j;
    if (s->fill < cap) {
        memcpy(s->buf + (size_t)s->fill * rs, in, rs);
        s->fill++;
        return 0;
    }
    j = (long)(rng_u32(&s->rng) % (uint32_t)cap);
    memcpy(out, s->buf + (size_t)j * rs, rs);
    memcpy(s->buf + (size_t)j * rs, in, rs);
    return 1;
}

/* Empty the window in random order at the end of a pass, so one epoch is one visit to each
 * row and no row is carried over into the next epoch. */
static int shuf_pop(Shuf *s, unsigned char *out, size_t rs)
{
    long j;
    if (s->fill <= 0) return 0;
    j = (long)(rng_u32(&s->rng) % (uint32_t)s->fill);
    memcpy(out, s->buf + (size_t)j * rs, rs);
    s->fill--;
    memcpy(s->buf + (size_t)j * rs, s->buf + (size_t)s->fill * rs, rs);
    return 1;
}

/* ---------------------------------------------------- the cache between runs
 *
 * --stream reads the CSV twice before the first epoch: once for the ranges, once to write the
 * scaled rows. Neither depends on the hyperparameters, so with --cache both passes are written
 * to a named file and a later run of any shape reuses them.
 *
 * The key is the input's size and modification time. That is what every make-like tool uses and
 * it is wrong in the same way theirs are: a file rewritten within the same second, to the same
 * length, with different contents reuses a stale cache. Nothing here can detect that without
 * reading the file, which is the cost the cache exists to avoid. */

#define CACHE_MAGIC   "BPNNCACH"
#define CACHE_VERSION 2

typedef struct { long size, mtime; } Stamp;

static int stamp_of(const char *path, Stamp *st)
{
    struct stat sb;
    if (strcmp(path, "-") == 0 || stat(path, &sb) != 0) return -1;
    st->size = (long)sb.st_size;
    st->mtime = (long)sb.st_mtime;
    return 0;
}

static int put_long(FILE *f, long v)   { return fwrite(&v, sizeof v, 1, f) == 1 ? 0 : -1; }
static int get_long(FILE *f, long *v)  { return fread(v, sizeof *v, 1, f) == 1 ? 0 : -1; }
static int put_dbl(FILE *f, double v)  { return fwrite(&v, sizeof v, 1, f) == 1 ? 0 : -1; }
static int get_dbl(FILE *f, double *v) { return fread(v, sizeof *v, 1, f) == 1 ? 0 : -1; }
static int put_name(FILE *f, const char *s) { return fwrite(s, NAMELEN, 1, f) == 1 ? 0 : -1; }
static int get_name(FILE *f, char *s)  { if (fread(s, NAMELEN, 1, f) != 1) return -1;
                                         s[NAMELEN - 1] = 0; return 0; }

/* Everything the two setup passes produced, so that a later run does not have to repeat them. */
static int cache_put_header(FILE *f, const Stamp *st, long runs, long steps)
{
    long i, j;
    if (fwrite(CACHE_MAGIC, 8, 1, f) != 1) return -1;
    if (put_long(f, CACHE_VERSION) || put_long(f, st->size) || put_long(f, st->mtime)) return -1;
    if (put_long(f, nterm) || put_long(f, ngroup) || put_long(f, nrow)) return -1;
    if (put_long(f, runs) || put_long(f, steps)) return -1;
    if (put_name(f, response)) return -1;
    for (i = 0; i < nterm; i++) if (put_name(f, term[i])) return -1;
    for (i = 0; i < ngroup; i++) {
        if (put_name(f, gp[i].name) || put_long(f, gp[i].n) || put_long(f, gp[i].flat)) return -1;
        if (put_dbl(f, gp[i].tlo) || put_dbl(f, gp[i].thi)) return -1;
        /* The response's spread is derived from the rows and cannot be recomputed without
         * them, so it travels with the cache. Leaving it out made a reused cache report that
         * a good fit explained nothing. */
        if (put_dbl(f, gp[i].ysd)) return -1;
        for (j = 0; j < nterm; j++)
            if (put_dbl(f, gp[i].lo[j]) || put_dbl(f, gp[i].hi[j])) return -1;
    }
    return 0;
}

/* Returns 0 with the tables filled and the file positioned at the first record, or -1 with the
 * reason printed, in which case the caller rebuilds. */
static int cache_get_header(FILE *f, const char *path, const Stamp *want, long *runs, long *steps)
{
    char magic[8];
    long v, i, j, sz, mt;
    if (fread(magic, 8, 1, f) != 1 || memcmp(magic, CACHE_MAGIC, 8) != 0) {
        fprintf(stderr, "bpnn: %s is not a bpnn row cache; rebuilding it\n", path);
        return -1;
    }
    if (get_long(f, &v) || v != CACHE_VERSION) {
        fprintf(stderr, "bpnn: %s was written by another version; rebuilding it\n", path);
        return -1;
    }
    if (get_long(f, &sz) || get_long(f, &mt)) return -1;
    if (sz != want->size || mt != want->mtime) {
        fprintf(stderr, "bpnn: the input has changed since %s was written; rebuilding it\n", path);
        return -1;
    }
    if (get_long(f, &nterm) || get_long(f, &ngroup) || get_long(f, &nrow)) return -1;
    if (nterm < 1 || nterm > MAXTERM || ngroup < 1 || ngroup > MAXGROUP || nrow < 1) {
        fprintf(stderr, "bpnn: %s is damaged; rebuilding it\n", path);
        return -1;
    }
    if (get_long(f, runs) || get_long(f, steps)) return -1;
    if (get_name(f, response)) return -1;
    for (i = 0; i < nterm; i++) if (get_name(f, term[i])) return -1;
    for (i = 0; i < ngroup; i++) {
        memset(&gp[i], 0, sizeof gp[i]);
        if (get_name(f, gp[i].name) || get_long(f, &gp[i].n) || get_long(f, &v)) return -1;
        gp[i].flat = (int)v;
        if (get_dbl(f, &gp[i].tlo) || get_dbl(f, &gp[i].thi)) return -1;
        if (get_dbl(f, &gp[i].ysd)) return -1;
        for (j = 0; j < nterm; j++)
            if (get_dbl(f, &gp[i].lo[j]) || get_dbl(f, &gp[i].hi[j])) return -1;
    }
    return 0;
}

/* Pass one: the ranges, the row counts, and whether the file arrives sorted by the response. */
static int scan_ranges(const char *path, long *sorted_runs, long *sorted_steps)
{
    Reader rd;
    double x[MAXTERM], y, *prev;
    long g, i, r;
    int rc = -1;

    prev = calloc(MAXGROUP, sizeof *prev);
    if (!prev) { fprintf(stderr, "bpnn: out of memory\n"); return -1; }
    if (reader_open(&rd, path) != 0) { free(prev); return -1; }
    while ((r = reader_row(&rd, &g, &y, x)) == 1) {
        if (gp[g].n == 0) range_init(&gp[g]);
        else {
            (*sorted_steps)++;
            if (y >= prev[g]) (*sorted_runs)++;
        }
        prev[g] = y;
        range_add(&gp[g], x, y);
        gp[g].n++;
        nrow++;
    }
    if (r != 0) goto done;
    for (i = 0; i < ngroup; i++) range_done(&gp[i]);
    rc = 0;
done:
    free(prev);
    reader_close(&rd);
    return rc;
}

/* Pass two: the same rows, scaled once and written to the cache the epochs read. */
static int pack_cache(const char *path, FILE *cache)
{
    Reader rd;
    double x[MAXTERM], y;
    unsigned char rec[REC_HEAD + (MAXTERM + 1) * sizeof(float)];
    long *ord = calloc((size_t)MAXGROUP, sizeof *ord);
    long g, r;
    int rc = -1;

    if (!ord) { fprintf(stderr, "bpnn: out of memory\n"); return -1; }
    if (reader_open(&rd, path) != 0) { free(ord); return -1; }
    while ((r = reader_row(&rd, &g, &y, x)) == 1) {
        int32_t gi = (int32_t)g, oi = (int32_t)ord[g];
        smb_real xn[MAXTERM], t;
        long j;
        scale_in(&gp[g], x, xn);
        t = scale_out(&gp[g], y);
        memcpy(rec, &gi, sizeof gi);
        memcpy(rec + sizeof gi, &oi, sizeof oi);
        for (j = 0; j < nterm; j++) {
            float v = (float)xn[j];
            memcpy(rec + REC_HEAD + (size_t)j * sizeof v, &v, sizeof v);
        }
        { float v = (float)t;
          memcpy(rec + REC_HEAD + (size_t)nterm * sizeof v, &v, sizeof v); }
        if (fwrite(rec, recsize(), 1, cache) != 1) {
            fprintf(stderr, "bpnn: cannot write the row cache (out of temporary space?)\n");
            goto done;
        }
        ord[g]++;
    }
    if (r != 0) goto done;
    rc = 0;
done:
    free(ord);
    reader_close(&rd);
    return rc;
}

/* Fit every group and every refit in one set of passes over the cache. */
static int fit_stream(const char *path)
{
    FILE          *cache = NULL;
    Net          **nets = NULL;
    Trainer      **trs = NULL;
    Shuf          *sh = NULL;
    unsigned char *rec = NULL, *out = NULL;
    double        *sse_tr = NULL, *sse_he = NULL, *held = NULL, *srt = NULL;
    double        *sse_st = NULL, *bestv = NULL;
    long          *n_tr = NULL, *n_he = NULL, *n_st = NULL, *since = NULL, *ran = NULL;
    long          *n_best = NULL;
    Net          **bests = NULL;
    char          *done = NULL, *nojudge = NULL;
    long           sorted_runs = 0, sorted_steps = 0;
    long           i, s, e, k;
    long           first = 0;
    Stamp          stamp;
    size_t         rs;
    int            reused = 0, rc = -1;

    /* With --cache, the two setup passes are skipped whenever the named file already describes
     * this input. Without it, the cache is a temporary file and the passes always run. */
    if (cachepath && stamp_of(path, &stamp) == 0) {
        cache = fopen(cachepath, "rb");
        if (cache) {
            if (cache_get_header(cache, cachepath, &stamp, &sorted_runs, &sorted_steps) == 0)
                reused = 1;
            else { fclose(cache); cache = NULL; }
        }
    }
    if (!reused) {
        if (scan_ranges(path, &sorted_runs, &sorted_steps) != 0) return -1;
        if (nrow == 0) {
            fprintf(stderr, "bpnn: %s has a header and no data rows\n", path);
            return -1;
        }
    }
    rs = recsize();
    /* A window larger than the file is a window the file cannot fill, and one window per refit is
     * what --stream costs on a small file. The first pass has counted the rows, so cap it here. */
    if (bufrows > nrow) bufrows = nrow;

    k = ngroup * nseed;
    nets   = calloc((size_t)k, sizeof *nets);
    trs    = calloc((size_t)k, sizeof *trs);
    sse_tr = calloc((size_t)k, sizeof *sse_tr);
    sse_he = calloc((size_t)k, sizeof *sse_he);
    n_tr   = calloc((size_t)k, sizeof *n_tr);
    n_he   = calloc((size_t)k, sizeof *n_he);
    sse_st = calloc((size_t)k, sizeof *sse_st);
    n_st   = calloc((size_t)k, sizeof *n_st);
    bestv  = calloc((size_t)k, sizeof *bestv);
    n_best = calloc((size_t)k, sizeof *n_best);
    since  = calloc((size_t)k, sizeof *since);
    ran    = calloc((size_t)k, sizeof *ran);
    done   = calloc((size_t)k, sizeof *done);
    nojudge = calloc((size_t)k, sizeof *nojudge);
    bests  = calloc((size_t)k, sizeof *bests);
    held   = calloc((size_t)nseed, sizeof *held);
    srt    = calloc((size_t)nseed, sizeof *srt);
    sh     = calloc((size_t)nseed, sizeof *sh);
    rec    = malloc(rs);
    out    = malloc(rs);
    if (!nets || !trs || !sse_tr || !sse_he || !n_tr || !n_he || !held || !srt || !sh
        || !rec || !out || !sse_st || !n_st || !bestv || !n_best || !since || !ran || !done
        || !bests || !nojudge) { fprintf(stderr, "bpnn: out of memory\n"); goto done; }

    for (s = 0; s < nseed; s++) {
        sh[s].buf = malloc((size_t)bufrows * rs);
        if (!sh[s].buf) { fprintf(stderr, "bpnn: out of memory for the shuffle window\n"); goto done; }
        rng_seed(&sh[s].rng, (uint32_t)(7001u + 65537u * (unsigned)s));
    }
    for (i = 0; i < ngroup; i++) {
        size_t dims[3];
        dims[0] = (size_t)nterm; dims[1] = (size_t)hidden; dims[2] = 1;
        for (s = 0; s < nseed; s++) {
            Rng r;
            Net *n = net_new(dims, 3);
            if (!n) { fprintf(stderr, "bpnn: out of memory for the networks\n"); goto done; }
            n->activation = activation;
            rng_seed(&r, (uint32_t)(1u + 104729u * (unsigned)s));
            net_init(n, &r);
            nets[i * nseed + s] = n;
            trs[i * nseed + s] = trainer_new(n, (smb_real)rate, (smb_real)momentum);
            if (!trs[i * nseed + s]) { fprintf(stderr, "bpnn: out of memory\n"); goto done; }
            trainer_decay(trs[i * nseed + s], (smb_real)decay);
            if (patience > 0) {
                bests[i * nseed + s] = net_new(dims, 3);
                if (!bests[i * nseed + s]) { fprintf(stderr, "bpnn: out of memory\n"); goto done; }
                bests[i * nseed + s]->activation = activation;
            }
        }
    }

    if (!reused) {
        cache = cachepath ? fopen(cachepath, "w+b") : tmpfile();
        if (!cache) {
            fprintf(stderr, "bpnn: cannot open %s for the row cache\n",
                    cachepath ? cachepath : "a temporary file");
            goto done;
        }
        if (cachepath && cache_put_header(cache, &stamp, sorted_runs, sorted_steps) != 0) {
            fprintf(stderr, "bpnn: cannot write %s\n", cachepath);
            goto done;
        }
    }
    /* Where the records start, taken before any of them are written and after the header is read
     * back, since every pass over the cache seeks here. */
    first = ftell(cache);
    if (first < 0) { fprintf(stderr, "bpnn: cannot seek the row cache\n"); goto done; }
    if (!reused && pack_cache(path, cache) != 0) goto done;

    for (e = 0; e < epochs; e++) {
        if (fseek(cache, first, SEEK_SET) != 0) goto done;
        while (fread(rec, rs, 1, cache) == 1) {
            for (s = 0; s < nseed; s++)
                if (shuf_push(&sh[s], rec, out, rs, bufrows)) {
                    int32_t g, ord;
                    memcpy(&g, out, sizeof g);
                    memcpy(&ord, out + sizeof g, sizeof ord);
                    if (done[(long)g * nseed + s] != 1 && rowrole(g, ord, s) == ROLE_TRAIN)
                        (void)trainer_learn(trs[(long)g * nseed + s],
                                            (const smb_real *)(void *)(out + REC_HEAD),
                                            (const smb_real *)(void *)(out + REC_HEAD +
                                                (size_t)nterm * sizeof(float)));
                }
        }
        for (s = 0; s < nseed; s++)
            while (shuf_pop(&sh[s], out, rs)) {
                int32_t g, ord;
                memcpy(&g, out, sizeof g);
                memcpy(&ord, out + sizeof g, sizeof ord);
                if (done[(long)g * nseed + s] != 1 && rowrole(g, ord, s) == ROLE_TRAIN)
                    (void)trainer_learn(trs[(long)g * nseed + s],
                                        (const smb_real *)(void *)(out + REC_HEAD),
                                        (const smb_real *)(void *)(out + REC_HEAD +
                                            (size_t)nterm * sizeof(float)));
            }

        /* Every CHECK epochs, one extra read of the cache scores the stop rows of every fit that
         * is still running. It costs one pass in twenty-five, and it is a separate pass because
         * the weights have to hold still while a fit is being judged. */
        if (patience > 0 && (e + 1) % CHECK == 0) {
            long live = 0;
            for (i = 0; i < k; i++) { sse_st[i] = 0; n_st[i] = 0; }
            if (fseek(cache, first, SEEK_SET) != 0) goto done;
            while (fread(rec, rs, 1, cache) == 1) {
                int32_t g, ord;
                float t;
                double y;
                memcpy(&g, rec, sizeof g);
                memcpy(&ord, rec + sizeof g, sizeof ord);
                memcpy(&t, rec + REC_HEAD + (size_t)nterm * sizeof t, sizeof t);
                y = unscale_out(&gp[g], (double)t);
                for (s = 0; s < nseed; s++) {
                    long at = (long)g * nseed + s;
                    double p;
                    if (done[at] == 1 || nojudge[at] || rowrole(g, ord, s) != ROLE_STOP) continue;
                    p = unscale_out(&gp[g], (double)net_forward(nets[at],
                            (const smb_real *)(void *)(rec + REC_HEAD))[0]);
                    sse_st[at] += (p - y) * (p - y);
                    n_st[at]++;
                }
            }
            for (i = 0; i < k; i++) {
                double v;
                if (done[i]) continue;
                /* Too few stop rows to judge this fit: it keeps training to the ceiling. This
                 * is a separate state from "stopped", which the training loop tests. */
                if (n_st[i] < 4) { nojudge[i] = 1; live++; continue; }
                live++;
                v = sqrt(sse_st[i] / (double)n_st[i]);
                if (n_best[i] == 0 || v < bestv[i]) {
                    bestv[i] = v; n_best[i] = 1; since[i] = 0; ran[i] = e + 1;
                    net_copy_weights(bests[i], nets[i]);
                } else if (++since[i] >= patience) {
                    done[i] = 1;
                }
            }
            if (live == 0) break;
        }
    }

    /* Every fit that stopped goes back to the weights of its best check. */
    if (patience > 0)
        for (i = 0; i < k; i++)
            if (n_best[i]) net_copy_weights(nets[i], bests[i]);

    /* One more pass for the errors, in the response's own units. */
    if (fseek(cache, first, SEEK_SET) != 0) goto done;
    while (fread(rec, rs, 1, cache) == 1) {
        int32_t g, ord;
        float t;
        double y;
        memcpy(&g, rec, sizeof g);
        memcpy(&ord, rec + sizeof g, sizeof ord);
        memcpy(&t, rec + REC_HEAD + (size_t)nterm * sizeof t, sizeof t);
        y = unscale_out(&gp[g], (double)t);
        for (s = 0; s < nseed; s++) {
            long at = (long)g * nseed + s;
            double p = unscale_out(&gp[g],
                (double)net_forward(nets[at], (const smb_real *)(void *)(rec + REC_HEAD))[0]);
            switch (rowrole(g, ord, s)) {
            case ROLE_REPORT: sse_he[at] += (p - y) * (p - y); n_he[at]++; break;
            case ROLE_TRAIN:  sse_tr[at] += (p - y) * (p - y); n_tr[at]++; break;
            default: break;                       /* a stop row is in neither figure */
            }
        }
    }

    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        double m = 0, v = 0, mid;
        long med = 0;
        int nohold = 0;
        if (g->n < MINROWS) {
            fprintf(stderr, "bpnn: group %s has %ld rows. Under %d the fitted, stopping and\n"
                            "reported samples are all too small to mean anything, so the group is\n"
                            "skipped. Pool it with another, or fit it with linearr.\n",
                    g->name, g->n, MINROWS);
            continue;
        }
        nohold = 0;
        for (s = 0; s < nseed; s++) {
            long at = i * nseed + s;
            /* No fallback to the training error under a column headed held-out: a refit with
             * no reported rows cannot be scored, and the group is dropped below. */
            if (n_he[at] < 1) { nohold = 1; break; }
            held[s] = sqrt(sse_he[at] / (double)n_he[at]);
            if (s == 0 || held[s] < g->best_held) g->best_held = held[s];
            m += held[s];
            srt[s] = held[s];
        }
        if (nohold) {
            fprintf(stderr, "bpnn: group %s had a refit with no rows left to report on. With\n"
                            "%ld rows and --holdout %g the split cannot be made; the group is\n"
                            "not fitted.\n", g->name, g->n, holdout);
            continue;
        }
        m /= (double)nseed;
        for (s = 0; s < nseed; s++) v += (held[s] - m) * (held[s] - m);
        g->run_sd = nseed > 1 ? sqrt(v / (double)(nseed - 1)) : 0.0;
        qsort(srt, (size_t)nseed, sizeof *srt, cmpd);
        mid = srt[(nseed - 1) / 2];              /* lower middle; see fit_group */
        for (s = 0; s < nseed; s++) if (held[s] == mid) { med = s; break; }
        g->net = nets[i * nseed + med];
        nets[i * nseed + med] = NULL;
        g->held_rmse = held[med];
        g->train_rmse = sqrt(sse_tr[i * nseed + med] / (double)n_tr[i * nseed + med]);
        g->nheld = n_he[i * nseed + med];
        g->nseed = nseed;
        g->epochs_ran = ran[i * nseed + med] ? ran[i * nseed + med] : epochs;
        {   /* the spread of the reported rows, for the variance explained */
            double q = sse_he[i * nseed + med];
            (void)q;
            g->hsd = g->ysd;   /* stream path: per-row roles are not retained past this pass */
        }
    }

    /* If the file is already ordered by the response, a window smaller than the file leaves
     * that order nearly intact and the fit sees it. Report it: the alternative is a worse fit
     * with no stated cause. */
    if (sorted_steps > 0 && sorted_runs > (long)(0.99 * (double)sorted_steps) && nrow > bufrows)
        fprintf(stderr, "bpnn: %s is sorted by %s, or nearly so, and the shuffle window holds\n"
                        "%ld of its %ld rows, so each pass trains on close to that order. Shuffle\n"
                        "the file, or raise --buffer above the row count.\n",
                path, response, bufrows, nrow);
    rc = 0;
done:
    if (cache) fclose(cache);
    if (sh) for (s = 0; s < nseed; s++) free(sh[s].buf);
    if (trs) for (i = 0; i < k; i++) if (trs[i]) trainer_free(trs[i]);
    if (bests) for (i = 0; i < k; i++) if (bests[i]) net_free(bests[i]);
    if (nets) for (i = 0; i < k; i++) if (nets[i]) net_free(nets[i]);
    free(nets); free(trs); free(bests); free(sh); free(rec); free(out);
    free(sse_tr); free(sse_he); free(n_tr); free(n_he); free(held); free(srt);
    free(sse_st); free(n_st); free(bestv); free(n_best); free(since); free(ran); free(done);
    free(nojudge);
    return rc;
}

/* ----------------------------------------------------------- the model file */

/* The share of the response's variance the fit accounts for: 1 - MSE/Var(y), against the group's
 * own mean as the baseline. Zero means the fit is worth no more than that mean, and it goes
 * negative when the fit is worse than it. */
static double explained(const Group *g)
{
    if (g->hsd <= 0) return 0.0;
    return 1.0 - (g->held_rmse * g->held_rmse) / (g->hsd * g->hsd);
}


static void write_model(void)
{
    long i, j;
    size_t l, k;
    printf("# bpnn model: one network per group, fitted by ./bpnn -t\n");
    printf("# Read the diag line before using it. train and held are RMSE in %s;\n", response);
    printf("# sd is the spread over refits and floor=2.77*sd is the smallest difference\n");
    printf("# between two configurations this pipeline can resolve at all.\n");
    printf("BPNN 1\n");
    printf("response %s\n", response);
    printf("terms %ld", nterm);
    for (j = 0; j < nterm; j++) printf(" %s", term[j]);
    printf("\n");
    printf("hyper hidden=%ld epochs=%ld rate=%g momentum=%g act=%d holdout=%g seeds=%ld decay=%g\n",
           hidden, epochs, rate, momentum, activation, holdout, nseed, decay);
    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        Net *n = g->net;
        if (!n) continue;
        printf("GROUP %s rows=%ld held=%ld\n", g->name, g->n, g->nheld);
        printf("diag train=%.6g held=%.6g sd=%.6g floor=%.6g best=%.6g expl=%.4f\n",
               g->train_rmse, g->held_rmse, g->run_sd, 2.7718 * g->run_sd, g->best_held,
               explained(g));
        /* 17 significant digits round-trip a double exactly, which the ranges must: they are
         * subtracted from the case at scoring time, so a lost digit here is a lost digit there. */
        printf("target %.17g %.17g\n", g->tlo, g->thi);
        for (j = 0; j < nterm; j++) printf("range %.17g %.17g\n", g->lo[j], g->hi[j]);
        printf("net %zu %d", n->nlayers, n->activation);
        for (l = 0; l < n->nlayers; l++) printf(" %zu", n->dim[l]);
        printf("\n");
        for (l = 1; l < n->nlayers; l++) {
            size_t ws = net_layer_wsize(n, l), bs = net_layer_bsize(n, l);
            for (k = 0; k < ws; k++) printf("w %.9g\n", (double)n->w[l][k]);
            for (k = 0; k < bs; k++) printf("b %.9g\n", (double)n->b[l][k]);
        }
        printf("END\n");
    }
}

static int read_model(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[LINELEN];
    Group *g = NULL;
    long ri = 0;
    size_t l = 1, wi = 0, bi = 0;
    if (!f) { fprintf(stderr, "bpnn: cannot open %s\n", path); return -1; }
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        if (!strncmp(line, "response ", 9)) {
            sscanf(line + 9, "%63s", response);
        } else if (!strncmp(line, "terms ", 6)) {
            char *p = line + 6; long j;
            nterm = strtol(p, &p, 10);
            if (nterm < 1 || nterm > MAXTERM) { fclose(f); return -1; }
            for (j = 0; j < nterm; j++) { while (*p == ' ') p++; sscanf(p, "%63s", term[j]);
                                          while (*p && *p != ' ') p++; }
        } else if (!strncmp(line, "GROUP ", 6)) {
            char nm[NAMELEN];
            long gi;
            if (sscanf(line + 6, "%63s", nm) != 1) { fclose(f); return -1; }
            gi = group_of(nm);
            if (gi < 0) { fclose(f); return -1; }     /* more groups than the build holds */
            g = &gp[gi];
            ri = 0; l = 1; wi = bi = 0;
        } else if (g && !strncmp(line, "diag ", 5)) {
            sscanf(line + 5, "train=%lf held=%lf sd=%lf",
                   &g->train_rmse, &g->held_rmse, &g->run_sd);
        } else if (g && !strncmp(line, "target ", 7)) {
            sscanf(line + 7, "%lf %lf", &g->tlo, &g->thi);
        } else if (g && !strncmp(line, "range ", 6)) {
            if (ri < nterm) sscanf(line + 6, "%lf %lf", &g->lo[ri], &g->hi[ri]);
            ri++;
        } else if (g && !strncmp(line, "net ", 4)) {
            size_t nl, dims[SMB_MAX_LAYERS]; int act; char *p = line + 4; size_t i;
            nl = (size_t)strtol(p, &p, 10);
            act = (int)strtol(p, &p, 10);
            if (nl < 2 || nl > SMB_MAX_LAYERS) { fclose(f); return -1; }
            for (i = 0; i < nl; i++) dims[i] = (size_t)strtol(p, &p, 10);
            g->net = net_new(dims, nl);
            if (!g->net) { fclose(f); return -1; }
            g->net->activation = act;
            l = 1; wi = bi = 0;
        } else if (g && g->net && line[0] == 'w' && line[1] == ' ') {
            while (l < g->net->nlayers && wi >= net_layer_wsize(g->net, l)) { l++; wi = bi = 0; }
            if (l < g->net->nlayers) g->net->w[l][wi++] = (smb_real)atof(line + 2);
        } else if (g && g->net && line[0] == 'b' && line[1] == ' ') {
            if (l < g->net->nlayers && bi < net_layer_bsize(g->net, l)) {
                g->net->b[l][bi++] = (smb_real)atof(line + 2);
                if (bi >= net_layer_bsize(g->net, l)) { l++; wi = bi = 0; }
            }
        }
    }
    fclose(f);
    return ngroup > 0 ? 0 : -1;
}

/* ------------------------------------------------------------------- score  */

static int score(const char *path, int argc, char **argv, int from)
{
    double raw[MAXTERM];
    int given[MAXTERM];
    const char *grp = NULL;
    long i, out = 0;
    Group *g;
    smb_real xn[MAXTERM];
    if (read_model(path) != 0) { fprintf(stderr, "bpnn: %s is not a bpnn model\n", path); return 1; }
    memset(given, 0, sizeof given);
    for (i = 0; i < nterm; i++) raw[i] = 0.0;
    for (i = from; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) { grp = argv[i]; continue; }
        *eq = '\0';
        { long j, hit = -1;
          char *end;
          double v;
          for (j = 0; j < nterm; j++) if (!strcmp(term[j], argv[i])) hit = j;
          if (hit < 0) { fprintf(stderr, "bpnn: '%s' is not a term in this model\n", argv[i]); return 2; }
          v = strtod(eq + 1, &end);
          if (end == eq + 1 || *end != '\0' || !isfinite(v)) {
              fprintf(stderr, "bpnn: %s=%s is not a finite number\n", argv[i], eq + 1);
              return 2;
          }
          raw[hit] = v; given[hit] = 1; }
    }
    if (!grp) { fprintf(stderr, "bpnn: name the group to score, e.g. ./bpnn -c %s A x=3\n", path); return 2; }
    /* An unnamed term is scored as zero, a value the caller did not supply and usually outside
     * the range the group was trained on. Refuse instead of answering a different case. */
    { long j, missing = 0;
      for (j = 0; j < nterm; j++) if (!given[j]) missing++;
      if (missing) {
          fprintf(stderr, "bpnn: this model has %ld term%s and %ld %s not given:",
                  nterm, nterm == 1 ? "" : "s", missing,
                  missing == 1 ? "was" : "were");
          for (j = 0; j < nterm; j++) if (!given[j]) fprintf(stderr, " %s", term[j]);
          fprintf(stderr, "\nA missing term would be scored as zero, which is a different case\n"
                          "from the one you asked about.\n");
          return 2;
      } }
    { long j, hit = -1;
      for (j = 0; j < ngroup; j++) if (!strcmp(gp[j].name, grp)) hit = j;
      if (hit < 0) { fprintf(stderr, "bpnn: group '%s' is not in this model\n", grp); return 2; }
      g = &gp[hit]; }
    if (!g->net) { fprintf(stderr, "bpnn: group '%s' has no fitted network\n", grp); return 2; }
    scale_in(g, raw, xn);
    { double a = (double)net_forward(g->net, xn)[0];
      double p = unscale_out(g, a);
      printf("%s %s = %.*g\n", grp, response, sigdigits(p, g->thi - g->tlo), p);
      /* The output unit saturates at 0 and 1, which is 12.5% of the fitted range past each end.
       * Within a hair of that the answer is the ceiling, not a prediction, and every input can
       * be in range while the output is not. */
      if (a < 0.02 || a > 0.98)
          printf("AT THE LIMIT OF WHAT THIS MODEL CAN SAY: the output unit is saturated, so\n"
                 "%.6g is the most extreme %s it can return (%s was fitted over [%.6g, %.6g]).\n"
                 "The true value may be far past it.\n",
                 p, response, response, g->tlo, g->thi); }
    printf("held-out RMSE at fit time %.6g, spread over refits %.6g\n", g->held_rmse, g->run_sd);
    /* the check linearr says a regression cannot make */
    for (i = 0; i < nterm; i++) {
        double span = g->hi[i] - g->lo[i];
        if (!given[i]) continue;
        if (raw[i] < g->lo[i] || raw[i] > g->hi[i]) {
            double over = raw[i] < g->lo[i] ? (g->lo[i] - raw[i]) : (raw[i] - g->hi[i]);
            printf("OUTSIDE THE TRAINING RANGE: %s=%g, trained on [%g, %g], out by %.3g of that span\n",
                   term[i], raw[i], g->lo[i], g->hi[i], over / span);
            out++;
        }
    }
    if (out)
        printf("A network does not extrapolate. Past the range above its units saturate and it\n"
               "returns a flat value with no warning of its own, so treat this number as a guess.\n");
    for (i = 0; i < ngroup; i++) if (gp[i].net) net_free(gp[i].net);
    return 0;
}

/* -------------------------------------------------------------------- misc  */

static void report(void)
{
    long i;
    fprintf(stderr, "%-10s %6s %6s %8s %7s %10s %10s %10s %10s %6s\n",
            "group", "rows", "held", "weights", "epochs", "train", "held-out", "refit sd",
            "floor", "expl");
    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        long nw, ntr;
        if (!g->net) continue;
        nw = (long)net_nweights(g->net);
        ntr = g->n - g->nheld;
        if (!isfinite(g->held_rmse) || !isfinite(g->train_rmse)) {
            fprintf(stderr, "%-10s %6ld %6ld %8ld %7ld   DIVERGED: the weights left the range of a\n"
                            "  number. The learning rate (-r %g), the momentum (-m %g) or the decay\n"
                            "  (--decay %g) is too large for this data. Nothing in this group's model\n"
                            "  is usable.\n",
                    g->name, g->n, g->nheld, nw, g->epochs_ran ? g->epochs_ran : epochs,
                    rate, momentum, decay);
            continue;
        }
        fprintf(stderr, "%-10s %6ld %6ld %8ld %7ld %10.5g %10.5g %10.5g %10.5g %5.0f%%\n",
                g->name, g->n, g->nheld, nw, g->epochs_ran ? g->epochs_ran : epochs,
                g->train_rmse, g->held_rmse, g->run_sd, 2.7718 * g->run_sd,
                100.0 * explained(g));
        /* the network's analogue of a regression's degrees of freedom: least squares pins the
         * terms it cannot identify and says how many, but a network has no such notion and will
         * quietly spend a free parameter per row, so the count has to be put next to the rows */
        if (nw > ntr)
            fprintf(stderr, "  %ld weights fitted to %ld training rows. There are more free\n"
                            "  parameters than examples, so some of what it learned is the rows\n"
                            "  themselves. Reduce -H, or use linearr if the relation may be linear.\n",
                            nw, ntr);
        /* Underfitting has no loud symptom: both errors are simply large, and large is only
         * meaningful next to the spread of the thing being predicted. */
        if (!g->flat && explained(g) < 0.05)
            fprintf(stderr, "  this fit explains %.0f%% of the variance in %s. The group's own mean\n"
                            "  scores %.4g and this model scores %.4g, so it is barely using its\n"
                            "  inputs. Try more hidden units (-H), a longer run (-e), or less\n"
                            "  --decay; or the relation may not be in these columns.\n",
                    100.0 * explained(g), response, g->ysd, g->held_rmse);
        if (g->tail > 20.0)
            fprintf(stderr, "  %s REACHES %.0f INTERQUARTILE RANGES past its own quartiles in this\n"
                            "  group. The target is scaled by its smallest and largest value, so\n"
                            "  every ordinary row is squeezed into a sliver of the output's range\n"
                            "  and the fit cannot separate them. The variance explained beside it\n"
                            "  is measured against that same spread and will look high regardless.\n"
                            "  Drop or cap the extreme rows before fitting.\n",
                            response, g->tail);
        if (g->flat)
            fprintf(stderr, "  %s NEVER VARIES in this group: every row has the same value, so the\n"
                            "  errors above are near zero because there was nothing to predict.\n",
                            response);
        if (g->nheld == 0)
            fprintf(stderr, "  NO HELD-OUT ROWS (--holdout 0): the 'held-out' column above is the\n"
                            "  training error and does not measure generalization at all.\n");
        else if (g->held_rmse > 2.0 * g->train_rmse && g->train_rmse > 0)
            fprintf(stderr, "  held-out error is %.1fx the training error: this network is fitting\n"
                            "  the rows rather than the relation. Fewer hidden units, or more rows.\n",
                            g->held_rmse / g->train_rmse);
        if (g->run_sd > 0.25 * g->held_rmse)
            fprintf(stderr, "  the spread over refits is %.0f%% of the error itself, so a single fit\n"
                            "  of this configuration does not pin down its quality.\n",
                            100.0 * g->run_sd / g->held_rmse);
    }
    /* What this fit cost to hold, once it is large enough to matter. The streaming path does not
     * pay it, and the figure is the one that decides whether a larger file will fit at all. */
    if (!streaming && rowcost() * (double)nrow >= 67108864.0)
        fprintf(stderr, "\nthe row store held %ld rows at %.0f bytes each, %.3g GB, and the sort\n"
                        "briefly needed a second copy. --stream fits the same model without\n"
                        "holding the rows.\n",
                nrow, rowcost(), rowcost() * (double)nrow / 1073741824.0);
}

static int selftest(void)
{
    int fail = 0;
    Group g;
    memset(&g, 0, sizeof g);
    nterm = 1;
    g.lo[0] = -6; g.hi[0] = 6; g.tlo = 10; g.thi = 46;
    /* scaling must round-trip exactly at both ends and the middle */
    { smb_real x[1]; double r[1];
      r[0] = -6; scale_in(&g, r, x);
      printf("input scaling at the low end: got %.6f want 0.0  %s\n", (double)x[0],
             fabs((double)x[0]) < 1e-6 ? "PASS" : (fail=1,"FAIL"));
      r[0] = 6; scale_in(&g, r, x);
      printf("input scaling at the high end: got %.6f want 1.0  %s\n", (double)x[0],
             fabs((double)x[0]-1.0) < 1e-6 ? "PASS" : (fail=1,"FAIL")); }
    { double a = (double)scale_out(&g, 10.0), b = (double)scale_out(&g, 46.0);
      printf("target maps into [%.2f, %.2f]: got [%.4f, %.4f]  %s\n", TLO, THI, a, b,
             (fabs(a-TLO) < 1e-6 && fabs(b-THI) < 1e-6) ? "PASS" : (fail=1,"FAIL")); }
    { double t = 28.0, back = unscale_out(&g, (double)scale_out(&g, t));
      printf("target round-trip of %.1f: got %.5f  %s\n", t, back,
             fabs(back-t) < 1e-4 ? "PASS" : (fail=1,"FAIL")); }
    /* the resolution floor is 1.959964*sqrt(2), the two-sided interval on a difference of two means */
    { double w = Z95 * sqrt(2.0);
      printf("resolution floor coefficient 2.7718 vs %.6f  %s\n", w,
             fabs(2.7718 - w) < 1e-4 ? "PASS" : (fail=1,"FAIL")); }
    /* a constant column must not divide by zero */
    { smb_real x[1]; double r[1] = {5};
      Group c; memset(&c, 0, sizeof c);
      c.first = 0; c.n = 0; c.lo[0] = 5; c.hi[0] = 5;
      if (c.hi[0] <= c.lo[0]) c.hi[0] = c.lo[0] + 1.0;
      scale_in(&c, r, x);
      printf("a constant column scales finitely: got %.6f  %s\n", (double)x[0],
             (x[0] == x[0] && fabs((double)x[0]) < 1e9) ? "PASS" : (fail=1,"FAIL")); }
    /* the network must be able to represent the parabola it exists for: 13 points, exact fit */
    {
        long ord[13], i;
        Group p; Net *n; double e;
        memset(&p, 0, sizeof p);
        nterm = 1;
        if (grow(13) != 0) { printf("selftest: out of memory\n"); return 1; }
        for (i = 0; i < 13; i++) {
            double x = (double)(i - 6);
            xs[i] = x; ys[i] = x * x + 10.0; ord[i] = i;      /* nterm is 1, so stride is 1 */
        }
        nrow = 13; p.first = 0; p.n = 13;
        ranges(&p);
        n = train_one(&p, ord, 13, 13, 1u, NULL);   /* no stop rows: the full epochs */
        if (!n) { printf("selftest: cannot train\n"); return 1; }
        e = rmse(&p, n, ord, 0, 13);
        printf("fitting y=x^2+10 on 13 points: RMSE %.4f (span %g) %s\n", e, p.thi - p.tlo,
               e < 0.05 * (p.thi - p.tlo) ? "PASS" : (fail=1,"FAIL"));
        net_free(n);
    }
    printf("\nself-test %s\n", fail ? "FAILED" : "PASSED");
    return fail;
}

static void bytes_out(const char *label, double b)
{
    if (b >= 1073741824.0)   printf("  %-22s %.3g GB\n", label, b / 1073741824.0);
    else if (b >= 1048576.0) printf("  %-22s %.3g MB\n", label, b / 1048576.0);
    else if (b >= 1024.0)    printf("  %-22s %.3g kB\n", label, b / 1024.0);
    else                     printf("  %-22s %.0f bytes\n", label, b);
}

/* What a fit of a given shape costs, printed rather than described. No figure below takes a row
 * count, which is the property --stream is for. */
static int footprint(long terms, long groups)
{
    double w, net, trainer, per, nets, wins, rowbytes, cachebytes;

    if (terms < 1 || groups < 1) {
        fprintf(stderr, "bpnn: --footprint takes a term count and a group count, both above 0\n");
        return 2;
    }
    w       = (double)terms * (double)hidden + (double)hidden;        /* weights */
    net     = (w + (double)hidden + 1.0) * (double)sizeof(smb_real);  /* plus the biases */
    trainer = net + ((double)hidden + 1.0) * (double)sizeof(smb_real);/* mirrors, plus the betas */
    per     = net + trainer;
    nets    = per * (double)groups * (double)nseed + (double)sizeof(Group) * (double)groups;
    wins    = (double)bufrows * (double)nseed
              * (2.0 * (double)sizeof(int32_t) + ((double)terms + 1.0) * (double)sizeof(float));
    rowbytes   = ((double)terms + 1.0) * (double)sizeof(double) + (double)sizeof(long);
    cachebytes = 2.0 * (double)sizeof(int32_t) + ((double)terms + 1.0) * (double)sizeof(float);

    printf("%ld terms, %ld groups, %ld hidden units, %ld refits\n\n",
           terms, groups, hidden, nseed);
    printf("fitting with --stream\n");
    bytes_out("per group per refit", per);
    bytes_out("networks in total", nets);
    bytes_out("shuffle windows", wins);
    bytes_out("total", nets + wins);
    printf("\nfitting without --stream, add the row store, which does take a row count:\n");
    bytes_out("per row", rowbytes);
    printf("\nThe --stream cache is a temporary file of %.0f bytes a row, removed on exit.\n",
           cachebytes);
    if (groups > MAXGROUP)
        printf("\nNote: this build fits at most %d groups, and %ld were asked for.\n",
               MAXGROUP, groups);
    return 0;
}

static void usage(void)
{
    printf("bpnn -- a backpropagation network for tabular data, where a line is not enough\n\n");
    printf("  bpnn -t data.csv > model.txt     fit one network per group\n");
    printf("  bpnn -c model.txt A x=3          score one case\n");
    printf("  bpnn --selftest                  check the arithmetic\n\n");
    printf("input CSV is linearr's: a header, then GROUP, the response, one column per term.\n\n");
    printf("  -H N        hidden units, --size N also (R's nnet calls it size)\n");
    printf("              (default %ld)\n", hidden);
    printf("  -e N        epochs (default %ld)\n", epochs);
    printf("  -s N        refits, to measure the spread (default %ld)\n", nseed);
    printf("  -r X -m X   learning rate, momentum (default %g, %g)\n", rate, momentum);
    printf("  -a NAME     hidden activation: sigmoid, tanh, relu (default tanh)\n");
    printf("  --holdout X fraction of rows kept out of the fit (default %g; 0 disables\n", holdout);
    printf("              it and makes the reported error meaningless as generalization)\n");
    printf("  --decay X   weight decay: each step also pulls every weight toward zero\n");
    printf("              by rate*X*w (default %g, no decay)\n", decay);
    printf("  --patience N  stop a fit after N checks, 25 epochs apart, with no\n");
    printf("              improvement on the rows kept back for that (default %ld; 0\n", patience);
    printf("              runs every epoch of -e)\n");
    printf("  --stream    fit without holding the rows in memory: two passes over the\n");
    printf("              file, then one per epoch over a cache. Different training\n");
    printf("              order, so different numbers from the default path.\n");
    printf("  --buffer N  rows in the shuffle window under --stream (default %ld)\n", bufrows);
    printf("  --cache F   keep --stream's scaled rows in F, so a later run of any shape\n");
    printf("              skips both passes over the CSV. Rebuilt when the input's size\n");
    printf("              or modification time changes.\n");
    printf("  --footprint TERMS GROUPS   what a fit of that shape costs in memory\n");
}

/* An option's argument. Text that is not a number is refused rather than read as zero: to the
 * caller `-H six` means six hidden units, and to atol it means none. */
static int optnum(int argc, char **argv, int *i, double *out)
{
    char *end;
    double v;
    if (*i + 1 >= argc) {
        fprintf(stderr, "bpnn: %s needs a value after it\n", argv[*i]);
        return -1;
    }
    v = strtod(argv[*i + 1], &end);
    if (end == argv[*i + 1] || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "bpnn: %s %s is not a number\n", argv[*i], argv[*i + 1]);
        return -1;
    }
    /* Casting a double outside long's range to long is undefined, and every caller casts. */
    if (v > 9.0e15 || v < -9.0e15) {
        fprintf(stderr, "bpnn: %s %s is outside the range this program can use\n",
                argv[*i], argv[*i + 1]);
        return -1;
    }
    (*i)++;
    *out = v;
    return 0;
}

static int optstr(int argc, char **argv, int *i, const char **out)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "bpnn: %s needs a value after it\n", argv[*i]);
        return -1;
    }
    *out = argv[++(*i)];
    return 0;
}

int main(int argc, char **argv)
{
    int i, mode = 0, first = 0;
    const char *path = NULL;
    long g;
    int rc = 0;

    for (i = 1; i < argc; i++) {
        double v;
        const char *s;
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (!strcmp(argv[i], "-t")) {
            if (optstr(argc, argv, &i, &path) != 0) return 2;
            mode = 1;
        } else if (!strcmp(argv[i], "-c")) {
            if (optstr(argc, argv, &i, &path) != 0) return 2;
            mode = 2; first = i + 1;
            break;                     /* everything after the model file is the case */
        } else if (!strcmp(argv[i], "-H") || !strcmp(argv[i], "--size")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            hidden = (long)v;
        } else if (!strcmp(argv[i], "-e")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            epochs = (long)v;
        } else if (!strcmp(argv[i], "-s")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            nseed = (long)v;
        } else if (!strcmp(argv[i], "-r")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            rate = v;
        } else if (!strcmp(argv[i], "-m")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            momentum = v;
        } else if (!strcmp(argv[i], "--holdout")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            holdout = v;
        } else if (!strcmp(argv[i], "--patience")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            patience = (long)v;
        } else if (!strcmp(argv[i], "--decay")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            decay = v;
        } else if (!strcmp(argv[i], "--stream")) {
            streaming = 1;
        } else if (!strcmp(argv[i], "--cache")) {
            if (optstr(argc, argv, &i, &cachepath) != 0) return 2;
        } else if (!strcmp(argv[i], "--buffer")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            bufrows = (long)v;
        } else if (!strcmp(argv[i], "--footprint")) {
            double t, n;
            if (optnum(argc, argv, &i, &t) != 0) return 2;
            if (optnum(argc, argv, &i, &n) != 0) return 2;
            return footprint((long)t, (long)n);
        } else if (!strcmp(argv[i], "-a")) {
            if (optstr(argc, argv, &i, &s) != 0) return 2;
            if (!strcmp(s, "sigmoid"))   activation = ACT_SIGMOID;
            else if (!strcmp(s, "tanh")) activation = ACT_TANH;
            else if (!strcmp(s, "relu")) activation = ACT_RELU;
            else {
                fprintf(stderr, "bpnn: -a %s is not an activation. Use sigmoid, tanh or relu.\n", s);
                return 2;
            }
        } else {
            fprintf(stderr, "bpnn: %s is not an option this program has. Try --help.\n", argv[i]);
            return 2;
        }
    }
    if (mode == 2) return score(path, argc, argv, first);
    if (mode != 1) { usage(); return 2; }

    if (hidden < 1 || epochs < 1 || nseed < 1) {
        fprintf(stderr, "bpnn: hidden units, epochs and refits must each be at least 1\n");
        return 2;
    }
    if (decay < 0) {
        fprintf(stderr, "bpnn: --decay is a penalty on the size of the weights and cannot be\n"
                        "negative; 0, the default, is no decay.\n");
        return 2;
    }
    /* The decay part of a step is -rate*lambda*w. At rate*lambda >= 1 it carries the weight
     * past zero and further each step, which diverges rather than regularises. */
    /* The decay term is subtracted once per ROW, and it sits inside the retained momentum
     * delta, so the shrink a weight actually sees over an epoch of n rows is
     * rate*lambda*n/(1-momentum). At the defaults that is 13,000 times lambda on 400 rows, which
     * is why lambda above about 1e-4 destroys the fit. The bound below is on that quantity, not
     * on lambda, and the row count is not known until the file is read, so it is checked again
     * per group once it is. */
    if (decay * rate >= 1.0) {
        fprintf(stderr, "bpnn: --decay %g at -r %g shrinks every weight by %g of itself per row.\n"
                        "At 1 or more it overshoots zero and diverges. The useful range is far\n"
                        "smaller: try 1e-5 to 1e-4.\n", decay, rate, decay * rate);
        return 2;
    }
    if (rate <= 0 || momentum < 0 || momentum >= 1) {
        fprintf(stderr, "bpnn: the learning rate must be positive and the momentum in [0, 1)\n");
        return 2;
    }
    if (holdout < 0 || holdout >= 1) {
        fprintf(stderr, "bpnn: --holdout is a fraction of the rows, so it must be in [0, 1).\n"
                        "Use 0 to fit on every row, and read the warning it prints.\n");
        return 2;
    }
    if (bufrows < 1) {
        fprintf(stderr, "bpnn: --buffer is a number of rows and must be at least 1\n");
        return 2;
    }
    if (cachepath && !streaming) {
        fprintf(stderr, "bpnn: --cache holds the scaled rows --stream reads, so it needs\n"
                        "--stream. Without it the rows are held in memory and there is\n"
                        "nothing to cache.\n");
        return 2;
    }
    if (streaming) {
        if (fit_stream(path) != 0) { rc = 1; goto out; }
        write_model();
        report();
        goto out;
    }
    if (read_csv(path) != 0) { rc = 1; goto out; }
    if (nrow == 0) { fprintf(stderr, "bpnn: %s has a header and no data rows\n", path); rc = 1; goto out; }
    if (regroup() != 0) { rc = 1; goto out; }
    for (g = 0; g < ngroup; g++) {
        if (gp[g].n < MINROWS) {
            fprintf(stderr, "bpnn: group %s has %ld rows. Under %d the fitted, stopping and\n"
                            "reported samples are all too small to mean anything, so the group is\n"
                            "skipped. Pool it with another, or fit it with linearr.\n",
                    gp[g].name, gp[g].n, MINROWS);
            continue;
        }
        if (fit_group(&gp[g]) != 0) {
            fprintf(stderr, "bpnn: cannot fit group %s\n", gp[g].name);
            rc = 1; goto out;
        }
    }
    write_model();
    report();
out:
    for (g = 0; g < ngroup; g++) if (gp[g].net) net_free(gp[g].net);
    free(xs); free(ys); free(rowgrp);
    return rc;
}
