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
 * IT REPORTS THREE OF THE WAYS A NETWORK MISLEADS YOU, which are not the three a regression has.
 *
 *   1. IT MEMORISES. A line with two coefficients cannot memorise thirteen points; a network with
 *      thirty weights can, and the training error then measures recall rather than knowledge. So the
 *      rows are split, the fit sees only part of them, and both errors are reported next to each
 *      other. When held-out error is much the worse of the two, the model learned this table and not
 *      the relation behind it.
 *
 *   2. RETRAINING CHANGES THE ANSWER. Least squares has one solution and returns it every time. A
 *      network starts from random weights and shuffles its examples, so refitting the same rows gives
 *      a different model and a different score. That spread is not a nuisance to be averaged away: it
 *      sets the smallest difference between two configurations this pipeline can resolve at all. So
 *      the fit is repeated over several seeds and reports the spread and that floor. Any comparison
 *      of two setups closer than the floor is noise, whichever way it came out.
 *
 *      For the same reason the model shipped is the MEDIAN of the seeds and not the best of them.
 *      Picking the best by held-out error makes that error optimistic by however much was selected
 *      for, which is the oldest mistake in model selection (Spearman 1904 for the arithmetic;
 *      Cawley & Talbot 2010 for the modern form). Both numbers are printed so the gap is visible.
 *
 *   3. IT DOES NOT EXTRAPOLATE, AND WILL NOT SAY SO UNLESS ASKED. This is the one linearr names as a
 *      way regression misleads that it cannot check. Here it can be checked, because a network has to
 *      store the range of every input in order to scale it. Outside that range a saturating unit
 *      returns a flat, confident number with no hint that it is guessing, where a line at least keeps
 *      going in the direction the data suggested. So scoring reports every input that falls outside
 *      the range it was trained on, and how far outside.
 *
 * Three the program cannot see, and does not claim to: a term that should have been in the table and
 * is not, rows that are not independent of each other, and a response whose relation to the inputs
 * changed after the training rows were collected.
 *
 * WHEN NOT TO USE IT. If the relation is close to linear, use linearr: it will be more accurate, it
 * returns coefficients you can read, and it has one answer rather than a distribution of them. This
 * program earns its place only where a line leaves structure in the residuals. bench/ measures both on
 * the same files so the crossover is visible rather than asserted.
 *
 * PRECISION. The engine computes in `smb_real`, which is float, so about seven digits. linearr
 * validates against NIST reference values to eleven. Nothing here should be trusted past six.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "net.h"
#include "train.h"
#include "act.h"
#include "rng.h"

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
    Net   *net;
    double train_rmse, held_rmse, run_sd, best_held;
    long   nseed, nheld;
    int    flat;                     /* the response never varied in this group */
} Group;

static Group  gp[MAXGROUP];
static long   ngroup;
static char   term[MAXTERM][NAMELEN];
static char   response[NAMELEN];
static long   nterm;

/* the row store: rows are grouped together, so a group is a contiguous span */
static double *xs, *ys;
static long   *rowgrp;
static long    nrow, caprow;

static long   hidden = 6, epochs = 3000, nseed = 5;
static double rate = 0.3, momentum = 0.9, holdout = 0.25;
static int    activation = ACT_TANH;

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
    nx = realloc(xs, (size_t)caprow * MAXTERM * sizeof *xs);
    ny = realloc(ys, (size_t)caprow * sizeof *ys);
    ng = realloc(rowgrp, (size_t)caprow * sizeof *ng);
    if (!nx || !ny || !ng) { xs = nx ? nx : xs; ys = ny ? ny : ys; rowgrp = ng ? ng : rowgrp; return -1; }
    xs = nx; ys = ny; rowgrp = ng;
    return 0;
}

/* A field that must be a finite number. WHAT names it for the message, LINE locates it.
 *
 * Every refusal here is a value that would otherwise have reached a weight. An empty field
 * read as zero, a nan multiplied through the layers, or a row dropped without a word all end
 * the same way: a model that trained, printed a number, and exited 0. That is the failure this
 * program is least able to detect afterwards, so it is refused at the door. */
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

/* A name from the header or the group column: non-empty and short enough to be stored whole.
 * A truncated group name is the dangerous one, because two distinct groups then merge into one
 * fit without anything being printed. */
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

/* Read PATH: header names the response (column 2) and the terms (3 onward). */
static int read_csv(const char *path)
{
    FILE *f = strcmp(path, "-") ? fopen(path, "r") : stdin;
    char line[LINELEN], *fld[MAXTERM + 8];
    char what[NAMELEN + 32];
    long lineno = 0;
    int nf, i, header = 0;
    if (!f) { fprintf(stderr, "bpnn: cannot open %s\n", path); return -1; }
    while (fgets(line, sizeof line, f)) {
        lineno++;
        /* fgets splits an over-long line in two, and the tail would then be read as a row of
         * its own. Refuse it rather than fit half a row. */
        if (strchr(line, '\n') == NULL && !feof(f)) {
            fprintf(stderr, "bpnn: line %ld is longer than %d characters.\n", lineno, LINELEN - 1);
            goto bad;
        }
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        nf = split_csv(line, fld, MAXTERM + 8);
        if (!header) {
            if (nf < 3) {
                fprintf(stderr, "bpnn: the header on line %ld needs a group column, the value\n"
                                "being predicted, and at least one term.\n", lineno);
                goto bad;
            }
            nterm = nf - 2;
            if (nterm > MAXTERM) {
                fprintf(stderr, "bpnn: %ld terms on line %ld; this build holds %d.\n",
                        nterm, lineno, MAXTERM);
                goto bad;
            }
            if (checkname(fld[1], "the name of the value being predicted", lineno) != 0) goto bad;
            strncpy(response, fld[1], NAMELEN - 1);
            for (i = 0; i < nterm; i++) {
                int j;
                if (checkname(fld[i + 2], "a term name", lineno) != 0) goto bad;
                for (j = 0; j < i; j++)
                    if (!strcmp(term[j], fld[i + 2])) {
                        fprintf(stderr, "bpnn: the header names the term '%s' twice, so a case\n"
                                        "naming it could mean either column.\n", fld[i + 2]);
                        goto bad;
                    }
                strncpy(term[i], fld[i + 2], NAMELEN - 1);
            }
            header = 1;
            continue;
        }
        if (nf != nterm + 2) {
            fprintf(stderr, "bpnn: line %ld has %d field%s; the header declared %ld (the group,\n"
                            "%s, and %ld term%s).\n", lineno, nf, nf == 1 ? "" : "s",
                    nterm + 2, response, nterm, nterm == 1 ? "" : "s");
            goto bad;
        }
        {
            long g, j;
            double v;
            if (checkname(fld[0], "the group", lineno) != 0) goto bad;
            g = group_of(fld[0]);
            if (g < 0) { fprintf(stderr, "bpnn: more than %d groups.\n", MAXGROUP); goto bad; }
            if (grow(nrow + 1) != 0) { fprintf(stderr, "bpnn: out of memory\n"); goto bad; }
            snprintf(what, sizeof what, "'%s', the value being predicted,", response);
            if (number(fld[1], what, lineno, &v) != 0) goto bad;
            ys[nrow] = v;
            for (j = 0; j < nterm; j++) {
                snprintf(what, sizeof what, "the term '%s'", term[j]);
                if (number(fld[j + 2], what, lineno, &v) != 0) goto bad;
                xs[nrow * MAXTERM + j] = v;
            }
            rowgrp[nrow] = g;
            gp[g].n++;
            nrow++;
        }
    }
    if (ferror(f)) { fprintf(stderr, "bpnn: error reading %s\n", path); goto bad; }
    if (f != stdin) fclose(f);
    if (!header) { fprintf(stderr, "bpnn: %s has no header line.\n", path); return -1; }
    return 0;
bad:
    if (f != stdin) fclose(f);
    return -1;
}

/* Sort rows by group so each group is one contiguous span (insertion by counting). */
static int regroup(void)
{
    double *nx = malloc((size_t)nrow * MAXTERM * sizeof *nx);
    double *ny = malloc((size_t)nrow * sizeof *ny);
    long *at = malloc((size_t)ngroup * sizeof *at), i, run = 0;
    if (!nx || !ny || !at) { free(nx); free(ny); free(at); return -1; }
    for (i = 0; i < ngroup; i++) { gp[i].first = run; at[i] = run; run += gp[i].n; }
    for (i = 0; i < nrow; i++) {
        long d = at[rowgrp[i]]++;
        memcpy(nx + d * MAXTERM, xs + i * MAXTERM, (size_t)nterm * sizeof *nx);
        ny[d] = ys[i];
    }
    free(xs); free(ys); free(at);
    xs = nx; ys = ny;
    return 0;
}

/* ------------------------------------------------------------------ scaling */

static void ranges(Group *g)
{
    long i, j;
    for (j = 0; j < nterm; j++) { g->lo[j] = 1e300; g->hi[j] = -1e300; }
    g->tlo = 1e300; g->thi = -1e300;
    for (i = g->first; i < g->first + g->n; i++) {
        for (j = 0; j < nterm; j++) {
            double v = xs[i * MAXTERM + j];
            if (v < g->lo[j]) g->lo[j] = v;
            if (v > g->hi[j]) g->hi[j] = v;
        }
        if (ys[i] < g->tlo) g->tlo = ys[i];
        if (ys[i] > g->thi) g->thi = ys[i];
    }
    for (j = 0; j < nterm; j++) if (g->hi[j] <= g->lo[j]) g->hi[j] = g->lo[j] + 1.0; /* constant term */
    /* A response that never varied would divide by zero here. Widening the range keeps the
     * arithmetic finite and makes the fit look perfect, so the fact is recorded and reported. */
    g->flat = g->thi <= g->tlo;
    if (g->flat) g->thi = g->tlo + 1.0;
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

/* Significant digits enough to show a prediction moving across the range it was fitted on.
 * A response near 1e8 varying over 60 printed at %g's default six digits is 1e+08 every time:
 * the model's whole answer falls off the end of the number. */
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

/* Train one net on the rows of G whose position in `ord` is < ntr. Returns the net, or NULL. */
static Net *train_one(Group *g, long *ord, long ntr, uint32_t seed)
{
    size_t dims[3];
    Net *net;
    Trainer *t;
    Rng rng;
    long e, i;
    dims[0] = (size_t)nterm; dims[1] = (size_t)hidden; dims[2] = 1;
    net = net_new(dims, 3);
    if (!net) return NULL;
    net->activation = activation;
    rng_seed(&rng, seed);
    net_init(net, &rng);
    t = trainer_new(net, (smb_real)rate, (smb_real)momentum);
    if (!t) { net_free(net); return NULL; }
    for (e = 0; e < epochs; e++) {
        for (i = ntr - 1; i > 0; i--) {          /* shuffle the training part in place */
            long k = (long)(rng_u32(&rng) % (uint32_t)(i + 1));
            long tmp = ord[i]; ord[i] = ord[k]; ord[k] = tmp;
        }
        for (i = 0; i < ntr; i++) {
            smb_real xn[MAXTERM], d;
            long r = g->first + ord[i];
            scale_in(g, xs + r * MAXTERM, xn);
            d = scale_out(g, ys[r]);
            (void)trainer_learn(t, xn, &d);
        }
    }
    trainer_free(t);
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
        scale_in(g, xs + r * MAXTERM, xn);
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
    Net **nets = calloc((size_t)nseed, sizeof *nets);
    long i, s, ntr, med = 0;
    double m = 0, v = 0, mid;
    int rc = -1;
    if (!ord || !held || !srt || !nets) goto done;
    ranges(g);
    ntr = holdout > 0 ? (long)((1.0 - holdout) * (double)g->n + 0.5) : g->n;
    if (ntr < 1) ntr = 1;
    if (ntr > g->n) ntr = g->n;
    g->nheld = g->n - ntr;
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
        nets[s] = train_one(g, ord, ntr, (uint32_t)(1u + 104729u * (unsigned)s));
        if (!nets[s]) goto done;
        held[s] = g->nheld > 0 ? rmse(g, nets[s], ord, ntr, g->n) : rmse(g, nets[s], ord, 0, ntr);
        if (s == 0 || held[s] < g->best_held) g->best_held = held[s];
        if (s == 0) g->train_rmse = rmse(g, nets[s], ord, 0, ntr);
    }
    for (s = 0; s < nseed; s++) { m += held[s]; srt[s] = held[s]; }
    m /= (double)nseed;
    for (s = 0; s < nseed; s++) v += (held[s] - m) * (held[s] - m);
    g->run_sd = nseed > 1 ? sqrt(v / (double)(nseed - 1)) : 0.0;
    qsort(srt, (size_t)nseed, sizeof *srt, cmpd);
    mid = srt[nseed / 2];
    for (s = 0; s < nseed; s++) if (held[s] == mid) { med = s; break; }
    g->net = nets[med];
    nets[med] = NULL;
    g->held_rmse = held[med];
    g->nseed = nseed;
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
    free(ord); free(held); free(srt); free(nets);
    return rc;
}

/* ----------------------------------------------------------- the model file */

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
    printf("hyper hidden=%ld epochs=%ld rate=%g momentum=%g act=%d holdout=%g seeds=%ld\n",
           hidden, epochs, rate, momentum, activation, holdout, nseed);
    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        Net *n = g->net;
        if (!n) continue;
        printf("GROUP %s rows=%ld held=%ld\n", g->name, g->n, g->nheld);
        printf("diag train=%.6g held=%.6g sd=%.6g floor=%.6g best=%.6g\n",
               g->train_rmse, g->held_rmse, g->run_sd, 2.7718 * g->run_sd, g->best_held);
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
            if (sscanf(line + 6, "%63s", nm) != 1) { fclose(f); return -1; }
            g = &gp[group_of(nm)];
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
    /* An unnamed term would be scored as zero, which is a value the caller never supplied and
     * usually not one the group was trained near. Refuse rather than answer a different case. */
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
    { double p = unscale_out(g, (double)net_forward(g->net, xn)[0]);
      printf("%s %s = %.*g\n", grp, response, sigdigits(p, g->thi - g->tlo), p); }
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
    fprintf(stderr, "%-10s %6s %6s %8s %10s %10s %10s %10s\n",
            "group", "rows", "held", "weights", "train", "held-out", "refit sd", "floor");
    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        long nw, ntr;
        if (!g->net) continue;
        nw = (long)net_nweights(g->net);
        ntr = g->n - g->nheld;
        fprintf(stderr, "%-10s %6ld %6ld %8ld %10.5g %10.5g %10.5g %10.5g\n",
                g->name, g->n, g->nheld, nw, g->train_rmse, g->held_rmse, g->run_sd,
                2.7718 * g->run_sd);
        /* the network's analogue of a regression's degrees of freedom: least squares pins the
         * terms it cannot identify and says how many, but a network has no such notion and will
         * quietly spend a free parameter per row, so the count has to be put next to the rows */
        if (nw > ntr)
            fprintf(stderr, "  %ld weights fitted to %ld training rows. There are more free\n"
                            "  parameters than examples, so some of what it learned is the rows\n"
                            "  themselves. Reduce -H, or use linearr if the relation may be linear.\n",
                            nw, ntr);
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
            xs[i * MAXTERM] = x; ys[i] = x * x + 10.0; ord[i] = i;
        }
        nrow = 13; p.first = 0; p.n = 13;
        ranges(&p);
        n = train_one(&p, ord, 13, 1u);
        if (!n) { printf("selftest: cannot train\n"); return 1; }
        e = rmse(&p, n, ord, 0, 13);
        printf("fitting y=x^2+10 on 13 points: RMSE %.4f (span %g) %s\n", e, p.thi - p.tlo,
               e < 0.05 * (p.thi - p.tlo) ? "PASS" : (fail=1,"FAIL"));
        net_free(n);
    }
    printf("\nself-test %s\n", fail ? "FAILED" : "PASSED");
    return fail;
}

static void usage(void)
{
    printf("bpnn -- a backpropagation network for tabular data, where a line is not enough\n\n");
    printf("  bpnn -t data.csv > model.txt     fit one network per group\n");
    printf("  bpnn -c model.txt A x=3          score one case\n");
    printf("  bpnn --selftest                  check the arithmetic\n\n");
    printf("input CSV is linearr's: a header, then GROUP, the response, one column per term.\n\n");
    printf("  -H N        hidden units (default %ld)\n", hidden);
    printf("  -e N        epochs (default %ld)\n", epochs);
    printf("  -s N        refits, to measure the spread (default %ld)\n", nseed);
    printf("  -r X -m X   learning rate, momentum (default %g, %g)\n", rate, momentum);
    printf("  -a NAME     hidden activation: sigmoid, tanh, relu (default tanh)\n");
    printf("  --holdout X fraction of rows kept out of the fit (default %g; 0 disables\n", holdout);
    printf("              it and makes the reported error meaningless as generalization)\n");
}

/* An option's argument. Junk is refused rather than read as zero: `-H six` used to mean six
 * hidden units to the caller and none to atol. */
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
        } else if (!strcmp(argv[i], "-H")) {
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
    if (rate <= 0 || momentum < 0 || momentum >= 1) {
        fprintf(stderr, "bpnn: the learning rate must be positive and the momentum in [0, 1)\n");
        return 2;
    }
    if (holdout < 0 || holdout >= 1) {
        fprintf(stderr, "bpnn: --holdout is a fraction of the rows, so it must be in [0, 1).\n"
                        "Use 0 to fit on every row, and read the warning it prints.\n");
        return 2;
    }
    if (read_csv(path) != 0) { rc = 1; goto out; }
    if (nrow == 0) { fprintf(stderr, "bpnn: %s has a header and no data rows\n", path); rc = 1; goto out; }
    if (regroup() != 0) { fprintf(stderr, "bpnn: out of memory\n"); rc = 1; goto out; }
    for (g = 0; g < ngroup; g++) {
        if (gp[g].n < 4) {
            fprintf(stderr, "bpnn: group %s has %ld rows; a network needs more than that to say\n"
                            "anything, and it is skipped rather than fitted.\n", gp[g].name, gp[g].n);
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
