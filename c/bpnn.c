/* bpnn.c -- a backpropagation network for tabular data, in linearr's shape.
 *
 *     ./bpnn -t data.csv > model.txt     fit one network per group
 *     ./bpnn -c model.txt A x=3          score one case
 *     cat cases.csv | ./bpnn -c model.txt   score a stream of them
 *     ./bpnn --selftest                  check the arithmetic
 *
 * Input is linearr's: a header, then the group, the response, one column per term.
 *
 *     group,value,x
 *     A,46,-6
 *
 * The model written out is the median of -s refits, not the best of them; picking the best by
 * held-out error makes that error optimistic by however much was selected for.
 *
 * The report carries three things a regression's does not: the error on rows the fit never saw,
 * the spread of that error over refits, and a warning when a case falls outside the range each
 * input was fitted over. README.md explains all three and what they cost.
 *
 * smb_real is float, so about seven digits; nothing here is good past six.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>   /* the row cache is keyed on the input's size and mtime */

#include "tab.h"
#include "act.h"
#include "rng.h"
#include <ctype.h>
#include "train.h"

#ifndef BPNN_VERSION
#define BPNN_VERSION "0.0.0-dev"
#endif
#include "net.h"
#include "train.h"
#include "act.h"
#include "rng.h"
#include <ctype.h>

/* Below this a group cannot be split into a fit, a stopping check and a reported sample of a
 * size worth printing: at 4 rows the reported error was one row and the floor beside it was ten
 * times the error it bounded. */

/* ------------------------------------------------------------------- score  */

/* One case: prediction on stdout, caveats on stderr, so the scorer works in a pipe. WHERE names
 * the case in the warnings. */
static void score_one(Group *g, const double *raw, const int *given, const char *where)
{
    smb_real xn[MAXTERM];
    double a, p;
    long i;
    int out = 0;

    scale_in(g, raw, xn);
    a = (double)net_forward(g->net, xn)[0];
    p = unscale_out(g, a);
    printf("%s,%.*g\n", g->name, sigdigits(p, g->thi - g->tlo), p);

    if (a < 0.02 || a > 0.98)
        fprintf(stderr, "%s: saturated: %.6g is the most extreme %s this model can return "
                        "(fitted over [%.6g, %.6g])\n", where, p, response, g->tlo, g->thi);
    for (i = 0; i < nterm; i++) {
        double span = g->hi[i] - g->lo[i];
        if (!given[i]) continue;
        if (raw[i] < g->lo[i] || raw[i] > g->hi[i]) {
            double over = raw[i] < g->lo[i] ? (g->lo[i] - raw[i]) : (raw[i] - g->hi[i]);
            fprintf(stderr, "%s: %s=%g is outside the fitted range [%g, %g], by %.3g of it\n",
                    where, term[i], raw[i], g->lo[i], g->hi[i], span > 0 ? over / span : 0.0);
            out++;
        }
    }
    if (out)
        fprintf(stderr, "%s: a network does not extrapolate; past its range the units saturate\n",
                where);
}

/* Cases from stdin, one per line: group, then one value per term in the model's order. */
static int score_stream(void)
{
    char line[LINELEN], *fld[MAXTERM + 8], where[64];
    double raw[MAXTERM];
    int given[MAXTERM];
    long lineno = 0, n = 0, i;
    int nf;

    for (i = 0; i < nterm; i++) given[i] = 1;
    fprintf(stderr, "scoring against %s: %ld group%s, %ld term%s. Errors and warnings below are\n"
                    "numbered by the input line they came from.\n",
            response, ngroup, ngroup == 1 ? "" : "s", nterm, nterm == 1 ? "" : "s");
    while (fgets(line, sizeof line, stdin)) {
        long gi = -1;
        lineno++;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        nf = split_csv(line, fld, MAXTERM + 8);
        if (nf != (int)nterm + 1) {
            fprintf(stderr, "-:%ld: %d field%s; this model wants the group and %ld term%s\n",
                    lineno, nf, nf == 1 ? "" : "s", nterm, nterm == 1 ? "" : "s");
            return 1;
        }
        /* A header must name this model's terms, in this model's order. It used to be skipped
         * whenever its second field matched the first term, and every row after it was then read
         * by position: 23 columns permuted scored 2.73643 where the truth was 13.4359, exit 0. */
        if (n == 0 && lineno == 1 && !strcmp(fld[1], term[0])) {
            for (i = 0; i < nterm; i++)
                if (strcmp(fld[i + 1], term[i])) {
                    fprintf(stderr, "-:%ld:%ld: this header says '%s' where the model's term %ld "
                                    "is '%s'.\nThe columns must be the model's terms in the "
                                    "model's order.\n",
                            lineno, i + 2, fld[i + 1], i + 1, term[i]);
                    return 1;
                }
            continue;
        }
        for (i = 0; i < ngroup; i++) if (!strcmp(gp[i].name, fld[0])) gi = i;
        if (gi < 0 || !gp[gi].net) {
            fprintf(stderr, "-:%ld:1: group '%s' is not in this model\n", lineno, fld[0]);
            return 1;
        }
        snprintf(where, sizeof where, "-:%ld", lineno);
        for (i = 0; i < nterm; i++) {
            char what[NAMELEN + 32];
            snprintf(what, sizeof what, "the term '%s'", term[i]);
            inpath = "-";
            if (number(fld[i + 1], what, lineno, i + 2, &raw[i]) != 0) return 1;
        }
        score_one(&gp[gi], raw, given, where);
        n++;
    }
    if (ferror(stdin)) { fprintf(stderr, "bpnn: error reading the cases\n"); return 1; }
    return 0;
}

static int score(const char *path, int argc, char **argv, int from)
{
    double raw[MAXTERM];
    int given[MAXTERM];
    const char *grp = NULL;
    long i;
    Group *g;
    int rc = 2;

    if (read_model(path) != 0) {
        fprintf(stderr, "bpnn: %s is not a bpnn model\n", path);
        return 1;
    }
    memset(given, 0, sizeof given);
    for (i = 0; i < nterm; i++) raw[i] = 0.0;

    /* No case on the command line: read them from stdin, one per line. This is the form a
     * pipeline uses, and the only one that scores more than a single row per process. */
    if (from >= argc) {
        rc = score_stream();
        goto out;
    }

    for (i = from; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) {
            if (grp) {
                fprintf(stderr, "bpnn: two group names given, '%s' and '%s'\n", grp, argv[i]);
                goto out;
            }
            grp = argv[i];
            continue;
        }
        *eq = '\0';
        {
            long j, hit = -1;
            char *end;
            double v;
            for (j = 0; j < nterm; j++) if (!strcmp(term[j], argv[i])) hit = j;
            if (hit < 0) {
                fprintf(stderr, "bpnn: '%s' is not a term in this model\n", argv[i]);
                goto out;
            }
            v = strtod(eq + 1, &end);
            if (end == eq + 1 || *end != '\0' || !isfinite(v)) {
                fprintf(stderr, "bpnn: %s=%s is not a finite number\n", argv[i], eq + 1);
                goto out;
            }
            raw[hit] = v;
            given[hit] = 1;
        }
    }
    if (!grp) {
        fprintf(stderr, "bpnn: name the group to score, e.g. ./bpnn -c %s A x=3\n", path);
        goto out;
    }
    {
        long j, hit = -1, missing = 0;
        for (j = 0; j < ngroup; j++) if (!strcmp(gp[j].name, grp)) hit = j;
        if (hit < 0) {
            fprintf(stderr, "bpnn: group '%s' is not in this model\n", grp);
            goto out;
        }
        g = &gp[hit];
        if (!g->net) {
            fprintf(stderr, "bpnn: group '%s' has no fitted network\n", grp);
            goto out;
        }
        /* An unnamed term is scored as zero, a value the caller did not supply and usually
         * outside the range the group was trained on. Refuse instead of answering another case. */
        for (j = 0; j < nterm; j++) if (!given[j]) missing++;
        if (missing) {
            fprintf(stderr, "bpnn: this model has %ld term%s and %ld %s not given:",
                    nterm, nterm == 1 ? "" : "s", missing, missing == 1 ? "was" : "were");
            for (j = 0; j < nterm; j++) if (!given[j]) fprintf(stderr, " %s", term[j]);
            fprintf(stderr, "\nA missing term would be scored as zero, a different case.\n");
            goto out;
        }
    }
    fprintf(stderr, "this model's held-out RMSE at fit time %.6g, spread over refits %.6g\n",
            g->shipped_held > 0 ? g->shipped_held : g->held_rmse, g->run_sd);
    score_one(g, raw, given, "the case");
    rc = 0;
out:
    for (i = 0; i < ngroup; i++) if (gp[i].net) net_free(gp[i].net);
    return rc;
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

static void usage(void)
{
    printf("bpnn -- a backpropagation network for tabular data, where a line is not enough\n\n");
    printf("  bpnn -t data.csv > model.txt     fit one network per group\n");
    printf("  bpnn -c model.txt A x=3          score one case\n");
    printf("  bpnn --selftest                  check the arithmetic\n\n");
    printf("input CSV is linearr's: a header, then GROUP, the response, one column per term.\n\n");
    printf("  -y NAME     the column to predict, when it is not column 2\n");
    printf("  -H N        hidden units, --size N also (R's nnet calls it size)\n");
    printf("              (default %ld)\n", hidden);
    printf("  -e N        epochs (default %ld)\n", epochs);
    printf("  -s N        refits, to measure the spread (default %ld)\n", nseed);
    printf("  -r X -m X   learning rate, momentum (default %g, %g). What matters is\n", rate, momentum);
    printf("              r/(1-m), the step a weight takes in the limit: %g here. Past\n",
           rate / (1.0 - momentum));
    printf("              about 5 the fit degrades, and past 10 it collapses\n");
    printf("  -a NAME     hidden activation: sigmoid, tanh, relu (default tanh)\n");
    printf("  --holdout X fraction of rows kept out of the fit (default %g; 0 disables\n", holdout);
    printf("              it and makes the reported error meaningless as generalization)\n");
    printf("  --min-rows N  skip a group with fewer than N rows (default %ld). Below it\n", minrows);
    printf("              the fitted, stopping and reported samples are all too small\n");
    printf("              for the report to describe anything\n");
    printf("  --decay X   weight decay: each step also pulls every weight toward zero\n");
    printf("              by rate*X*w (default %g, no decay)\n", decay);
    printf("  --patience N  stop a fit after N epochs with no improvement on the rows\n");
    printf("              kept back to judge that (default %ld; 0 disables the check\n", patience);
    printf("              and runs every epoch of -e)\n");
    printf("  --stream    fit without holding the rows in memory: two passes over the\n");
    printf("              file, then one per epoch over a cache. Different training\n");
    printf("              order, so different numbers from the default path.\n");
    printf("  --buffer N  rows in the shuffle window under --stream (default %ld)\n", bufrows);
    printf("  --per-refit F  write each refit's held-out error to F, so two runs can be\n");
    printf("              compared as matched pairs: pairstat --paired A F\n");
    printf("  --cache F   keep --stream's scaled rows in F, so a later run of any shape\n");
    printf("              skips both passes over the CSV. Rebuilt when the input's size\n");
    printf("              or modification time changes.\n");
    printf("  --footprint TERMS GROUPS   what a fit of that shape costs in memory\n");
}

/* A numeric option argument. Non-numeric text is an error, not zero. */
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

/* Did any group produce a network? A model file with none is written, and was exiting 0. */
static int minrows_set;

static int fitted_any(void)
{
    long i;
    for (i = 0; i < ngroup; i++) if (gp[i].net) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    int i, mode = 0, first = 0;
    const char *path = NULL;
    long g;
    int rc = 0;

    /* --opt=value, rewritten as --opt value before anything reads it. The joined form is the
     * convention most fingers have, and rejecting it said the option did not exist. */
    {
        static char  buf[64][NAMELEN + 8];
        static char *av[128];
        int n = 0, nb = 0;
        av[n++] = argv[0];
        for (i = 1; i < argc && n < 126; i++) {
            char *eq = (argv[i][0] == '-' && argv[i][1] == '-') ? strchr(argv[i], '=') : NULL;
            size_t len = eq ? (size_t)(eq - argv[i]) : 0;
            if (eq && len > 0 && len < sizeof buf[0] && nb < 64) {
                memcpy(buf[nb], argv[i], len);
                buf[nb][len] = '\0';
                av[n++] = buf[nb++];
                av[n++] = eq + 1;
            } else {
                av[n++] = argv[i];
            }
        }
        av[n] = NULL;
        argv = av;
        argc = n;
    }

    for (i = 1; i < argc; i++) {
        double v;
        const char *s;
        if (!strcmp(argv[i], "--version")) {
            printf("bpnn %s\n", BPNN_VERSION);
            return 0;
        }
        else if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            /* -h beside -t redirected help into the model file and exited 0. Help is an action,
             * not a modifier, and it is the only thing a run does. */
            if (argc != 2) {
                fprintf(stderr, "bpnn: --help takes no other arguments. Did you mean -H %s?\n",
                        i + 1 < argc ? argv[i + 1] : "N");
                return 2;
            }
            usage();
            return 0;
        }
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
        } else if (!strcmp(argv[i], "--min-rows")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            minrows = (long)v;
            minrows_set = 1;
        } else if (!strcmp(argv[i], "--decay")) {
            if (optnum(argc, argv, &i, &v) != 0) return 2;
            decay = v;
        } else if (!strcmp(argv[i], "--stream")) {
            streaming = 1;
        } else if (!strcmp(argv[i], "-y")) {
            if (optstr(argc, argv, &i, &ycol) != 0) return 2;
        } else if (!strcmp(argv[i], "--per-refit")) {
            if (optstr(argc, argv, &i, &refitpath) != 0) return 2;
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
    if (minrows < 1) {
        fprintf(stderr, "bpnn: --min-rows is a row count and must be at least 1\n");
        return 2;
    }
    if (decay < 0) {
        fprintf(stderr, "bpnn: --decay is a penalty on the size of the weights and cannot be\n"
                        "negative; 0, the default, is no decay.\n");
        return 2;
    }
    /* The decay part of a step is -rate*lambda*w. At rate*lambda >= 1 it carries the weight
     * past zero and further each step, which diverges rather than regularises. */
    /* Decay is subtracted per row and sits inside the momentum delta, so a weight's shrink over
     * an epoch of n rows is rate*lambda*n/(1-momentum) -- 13,000 lambda at the defaults on 400
     * rows. Checked again per group, where n is known. */
    if (decay * rate >= 1.0) {
        fprintf(stderr, "bpnn: --decay %g at -r %g shrinks every weight by %g of itself per row.\n"
                        "At 1 or more it overshoots zero and diverges. The useful range is far\n"
                        "smaller: try 1e-5 to 1e-4.\n", decay, rate, decay * rate);
        return 2;
    }
    if (rate > 0 && momentum >= 0 && momentum < 1 && rate / (1.0 - momentum) > 5.0)
        fprintf(stderr, "bpnn: -r %g with -m %g is an effective step of %.3g. Momentum multiplies\n"
                        "the rate by 1/(1-m), so the two are one knob; past about 5 the fit gets\n"
                        "worse and past 10 it collapses.\n",
                rate, momentum, rate / (1.0 - momentum));
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
    reading_line(path);
    refit_open();
    if (nrow == 0) { fprintf(stderr, "bpnn: %s has a header and no data rows\n", path); rc = 1; goto out; }
    if (regroup() != 0) { rc = 1; goto out; }
    for (g = 0; g < ngroup; g++) {
        /* The floor is a function of the model's size, not a constant: 24 rows admits exactly
         * the groups that memorise themselves, and at 24 terms a usable fit needs ten times that.
         * 31 is where the stopping split stops collapsing at the default holdout. */
        {
            long need = (long)((double)((long)nterm * hidden + hidden * 2 + 1) / (1.0 - holdout));
            if (need < minrows) need = minrows;
            if (minrows_set) need = minrows;
            if (gp[g].n < need) {
                fprintf(stderr, "bpnn: group %s has %ld rows. This shape fits %ld parameters and\n"
                                "needs about %ld rows for the samples to mean anything, so the group\n"
                                "is skipped. Pool it, fit it with linearr, or say --min-rows %ld.\n",
                        gp[g].name, gp[g].n, (long)nterm * hidden + hidden * 2 + 1, need, gp[g].n);
                continue;
            }
        }
        if (0) {
            fprintf(stderr, "bpnn: group %s has %ld rows. Under %ld the fitted, stopping and\n"
                            "reported samples are all too small to mean anything, so the group is\n"
                            "skipped. Pool it with another, fit it with linearr, or say\n"
                            "--min-rows %ld and read the numbers knowing what they rest on.\n",
                    gp[g].name, gp[g].n, minrows, gp[g].n);
            continue;
        }
        if (fit_group(&gp[g]) != 0) {
            fprintf(stderr, "bpnn: cannot fit group %s\n", gp[g].name);
            rc = 1; goto out;
        }
    }
    write_model();
    report();
    rc = fitted_any() ? 0 : 1;
    /* A model nobody can score from is not a success, and a short write is not one either:
     * ./bpnn -t data.csv > /dev/full exited 0 with a truncated model. */
    if (fflush(stdout) != 0 || ferror(stdout)) {
        fprintf(stderr, "bpnn: writing the model failed\n");
        rc = 1;
    }
out:
    refit_close();
    for (g = 0; g < ngroup; g++) if (gp[g].net) net_free(gp[g].net);
    free(xs); free(ys); free(rowgrp);
    return rc;
}
