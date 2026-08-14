/* model.c -- the model file: what is written out, and what is accepted back.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"
#include "act.h"

#ifndef BPNN_VERSION
#define BPNN_VERSION "0.0.0-dev"
#endif

/* The model file format this build writes and the highest it will read. */
#define MODEL_VERSION 1

void write_model(void)
{
    long i, j;
    size_t l, k;
    /* Provenance, as comments so the reader skips them and an oracle diff can too. Which model
     * produced a number is part of the number, and the file said nothing about where it came
     * from. Deliberately no timestamp: two fits of one file stay byte-identical. */
    printf("# bpnn model: one network per group, fitted by ./bpnn -t\n");
    printf("# bpnn %s, from %s: %ld rows, %ld group%s, %ld term%s\n",
           BPNN_VERSION, strcmp(inpath, "-") ? inpath : "standard input", nrow, ngroup,
           ngroup == 1 ? "" : "s", nterm, nterm == 1 ? "" : "s");
    printf("# Read the diag line before using it. train and held are RMSE in %s;\n", response);
    printf("# sd is the spread over refits and floor=2.77*sd is the smallest difference\n");
    printf("# between two configurations this pipeline can resolve at all. shipped is the error\n");
    printf("# of the model in this file and best is the best refit's; doc/DIAGNOSTICS.md says\n");
    printf("# which of them to quote.\n");
    printf("BPNN %d\n", MODEL_VERSION);
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
        /* held is the mean over refits, which is what the report shows and what nothing
         * selected on. shipped is this model's own held-out error, which did the selecting. */
        printf("diag train=%.6g held=%.6g shipped=%.6g sd=%.6g floor=%.6g best=%.6g expl=%.4f\n",
               g->train_rmse, g->held_rmse, g->shipped_held, g->run_sd, 2.7718 * g->run_sd,
               g->best_held, explained(g));
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

int read_model(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[LINELEN];
    Group *g = NULL;
    long ri = 0, lineno = 0, nfilled = 0;
    int seen_version = 0;
    size_t l = 1, wi = 0, bi = 0;
    int rc = -1;

    if (!f) { fprintf(stderr, "bpnn: cannot open %s\n", path); return -1; }
    while (fgets(line, sizeof line, f)) {
        lineno++;
        if (line[0] == '#') continue;
        if (!strncmp(line, "BPNN ", 5)) {
            /* Written since the first version and never read, so this build would happily score
             * a file from a later one, ignoring every line it did not recognise. A format is not
             * versioned until the version is checked. */
            char *end;
            long v = strtol(line + 5, &end, 10);
            if (end == line + 5 || v < 1) goto bad;
            if (v > MODEL_VERSION) {
                fprintf(stderr, "%s:%ld: this model is version %ld and this build reads up to %d."
                                " Upgrade bpnn.\n", path, lineno, v, MODEL_VERSION);
                fclose(f);
                return -1;
            }
            seen_version = 1;
            continue;
        }
        if (!strncmp(line, "response ", 9)) {
            if (sscanf(line + 9, "%63s", response) != 1) goto bad;
        } else if (!strncmp(line, "terms ", 6)) {
            char *p = line + 6, *end; long j;
            nterm = strtol(p, &end, 10);
            if (end == p || nterm < 1 || nterm > MAXTERM) goto bad;
            p = end;
            for (j = 0; j < nterm; j++) {
                while (*p == ' ') p++;
                if (sscanf(p, "%63s", term[j]) != 1) goto bad;
                while (*p && *p != ' ') p++;
            }
        } else if (!strncmp(line, "GROUP ", 6)) {
            char nm[NAMELEN];
            long gi;
            if (nterm < 1) goto bad;                  /* GROUP before terms */
            if (sscanf(line + 6, "%63s", nm) != 1) goto bad;
            gi = group_of(nm);
            if (gi < 0) goto bad;                     /* more groups than the build holds */
            g = &gp[gi];
            ri = 0; l = 1; wi = bi = 0;
        } else if (g && !strncmp(line, "diag ", 5)) {
            /* Scanned by key, not by position: a positional format silently stopped at the
             * first field this writer added and left every later one at zero. */
            char *p;
            if ((p = strstr(line, "train=")))   g->train_rmse   = atof(p + 6);
            if ((p = strstr(line, "held=")))    g->held_rmse    = atof(p + 5);
            if ((p = strstr(line, "shipped="))) g->shipped_held = atof(p + 8);
            if ((p = strstr(line, "sd=")))      g->run_sd       = atof(p + 3);
            if ((p = strstr(line, "best=")))    g->best_held    = atof(p + 5);
        } else if (g && !strncmp(line, "target ", 7)) {
            if (sscanf(line + 7, "%lf %lf", &g->tlo, &g->thi) != 2) goto bad;
            if (!isfinite(g->tlo) || !isfinite(g->thi) || g->thi <= g->tlo) goto bad;
        } else if (g && !strncmp(line, "range ", 6)) {
            if (ri >= nterm) goto bad;                /* more ranges than terms */
            if (sscanf(line + 6, "%lf %lf", &g->lo[ri], &g->hi[ri]) != 2) goto bad;
            if (!isfinite(g->lo[ri]) || !isfinite(g->hi[ri]) || g->hi[ri] <= g->lo[ri]) goto bad;
            ri++;
        } else if (g && !strncmp(line, "net ", 4)) {
            size_t dims[SMB_MAX_LAYERS], i;
            long nl, act, d;
            char *p = line + 4, *end;
            if (g->net) goto bad;                     /* two net lines for one group */
            nl = strtol(p, &end, 10); if (end == p) goto bad; p = end;
            act = strtol(p, &end, 10); if (end == p) goto bad; p = end;
            if (nl < 2 || nl > SMB_MAX_LAYERS) goto bad;
            if (act < 0 || act > 2) goto bad;
            for (i = 0; i < (size_t)nl; i++) {
                d = strtol(p, &end, 10);
                if (end == p) goto bad;               /* fewer widths than the layer count */
                p = end;
                /* A width is the size of a buffer the forward pass writes into. Unchecked, a
                 * model file chooses that size: 4000 inputs overran the caller's stack. */
                if (d < 1 || d > MAXTERM * 16) goto bad;
                dims[i] = (size_t)d;
            }
            if (dims[0] != (size_t)nterm || dims[nl - 1] != 1) goto bad;
            g->net = net_new(dims, (size_t)nl);
            if (!g->net) goto bad;
            g->net->activation = (int)act;
            l = 1; wi = bi = 0;
        } else if (g && g->net && line[0] == 'w' && line[1] == ' ') {
            char *end;
            double v = strtod(line + 2, &end);
            if (end == line + 2 || !isfinite(v)) goto bad;
            while (l < g->net->nlayers && wi >= net_layer_wsize(g->net, l)) { l++; wi = bi = 0; }
            if (l >= g->net->nlayers) goto bad;       /* more weights than the shape holds */
            g->net->w[l][wi++] = (smb_real)v;
        } else if (g && g->net && line[0] == 'b' && line[1] == ' ') {
            char *end;
            double v = strtod(line + 2, &end);
            if (end == line + 2 || !isfinite(v)) goto bad;
            if (l >= g->net->nlayers || bi >= net_layer_bsize(g->net, l)) goto bad;
            g->net->b[l][bi++] = (smb_real)v;
            if (bi >= net_layer_bsize(g->net, l)) { l++; wi = bi = 0; }
        } else if (g && g->net && !strncmp(line, "END", 3)) {
            /* Every weight and bias must have arrived. net_new mallocs the weight arrays, so a
             * short model would otherwise be scored from whatever was in that memory, giving a
             * different answer per run. */
            if (l != g->net->nlayers) goto bad;
            if (ri != nterm) goto bad;
            nfilled++;
            g = NULL;
        }
    }
    if (ferror(f)) goto bad;
    if (nfilled < 1 || !seen_version) goto bad;
    rc = 0;
bad:
    if (rc != 0)
        fprintf(stderr, "%s:%ld: this is not a usable bpnn model\n", path, lineno);
    fclose(f);
    return rc;
}

