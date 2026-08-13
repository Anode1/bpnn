/* model.c -- the model file: what is written out, and what is accepted back.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"

void write_model(void)
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

int read_model(const char *path)
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

