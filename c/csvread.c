/* csvread.c -- reading linearr's CSV, and refusing what should not become a weight.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"

int split_csv(char *line, char **f, int maxf)
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

int number(const char *s, const char *what, long line, long col, double *out)
{
    char *end;
    double v;

    if (*s == '\0') {
        fprintf(stderr, "%s:%ld:%ld: %s is empty\n", inpath, line, col, what);
        return -1;
    }
    v = strtod(s, &end);
    while (*end == ' ' || *end == '\t') end++;
    if (end == s || *end != '\0') {
        fprintf(stderr, "%s:%ld:%ld: %s is '%s', which is not a number.\n",
                inpath, line, col, what, s);
        return -1;
    }
    if (!isfinite(v)) {
        fprintf(stderr, "%s:%ld:%ld: %s is '%s': not finite\n", inpath, line, col, what, s);
        return -1;
    }
    *out = v;
    return 0;
}

/* A header or group name: non-empty, and short enough to store whole. Truncating a group name
 * merges two groups into one fit. */
static int checkname(const char *s, const char *what, long line, long col)
{
    if (*s == '\0') {
        fprintf(stderr, "%s:%ld:%ld: %s is empty\n", inpath, line, col, what);
        return -1;
    }
    if (strlen(s) >= NAMELEN) {
        fprintf(stderr, "%s:%ld:%ld: %s is longer than %d characters: '%s'\n",
                inpath, line, col, what, NAMELEN - 1, s);
        return -1;
    }
    return 0;
}

/* One parsed row at a time. Both fitting paths read through this, so they cannot come to
 * disagree about what a file said or about which rows are refused. */
int reader_open(Reader *rd, const char *path)
{
    rd->f = strcmp(path, "-") ? fopen(path, "r") : stdin;
    rd->path = path;
    inpath = path;
    rd->lineno = 0;
    rd->header = 0;
    if (!rd->f) { fprintf(stderr, "bpnn: cannot open %s\n", path); return -1; }
    return 0;
}

void reader_close(Reader *rd)
{
    if (rd->f && rd->f != stdin) fclose(rd->f);
    rd->f = NULL;
}

/* Returns 1 with a row in GRP/Y/X, 0 at end of file, -1 refused with the reason printed.
 * The header is consumed on the first non-comment line and names the response and the terms. */
int reader_row(Reader *rd, long *grp, double *y, double *x)
{
    char line[LINELEN], *fld[MAXTERM + 8];
    char what[NAMELEN + 32];
    int nf, i;

    while (fgets(line, sizeof line, rd->f)) {
        rd->lineno++;
        /* fgets splits an over-long line in two, and the tail would then be read as a row of
         * its own. Refuse it rather than fit half a row. */
        if (strchr(line, '\n') == NULL && !feof(rd->f)) {
            fprintf(stderr, "%s:%ld: line is longer than %d characters\n",
                    inpath, rd->lineno, LINELEN - 1);
            return -1;
        }
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        nf = split_csv(line, fld, MAXTERM + 8);
        if (!rd->header) {
            if (nf < 3) {
                fprintf(stderr, "%s:%ld: the header needs a group column, the value being\n"
                                "predicted, and at least one term\n", inpath, rd->lineno);
                return -1;
            }
            nterm = nf - 2;
            if (nterm > MAXTERM) {
                fprintf(stderr, "%s:%ld: %ld terms; this build holds %d\n",
                        inpath, rd->lineno, nterm, MAXTERM);
                return -1;
            }
            if (checkname(fld[1], "the name of the value being predicted", rd->lineno, 2) != 0)
                return -1;
            strncpy(response, fld[1], NAMELEN - 1);
            for (i = 0; i < nterm; i++) {
                int j;
                if (checkname(fld[i + 2], "a term name", rd->lineno, (long)(i + 3)) != 0)
                    return -1;
                for (j = 0; j < i; j++)
                    if (!strcmp(term[j], fld[i + 2])) {
                        fprintf(stderr, "%s:%ld:%ld: the term '%s' is named twice, so a case\n"
                                        "naming it could mean either column\n",
                                inpath, rd->lineno, (long)(i + 3), fld[i + 2]);
                        return -1;
                    }
                strncpy(term[i], fld[i + 2], NAMELEN - 1);
            }
            rd->header = 1;
            continue;
        }
        if (nf != nterm + 2) {
            fprintf(stderr, "%s:%ld: %d field%s; the header declared %ld (the group, %s, and\n"
                            "%ld term%s)\n", inpath, rd->lineno, nf, nf == 1 ? "" : "s",
                    nterm + 2, response, nterm, nterm == 1 ? "" : "s");
            return -1;
        }
        if (checkname(fld[0], "the group", rd->lineno, 1) != 0) return -1;
        *grp = group_of(fld[0]);
        if (*grp < 0) {
            fprintf(stderr, "%s:%ld:1: more than %d groups\n", inpath, rd->lineno, MAXGROUP);
            return -1;
        }
        snprintf(what, sizeof what, "'%s', the value being predicted,", response);
        if (number(fld[1], what, rd->lineno, 2, y) != 0) return -1;
        for (i = 0; i < nterm; i++) {
            snprintf(what, sizeof what, "the term '%s'", term[i]);
            if (number(fld[i + 2], what, rd->lineno, (long)(i + 3), &x[i]) != 0) return -1;
        }
        return 1;
    }
    if (ferror(rd->f)) { fprintf(stderr, "bpnn: error reading %s\n", rd->path); return -1; }
    if (!rd->header) { fprintf(stderr, "bpnn: %s has no header line.\n", rd->path); return -1; }
    return 0;
}

/* Read PATH into the row store: every row resident, which is what the default path fits from. */
int read_csv(const char *path)
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
int regroup(void)
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

