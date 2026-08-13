/* stream.c -- fitting without the row store: a cache of scaled rows, one pass per epoch.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"
#include <sys/stat.h>
#include "train.h"
#include "act.h"
#include "rng.h"

/* Fitting without the row store. Two passes over the CSV -- ranges, then a cache of the scaled
 * rows -- and one pass over that cache per epoch. Memory is the networks plus the shuffle
 * windows; still one pass per epoch, only the row store goes away.
 *
 * Different training order from the default path, so different numbers. Both deterministic.
 *
 * A cache record: group, the row's ordinal in it, the scaled terms, the scaled target. */

#define REC_HEAD  (2 * sizeof(int32_t))

static size_t recsize(void)
{
    return REC_HEAD + (size_t)(nterm + 1) * sizeof(float);
}

/* Row roles from a hash of the row's position, not from the training PRNG: a draw would move
 * the split whenever training consumed a different number of random values. The seed is in the
 * hash, so each refit holds out a different quarter. */
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

/* Shuffle window: push a record in, take one out. A full permutation would cost 8 bytes a row. */
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

/* Drain the window at the end of a pass: one epoch, one visit per row. */
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

/* The cache between runs.
 *
 * Neither setup pass depends on a hyperparameter, so --cache keeps them in a named file and a
 * later run of any shape reuses them. Keyed on the input's size and mtime, with the usual
 * weakness: a rewrite inside the same second to the same length is not detected. */

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
int fit_stream(const char *path)
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
        if (g->n < minrows) {
            fprintf(stderr, "bpnn: group %s has %ld rows. Under %ld the fitted, stopping and\n"
                            "reported samples are all too small to mean anything, so the group is\n"
                            "skipped. Pool it with another, fit it with linearr, or say\n"
                            "--min-rows %ld and read the numbers knowing what they rest on.\n",
                    g->name, g->n, minrows, g->n);
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

