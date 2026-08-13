/* fit.c -- fitting one group from the row store, over several refits.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"
#include "train.h"
#include "act.h"
#include "rng.h"

void net_copy_weights(Net *dst, const Net *src)
{
    size_t l;
    for (l = 1; l < src->nlayers; l++) {
        memcpy(dst->w[l], src->w[l], net_layer_wsize(src, l) * sizeof *src->w[l]);
        memcpy(dst->b[l], src->b[l], net_layer_bsize(src, l) * sizeof *src->b[l]);
    }
}

double rmse(Group *g, Net *net, const long *ord, long a, long b);

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

Net *train_one(Group *g, long *ord, long ntr, long nstop, uint32_t seed, long *epochs_out)
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
double rmse(Group *g, Net *net, const long *ord, long a, long b)
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

int fit_group(Group *g)
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
    refit_group(g, held, ran);
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

