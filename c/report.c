/* report.c -- the fit report, and the memory a shape would cost.
 *
 * Copyright (c) 2001-2026 Vasili Gavrilov. BSD 2-Clause; see LICENSE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tab.h"

/* What was read, before anything is fitted. A CSV carries no statement of which column is the
 * response, so a file written in another order fits the wrong one and exits 0; the commonest
 * mistake there is has to be visible at the moment it is made, on the stream a person is
 * watching. linearr prints the same line for the same reason. */
void reading_line(const char *path)
{
    long j;
    fprintf(stderr, "reading %s: column 1 is the group, '%s' is the value being predicted, and\n"
                    "the other %ld column%s %s the term%s:",
            strcmp(path, "-") ? path : "standard input", response,
            nterm, nterm == 1 ? "" : "s", nterm == 1 ? "is" : "are", nterm == 1 ? "" : "s");
    for (j = 0; j < nterm && j < 8; j++) fprintf(stderr, " %s", term[j]);
    if (nterm > 8) fprintf(stderr, " ... (%ld more)", nterm - 8);
    fprintf(stderr, "\nUse -y NAME if that is the wrong column.\n");
}

/* --per-refit: one line per group per refit, so two runs can be compared as matched pairs
 * rather than as two summaries. The seed line is the pairing stamp: both arms draw their split
 * and their starting weights from the refit index alone, so refit s of one configuration and
 * refit s of another see the same split and the same initial weights. pairstat refuses two
 * files whose stamps differ, because a paired test on unpaired runs claims a precision that is
 * not there. */
static FILE *refitf;

void refit_open(void)
{
    if (!refitpath) return;
    refitf = fopen(refitpath, "w");
    if (!refitf) {
        fprintf(stderr, "bpnn: cannot write %s\n", refitpath);
        return;
    }
    fprintf(refitf, "# bpnn per-refit held-out errors\n");
    fprintf(refitf, "# pairstat --paired A B compares two of these on matched refits.\n");
    /* Everything two runs must share for refit s of each to have seen the same rows in the same
     * roles. The path matters: --stream splits by a hash, the default path by Fisher-Yates, so
     * the two are not paired with each other even at identical options. */
    fprintf(refitf, "PAIRING path=%s refits=%ld holdout=%g patience=%ld terms=%ld groups=%ld "
                    "rows=%ld response=%s\n",
            streaming ? "stream" : "default", nseed, holdout, patience, nterm, ngroup, nrow,
            response);
    fprintf(refitf, "# group refit held train epochs\n");
}

void refit_group(const Group *g, const double *held, const double *train, const long *ran)
{
    long s;
    if (!refitf) return;
    for (s = 0; s < nseed; s++)
        fprintf(refitf, "REFIT %s %ld %.9g %.9g %ld\n",
                g->name, s, held[s], train ? train[s] : 0.0, ran ? ran[s] : 0);
}

void refit_close(void)
{
    if (refitf) fclose(refitf);
    refitf = NULL;
}

void report(void)
{
    long i;
    fprintf(stderr, "%-10s %6s %6s %8s %7s %10s %10s %10s %10s %6s\n",
            "group", "rows", "held", "weights", "epochs", "train", "held-out", "refit sd",
            "floor", "expl");
    for (i = 0; i < ngroup; i++) {
        Group *g = &gp[i];
        long nw, ntr;
        if (!g->net) continue;
        /* Every fitted number, biases included: net_nweights counts only the weights, and the
         * advisory below is offered as the degrees-of-freedom check a network does not have. */
        nw = (long)net_nweights(g->net) + (long)g->net->dim[1] + (long)g->net->dim[2];
        /* The stopping rows are not trained on either. */
        ntr = g->n - g->nheld - (patience > 0 ? g->nheld : 0);
        if (ntr < 1) ntr = g->n - g->nheld;
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
        /* No degrees-of-freedom check exists for a network, so print the weights beside the
         * rows and let the reader make it. */
        if (nw > ntr)
            fprintf(stderr, "  %ld weights fitted to %ld training rows. There are more free\n"
                            "  parameters than examples, so some of what it learned is the rows\n"
                            "  themselves. Reduce -H, or use linearr if the relation may be linear.\n",
                            nw, ntr);
        /* Underfitting shows only as both errors being large, which means nothing without the
         * response's own spread to read it against. */
        if (!g->flat && explained(g) < 0.05)
            fprintf(stderr, "  this fit explains %.0f%% of the variance in %s. The group's own mean\n"
                            "  scores %.4g and this model scores %.4g, so it is barely using its\n"
                            "  inputs. Try more hidden units (-H), a longer run (-e), or less\n"
                            "  --decay; or the relation may not be in these columns.\n",
                    100.0 * explained(g), response, g->gysd, g->held_rmse);
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
    fprintf(stderr, "what each column means, and how far to trust it: doc/DIAGNOSTICS.md\n");
    if (!streaming && rowcost() * (double)nrow >= 67108864.0)
        fprintf(stderr, "\nthe row store held %ld rows at %.0f bytes each, %.3g GB, and the sort\n"
                        "briefly needed a second copy. --stream fits the same model without\n"
                        "holding the rows.\n",
                nrow, rowcost(), rowcost() * (double)nrow / 1073741824.0);
}

static void bytes_out(const char *label, double b)
{
    if (b >= 1073741824.0)   printf("  %-22s %.3g GB\n", label, b / 1073741824.0);
    else if (b >= 1048576.0) printf("  %-22s %.3g MB\n", label, b / 1048576.0);
    else if (b >= 1024.0)    printf("  %-22s %.3g kB\n", label, b / 1024.0);
    else                     printf("  %-22s %.0f bytes\n", label, b);
}

/* Memory for a given shape. No figure below takes a row count. */
int footprint(long terms, long groups)
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

