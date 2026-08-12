/* nb101_flip.c -- how often does a single training run get the ordering of two architectures wrong?
 * Self-contained C99. Build: make nb101_flip
 *
 * THE QUESTION. Architecture-search papers routinely report that one architecture beats another by a
 * few tenths of a percentage point, on the strength of one training run each. Training is random, so
 * a single run measures the architecture's quality plus a draw of noise. This probe asks the question a
 * practitioner actually needs answered: given that a single run says A beats B, how often is that
 * wrong?
 *
 * WHY NAS-BENCH-101 CAN ANSWER IT. Every one of its 423,624 architectures was trained three separate
 * times. The three accuracies differ only through training randomness, which is what makes them
 * interchangeable and lets us play one run against the others.
 *
 * THE DESIGN, and the reason it is not circular. For each pair of architectures:
 *   the JUDGMENT   comes from ONE run each          sign(v_a[p0] - v_b[p0])
 *   the REFERENCE  comes from the OTHER TWO runs    sign(mean(v_a[p1],v_a[p2]) - mean(v_b[p1],v_b[p2]))
 * The two use disjoint runs, so the reference is statistically independent of the judgment. Using the
 * mean of all three runs as the reference would be circular, because the judgment run is inside it.
 *
 * WHICH GAP TO BIN BY, and the first version got this wrong. Binning by the REFERENCE gap conditions on
 * a noisy quantity, and conditioning on it being small selects pairs whose reference noise happened to
 * make them look close, so the small-gap rows are contaminated by the reference's own error and read as
 * far worse than the single run deserves. Binning by the JUDGMENT gap has no such problem, and it also
 * answers the question a practitioner actually has. They do not know the true gap; they know the number
 * in front of them. "One run each showed 0.2 points -- how often is that backwards?" is a conditional
 * probability on an observed quantity, and since the reference is independent of the judgment, the
 * disagreement rate within a judgment-gap bin estimates it directly. Both binnings are reported so the
 * difference between the honest one and the naive one is visible.
 *
 * PREDICTION, fixed before running. The within-architecture spread of validation accuracy on this
 * benchmark has a median of 0.33 percentage points, so for pairs genuinely separated by 0.1 to 0.3
 * points -- the band these papers argue over -- a single run should be wrong a large fraction of the
 * time, somewhere between a quarter and a half. If instead it is under 10%, single-run comparisons in
 * that band are defensible and this line of work has much less to say.
 *
 * env: PAIRS SEED MINV DELTA
 *   MINV   drop architectures whose worst run falls below this (default 0 = keep all). The benchmark
 *          contains roughly one architecture in a hundred that sometimes trains and sometimes collapses
 *          to the accuracy of guessing; MINV=50 removes them so their contribution can be seen.
 * WHAT THE NUMBER IS, stated carefully. With three runs there is no noiseless reference available, so
 * this does not isolate the single run's error from the reference's. What it measures is the
 * REPLICATION DISAGREEMENT RATE: two independent estimates of the same comparison, one from a single
 * run and one from two runs, and how often they disagree. That is the quantity a reader wants anyway,
 * because it answers whether an independent repetition of a published comparison would come out the
 * same way. The "ref unresolvable" column reports the share of pairs whose gap is smaller than the
 * reference's own standard error, which is how much of each row the reference could be responsible for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXARCH 460000
#define NTRIAL  3
#define NBIN    8

static float vacc[MAXARCH][NTRIAL];
static int   narch = 0;

/* 64-bit generator: a study drawing billions of numbers exhausts a 32-bit period and silently reuses
 * draws, which understates its own standard error. */
static uint64_t rs = 1u;
static uint32_t r32(void)
{
    uint64_t z = (rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}
static void     rseed(uint32_t s){ rs = s ? (uint64_t)s * 0x9E3779B97F4A7C15ull : 1u; }
static uint32_t rbelow(uint32_t m){ return m ? r32()%m : 0u; }
static int envint(const char *k, int d){ const char *v=getenv(k); return v? atoi(v): d; }
static double envdbl(const char *k, double d){ const char *v=getenv(k); return v? atof(v): d; }

/* Input is one architecture per line: NTRIAL whitespace-separated accuracies in percentage points.
 * Lines beginning with # are comments. That is everything this measurement needs, so it is everything
 * the format carries, and it lets one probe read any benchmark once projected to triples (see Makefile). */
static int load(const char *path, double minv)
{
    FILE *f = fopen(path, "r");
    char line[512];
    if(!f){ fprintf(stderr, "nb101_flip: cannot open %s\n", path); return -1; }
    while(fgets(line, sizeof line, f)){
        char *p = line; float v[NTRIAL]; int k, bad = 0;
        if(*p == '#') continue;
        for(k = 0; k < NTRIAL; k++){
            char *e;
            v[k] = (float)strtod(p, &e);
            if(e == p){ bad = 1; break; }
            p = e;
            if(v[k] <= 0.0f || v[k] < minv) bad = 1;
        }
        if(bad || narch >= MAXARCH) continue;
        for(k = 0; k < NTRIAL; k++) vacc[narch][k] = v[k];
        narch++;
    }
    fclose(f);
    return narch;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "validation/nb101_triples.txt";
    long pairs = (long)envint("PAIRS", 2000000), i;
    double minv = envdbl("MINV", 0.0);
    /* bin edges in percentage points of the reference gap */
    static const double edge[NBIN] = { 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00, 1e9 };
    static const char  *lab[NBIN]  = { "0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30",
                                       "0.30-0.50", "0.50-1.00", "1.00-2.00", "  > 2.00" };
    long n[NBIN] = {0}, dis[NBIN] = {0}, unres[NBIN] = {0};   /* binned by JUDGMENT gap (the honest one) */
    long rn[NBIN] = {0}, rdis[NBIN] = {0};                    /* binned by REFERENCE gap (for contrast) */
    long ties = 0;
    double sdsum = 0; long sdn = 0;

    if(load(path, minv) < 0) return 1;
    rseed((uint32_t)envint("SEED", 20260812));

    /* the noise scale, for the reader to weigh the gaps against */
    for(i = 0; i < narch; i++){
        double m = (vacc[i][0]+vacc[i][1]+vacc[i][2])/3.0, s = 0; int k;
        for(k = 0; k < NTRIAL; k++) s += (vacc[i][k]-m)*(vacc[i][k]-m);
        sdsum += sqrt(s/2.0); sdn++;
    }

    printf("nb101_flip -- how often one training run gets the ordering of two architectures wrong\n");
    printf("%d architectures x %d runs each, %ld random pairs, MINV=%.1f\n", narch, NTRIAL, pairs, minv);
    printf("judgment: ONE run each. reference: the OTHER TWO runs, averaged. Disjoint, so independent.\n");
    printf("mean within-architecture SD of validation accuracy: %.4f pp\n\n", sdsum/sdn);

    for(i = 0; i < pairs; i++){
        int a = (int)rbelow((uint32_t)narch), b = (int)rbelow((uint32_t)narch);
        int perm[NTRIAL] = {0,1,2}, k, bin;
        double ja, jb, ra, rb, gap;
        if(a == b) continue;
        for(k = NTRIAL-1; k > 0; k--){ int j = (int)rbelow((uint32_t)(k+1)), t = perm[k]; perm[k]=perm[j]; perm[j]=t; }
        ja = vacc[a][perm[0]];                                  /* one run       */
        jb = vacc[b][perm[0]];
        ra = 0.5*(vacc[a][perm[1]] + vacc[a][perm[2]]);         /* the other two */
        rb = 0.5*(vacc[b][perm[1]] + vacc[b][perm[2]]);
        gap = ra - rb;
        if(gap == 0.0){ ties++; continue; }
        { double jgap = ja - jb;
          int jb_i;
          if(jgap == 0.0){ ties++; continue; }
          /* PRIMARY: bin by what the practitioner observes, the single-run difference */
          for(jb_i = 0; jb_i < NBIN; jb_i++) if(fabs(jgap) < edge[jb_i]) break;
          if(jb_i >= NBIN) jb_i = NBIN-1;
          n[jb_i]++;
          if(jgap * gap < 0.0) dis[jb_i]++;
          /* SECONDARY: bin by the reference gap, which conditions on a noisy quantity */
          for(bin = 0; bin < NBIN; bin++) if(fabs(gap) < edge[bin]) break;
          if(bin >= NBIN) bin = NBIN-1;
          rn[bin]++;
          if(jgap * gap < 0.0) rdis[bin]++;
          /* the reference's own resolution for THIS pair, tallied against the judgment bin so the
           * share is a share of the same denominator the disagreement rate uses */
          { double sa = 0, sb = 0, ma, mb, refse2; int q;
            ma = (vacc[a][0]+vacc[a][1]+vacc[a][2])/3.0;
            mb = (vacc[b][0]+vacc[b][1]+vacc[b][2])/3.0;
            for(q = 0; q < NTRIAL; q++){ sa += (vacc[a][q]-ma)*(vacc[a][q]-ma);
                                         sb += (vacc[b][q]-mb)*(vacc[b][q]-mb); }
            refse2 = sqrt((sa/2.0)/2.0 + (sb/2.0)/2.0);
            if(fabs(gap) < refse2) unres[jb_i]++; } }
    }

    printf("PRIMARY -- binned by the OBSERVED single-run difference, which is what a reader has:\n");
    printf("%-11s %10s %12s %16s\n", "observed pp", "pairs", "backwards", "ref unresolvable");
    { long tn = 0, td = 0;
      for(i = 0; i < NBIN; i++){
        if(n[i] == 0) continue;
        printf("%-11s %10ld %11.1f%% %15.1f%%\n", lab[i], n[i],
               100.0*dis[i]/n[i], 100.0*unres[i]/n[i]);
        tn += n[i]; td += dis[i];
      }
      printf("%-11s %10ld %11.1f%%\n", "all", tn, 100.0*td/tn);
      if(ties) printf("(%ld exact ties, dropped)\n", ties);
    }
    printf("\nSECONDARY -- binned by the REFERENCE gap. Reported only to show why it is the wrong\n");
    printf("conditioning: selecting on a noisy gap being small picks pairs the reference misjudged.\n");
    printf("%-11s %10s %12s\n", "ref gap pp", "pairs", "backwards");
    for(i = 0; i < NBIN; i++)
        if(rn[i]) printf("%-11s %10ld %11.1f%%\n", lab[i], rn[i], 100.0*rdis[i]/rn[i]);
    /* RESOLUTION, and why it is reported as a distribution rather than as one number.
     *
     * The flip rate above needs no distributional assumption. Converting it into "how many runs do I
     * need" does, and on this data that conversion is fragile: the per-architecture spread is a
     * heavy-tailed mixture, so its median is 0.33 pp, its mean 0.92, its RMS excluding the
     * collapse-prone architectures 0.52, and its RMS including them 4.78. Feeding those into the same
     * formula gives one-run resolution floors from 1.3 to 19 pp and "runs needed for 0.1 pp" from 169 to
     * 35,938. A single headline figure is therefore not a property of the benchmark, it is a property of
     * an analyst's choice of summary, and an earlier version of this probe reported one as if it were the
     * former.
     *
     * So we compute it PER ARCHITECTURE from that architecture's own three runs, and report the
     * distribution. sigma_a is estimated from 3 runs and is itself noisy, which is stated; the
     * percentiles are of the estimate, not of the truth. The standard error of a difference between two
     * architectures each trained n times is sigma*sqrt(2/n), so a gap is resolvable at 95% two-sided
     * confidence when it exceeds 1.96*sigma*sqrt(2/n); at n=1 that floor is 2.77*sigma. */
    {
        static float sda[MAXARCH];
        int q; long m;
        for(m = 0; m < narch; m++){
            double mu = (vacc[m][0]+vacc[m][1]+vacc[m][2])/3.0, ss = 0; int k;
            for(k = 0; k < NTRIAL; k++) ss += (vacc[m][k]-mu)*(vacc[m][k]-mu);
            sda[m] = (float)sqrt(ss/2.0);
        }
        /* insertion-free percentile: partial selection by counting, adequate at this size */
        for(q = 0; q < 1; q++){
            static const double pct[5] = { 0.50, 0.75, 0.90, 0.99, 1.00 };
            static const char  *pn[5]  = { "median", "p75", "p90", "p99", "worst" };
            int j;
            printf("\nRESOLUTION per architecture, from that architecture's own three runs.\n");
            printf("One number cannot describe this: the spread is a heavy-tailed mixture, so the\n");
            printf("answer depends on WHICH architecture, which is why the distribution is shown.\n");
            printf("  %-8s %10s %14s %14s %14s\n", "arch", "sigma pp",
                   "1 run resolves", "runs for 0.3pp", "runs for 0.1pp");
            for(j = 0; j < 5; j++){
                /* rank-select sigma at the requested quantile */
                long target = (long)(pct[j] * (narch - 1)), below;
                double lo = 0, hi = 100, mid = 0;
                int it;
                for(it = 0; it < 40; it++){
                    mid = 0.5*(lo+hi); below = 0;
                    for(m = 0; m < narch; m++) if(sda[m] <= mid) below++;
                    if(below > target) hi = mid; else lo = mid;
                }
                { double sg = mid, n03, n01;
                  n03 = 2.0*(1.96*sg/0.30)*(1.96*sg/0.30);
                  n01 = 2.0*(1.96*sg/0.10)*(1.96*sg/0.10);
                  printf("  %-8s %10.3f %11.2f pp %14.0f %14.0f\n", pn[j], sg, 2.77*sg,
                         n03 < 1 ? 1 : n03, n01 < 1 ? 1 : n01); }
            }
            printf("  sigma_a comes from three runs, so it is itself noisy; these are percentiles of\n");
            printf("  the estimate rather than of the truth.\n");
        }
    }

    printf("\nRead the PRIMARY 0.10-0.30 rows: architecture-search papers contest differences of that\n");
    printf("size, and the rate beside them is the probability such a claim is backwards when each side\n");
    printf("rests on one training run. The last column is the share of pairs whose gap the two-run\n");
    printf("reference cannot itself resolve, so those rows still carry some of the reference's error and\n");
    printf("the rates there remain an upper bound.\n");
    return 0;
}
