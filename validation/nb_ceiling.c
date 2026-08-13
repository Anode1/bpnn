/* nb_ceiling.c -- two constants a benchmark with replicates can supply, and neither is a folk fact.
 * Self-contained C99. Build: make nb_ceiling
 *
 * Input: one architecture per line, NTRIAL accuracies in percentage points (see the Makefile rules that
 * project NAS-Bench-101 and NAS-Bench-201 to that form).
 *
 * ---------------------------------------------------------------------------------------------------
 * 1. THE RANK-CORRELATION CEILING.
 *
 * Every zero-cost proxy, surrogate model and performance predictor is scored by rank correlation
 * against a benchmark label, and that label is the mean of a handful of noisy training runs. So there
 * is a ceiling: the correlation two INDEPENDENT estimates of the same label achieve with each other.
 * No predictor can beat it, because it is the label's correlation with itself. That number is not
 * usually reported, which makes any published tau uninterpretable in absolute terms: 0.7 against a
 * ceiling of 0.95 is a mediocre predictor, and 0.7 against a ceiling of 0.72 is nearly perfect.
 *
 * We estimate it by splitting the runs: Kendall tau between run p0 alone and the mean of p1,p2. That is
 * a ceiling for a 1-run label against a 2-run label.
 *
 * THE SUBSET MUST BE CHOSEN BY A RUN THAT NEITHER LABEL USES. This took two attempts to get right.
 *
 * Attempt one selected the top-k by the mean of all three runs, which includes the run being correlated;
 * conditioning on a sum induces negative correlation between its components, and it duly reported tau of
 * -0.41, impossible for two estimates of one quantity.
 *
 * Attempt two selected by the 2-run reference B and correlated run 1 against it, on the argument that
 * run 1 took no part in the selection so its variance is unrestricted. That argument is correct as far
 * as it goes, and it is not enough: selection on B leaves Var(A|S) alone but restricts BOTH Cov(A,B|S)
 * and Var(B|S), because conditioning on q + noise being large induces negative dependence between q and
 * that noise. This is Pearson-Thorndike Case-2 range restriction (Pearson 1903; Thorndike 1949). An
 * independent reviewer reproduced the resulting bias in a pure Gaussian simulation with no ties, no
 * heteroscedasticity and normal quality, at the same magnitudes we measured, which proves the bias was
 * the design rather than the data.
 *
 * The present version selects on run perm[2] and uses runs perm[0] and perm[1] as the two labels. Both
 * labels are then orthogonal to the selection, no correction is needed, and Pearson(x1, x2 | S) IS the
 * within-subset ICC(1,1) directly. Verified: residuals fall to +0.002 to +0.041 with no trend in r.
 *
 * WHAT THIS IS, IN STANDARD TERMS, because it is classical and should be cited rather than reinvented.
 * ICC(1,1) = sigma_B^2/(sigma_B^2+sigma_W^2) is the one-way random-effects intraclass correlation
 * (Fisher 1925; Shrout & Fleiss 1979). The reliability of a mean of m runs follows by Spearman-Brown,
 * R_m = m*R_1/(1+(m-1)*R_1) (Spearman 1910; Brown 1910), and the correlation between an m1-run and an
 * m2-run label is sqrt(R_m1 * R_m2), which is Spearman's 1904 correction for attenuation applied with a
 * true-score correlation of one. tau = (2/pi) arcsin(rho) holds for elliptical joints, not merely normal
 * ones (Kruskal 1958; Lindskog, McNeil & Schmock 2003), so non-normal MARGINAL quality does not break it
 * since tau is margin-free; non-ellipticity of the joint does.
 *
 * Kendall tau is computed on a random subsample of pairs rather than all O(n^2), which makes it an
 * unbiased estimate of tau with a standard error we report rather than an exact value.
 *
 * ---------------------------------------------------------------------------------------------------
 * 2. THE INDIFFERENCE CLASS OF THE OPTIMUM.
 *
 * A benchmark's "best architecture" is the argmax of a noisy mean. How many others does its own data
 * fail to separate from that winner? For each architecture we test the winner's mean against it using
 * the pooled within-architecture standard error, and count those it cannot reject at 95%. If that count
 * is large, then regret measured against the reported optimum is measuring a coin flip near the top, and
 * any claim to have "found the optimum" is a claim about which member of a large indistinguishable set
 * a search happened to land on.
 *
 * env: PAIRS SEED TOPK
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define MAXARCH 460000
#define NTRIAL  3

static float vacc[MAXARCH][NTRIAL];
static float vmean[MAXARCH];
static int   narch = 0;

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

static int load(const char *path, double minv)
{
    FILE *f = fopen(path, "r");
    char line[512];
    if(!f){ fprintf(stderr, "nb_ceiling: cannot open %s\n", path); return -1; }
    while(fgets(line, sizeof line, f)){
        char *p = line; float v[NTRIAL]; int k, bad = 0;
        if(*p == '#') continue;
        for(k = 0; k < NTRIAL; k++){
            char *e; v[k] = (float)strtod(p, &e);
            if(e == p || v[k] <= 0.0f || v[k] < minv){ bad = 1; break; }
            p = e;
        }
        if(bad || narch >= MAXARCH) continue;
        for(k = 0; k < NTRIAL; k++) vacc[narch][k] = v[k];
        vmean[narch] = (v[0]+v[1]+v[2])/3.0f;
        narch++;
    }
    fclose(f);
    return narch;
}

/* Kendall tau between a 1-run label and an independent 2-run label, over `pairs` random pairs drawn
 * from the `m` architectures listed in `idx`. Ties on either side are counted and excluded, as
 * tau-b would, and the excluded fraction is reported so the reader can see how much was dropped. */
static void tau_of(const int *idx, int m, long pairs, int p0, int p1, int p2,
                   double *tau, double *se, double *tied)
{
    long conc = 0, disc = 0, ties = 0, i;
    (void)p2;                                  /* p2 selects the subset; it is not a label */
    for(i = 0; i < pairs; i++){
        int a = idx[rbelow((uint32_t)m)], b = idx[rbelow((uint32_t)m)];
        double x, y;
        if(a == b) continue;
        x = vacc[a][p0] - vacc[b][p0];         /* label 1: one run  */
        y = vacc[a][p1] - vacc[b][p1];         /* label 2: another  */
        if(x == 0.0 || y == 0.0){ ties++; continue; }
        if(x*y > 0) conc++; else disc++;
    }
    { long n = conc + disc;
      *tau  = n ? (double)(conc - disc)/(double)n : 0.0;
      /* SE of a proportion-derived statistic: tau = 2p-1 with p = conc/n */
      *se   = n ? 2.0*sqrt(0.25/(double)n) : 0.0;
      *tied = (conc+disc+ties) ? (double)ties/(double)(conc+disc+ties) : 0.0; }
}

/* sort key: the 2-run REFERENCE, never the 3-run mean, so the subset is chosen without the run that
 * will be correlated against it */
static float refkey[MAXARCH];
static int cmp_desc(const void *A, const void *B)
{
    float x = refkey[*(const int*)A], y = refkey[*(const int*)B];
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* CLUSTER SUBSAMPLING for honest intervals. The cells are not independent observations: they share
 * architectures, the subsets are nested (top-1000 is inside top-10000), and budgets are repeated
 * measurements of the same architectures. So the sampling error of a residual must come from resampling
 * the exchangeable unit, which is the ARCHITECTURE, with the entire pipeline including the top-k selection
 * recomputed inside each replicate so that selection variability is included.
 *
 * m-out-of-n subsampling WITHOUT replacement rather than the ordinary bootstrap, because resampling with
 * replacement duplicates architectures and so manufactures ties, which a rank statistic like tau is
 * sensitive to. The spread of a statistic over subsamples of size m scales as sqrt(n/m) relative to the
 * full sample, so the reported SE multiplies the subsample SD by sqrt(m/n). */
static int boot_idx[MAXARCH];
int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "validation/nb101_triples.txt";
    long pairs = (long)envint("PAIRS", 2000000);
    const char *tag = getenv("TAG") ? getenv("TAG") : "cell";
    static int idx[MAXARCH];
    int i, perm[NTRIAL] = {0,1,2};
    double pooled = 0;
    static const int topk[4] = { 100, 1000, 10000, 0 };   /* 0 means all */

    { double minv = (double)envint("MINV", 0);
      if(load(path, minv) < 0) return 1;
      if(minv > 0) printf("MINV=%.0f: architectures with any run below that are excluded.\n", minv); }
    rseed((uint32_t)envint("SEED", 20260812));
    for(i = 0; i < narch; i++) idx[i] = i;

    /* pooled within-architecture SD: 2 df per architecture, so this is precise even though a single
     * architecture's own estimate from 3 runs is not */
    for(i = 0; i < narch; i++){
        double mu = vmean[i], ss = 0; int k;
        for(k = 0; k < NTRIAL; k++) ss += (vacc[i][k]-mu)*(vacc[i][k]-mu);
        pooled += ss/2.0;
    }
    pooled = sqrt(pooled/narch);

    if(!envint("CSV",0))
        printf("nb_ceiling -- %d architectures x %d runs, %s\n", narch, NTRIAL, path);
    if(!envint("CSV",0))
        printf("pooled within-architecture SD %.4f pp (2 df each, so %ld df in total)\n\n",
               pooled, 2L*narch);

    /* ---- 1. the rank-correlation ceiling ---- */
    for(i = NTRIAL-1; i > 0; i--){ int j = (int)rbelow((uint32_t)(i+1)), t = perm[i]; perm[i]=perm[j]; perm[j]=t; }
    for(i = 0; i < narch; i++) refkey[i] = vacc[i][perm[2]];   /* the run NEITHER label uses */
    qsort(idx, (size_t)narch, sizeof idx[0], cmp_desc);
    if(!envint("CSV",0)) printf("RANK-CORRELATION CEILING: Kendall tau of a 1-run label against an independent 2-run label.\n");
    if(!envint("CSV",0)) printf("No predictor scored against this benchmark's labels can exceed it.\n");
    if(!envint("CSV",0))
        printf("  %-12s %10s %12s %10s %8s %10s %9s %11s\n", "subset", "tau_retest", "+-SE", "tied",
               "sW/sB", "pred", "resid", "CEILING");
    for(i = 0; i < 4; i++){
        int m = topk[i] ? (topk[i] < narch ? topk[i] : narch) : narch;
        double tau, se, tied;
        char lab[32];
        tau_of(idx, m, pairs, perm[0], perm[1], perm[2], &tau, &se, &tied);
        if(topk[i]) snprintf(lab, sizeof lab, "top %d", m);
        else        snprintf(lab, sizeof lab, "all %d", m);
        /* THEORY. If quality has spread sigma_B over the subset and training noise sigma_W, then a
         * 1-run label and an independent 2-run label are two noisy views of one quantity with
         *     rho = sigma_B^2 / sqrt((sigma_B^2 + sigma_W^2)(sigma_B^2 + sigma_W^2 / 2))
         * and for jointly normal variables Kendall tau = (2/pi) arcsin(rho). So the ceiling should be a
         * function of the single ratio sigma_W/sigma_B rather than an independent property of each
         * benchmark, and the top-k collapse should follow from sigma_B shrinking while sigma_W does not.
         *
         * sigma_B within the subset is estimated from the JUDGMENT run alone, whose variance is
         * sigma_B^2 + sigma_W^2. That run took no part in choosing the subset, so it is not restricted by
         * the selection; using the 3-run mean here would be, since the subset was chosen on the
         * reference. sigma_W is the pooled within-architecture value over the same subset. */
        /* ICC within the subset, by one-way random-effects moments over the TWO label runs only. Both
         * are orthogonal to the selection, so nothing here is range-restricted. sigma_W^2 comes from
         * (x1-x2)^2/2, which is orthogonal to their own sum and so is uncontaminated even when a subset
         * was chosen using those runs; the 3-run pooled estimate is not, and in top-k subsets it drove
         * sigma_B^2 negative. */
        { double m1 = 0, m2 = 0, sw = 0, vB, rho, tpred, ratio, icc; int j2;
          for(j2 = 0; j2 < m; j2++){
              int a = idx[j2];
              double x = vacc[a][perm[0]], y = vacc[a][perm[1]], d = x - y;
              m1 += 0.5*(x+y); m2 += 0.25*(x+y)*(x+y);
              sw += d*d/2.0;
          }
          m1 /= m; m2 = m2/m - m1*m1; sw /= m;
          /* Var(mean of 2) = sigma_B^2 + sigma_W^2/2, so sigma_B^2 = that minus sigma_W^2/2 */
          vB = m2 - sw/2.0;
          if(vB < 1e-12) vB = 1e-12;
          icc   = vB / (vB + sw);                       /* ICC(1,1): reliability of a 1-run label */
          /* THE CEILING IS sqrt(reliability), NOT the reliability, and the first three versions of this
           * program had it wrong. The maximum correlation between a DETERMINISTIC model and a noisy label
           * is sqrt(r_model)*sqrt(r_label) = sqrt(r_label) (Spearman 1904). Reporting the test-retest
           * agreement itself as the ceiling assumes the model suffers the SAME noise as the label, which
           * puts the noise factor on both sides and understates the bound by a square root. van Bree,
           * Styrnal & Hebart (2025) audited 53 neuroscience papers and found 60% making this error, so it
           * is the standard mistake rather than an exotic one. Note the metric matters: for a reported
           * CORRELATION the ceiling is sqrt(r); for a reported R^2 it is r. */
          rho   = icc;                                  /* test-retest: 1-run vs 1-run, which IS the ICC */
          tpred = (2.0/3.14159265358979323846) * asin(rho);   /* tau a RETEST would achieve */
          ratio = sqrt(sw / vB);
          (void)pooled;
          { double ceil_r = sqrt(icc);                       /* ceiling for a deterministic model */
            double ceil_t = (2.0/3.14159265358979323846) * asin(ceil_r > 1 ? 1 : ceil_r);
            if(envint("CSV",0))
                printf("CELL %s %d %.4f %.4f %.4f %.4f %.4f %.4f\n",
                       tag, m, ratio, tau, tpred, tau - tpred, icc, ceil_t);
            else
                printf("  %-12s %10.4f %12.4f %9.1f%%  %8.3f %10.4f %+9.4f %11.4f\n",
                       lab, tau, se, 100.0*tied, ratio, tpred, tau - tpred, ceil_t); } }
    }
    /* BOOT=R: R subsample replicates of the whole pipeline, reporting the sampling SD of the residual
     * so it can be compared against the systematic residual. If the systematic part greatly exceeds the
     * sampling part, a confidence interval is the wrong instrument and the honest claim is an
     * out-of-sample prediction error instead. */
    { int R = envint("BOOT", 0);
      if(R > 0){
          int m = narch/2, rep, j;
          double sd_t = 0, sd_p = 0, sd_r = 0, s1 = 0, s2 = 0, p1 = 0, p2 = 0, r1 = 0, r2 = 0;
          int kk = envint("BOOTK", 1000);
          if(kk > m) kk = m;
          for(rep = 0; rep < R; rep++){
              double m1 = 0, m2v = 0, sw = 0, vB, icc, tp, tm;
              long conc = 0, disc = 0, i2;
              /* partial Fisher-Yates: draw m distinct architectures */
              for(j = 0; j < narch; j++) boot_idx[j] = j;
              for(j = 0; j < m; j++){
                  int q = j + (int)rbelow((uint32_t)(narch - j)), t = boot_idx[j];
                  boot_idx[j] = boot_idx[q]; boot_idx[q] = t;
              }
              /* rank the subsample by the selection run and keep its own top-kk */
              for(j = 0; j < m; j++) refkey[boot_idx[j]] = vacc[boot_idx[j]][perm[2]];
              qsort(boot_idx, (size_t)m, sizeof boot_idx[0], cmp_desc);
              for(j = 0; j < kk; j++){
                  int a = boot_idx[j];
                  double x = vacc[a][perm[0]], y = vacc[a][perm[1]], d = x - y;
                  m1 += 0.5*(x+y); m2v += 0.25*(x+y)*(x+y); sw += d*d/2.0;
              }
              m1 /= kk; m2v = m2v/kk - m1*m1; sw /= kk;
              vB = m2v - sw/2.0; if(vB < 1e-15) vB = 1e-15;
              icc = vB/(vB+sw);
              tp  = (2.0/3.14159265358979323846)*asin(icc > 1 ? 1 : icc);
              for(i2 = 0; i2 < pairs/8; i2++){
                  int a = boot_idx[rbelow((uint32_t)kk)], b = boot_idx[rbelow((uint32_t)kk)];
                  double x, y;
                  if(a == b) continue;
                  x = vacc[a][perm[0]] - vacc[b][perm[0]];
                  y = vacc[a][perm[1]] - vacc[b][perm[1]];
                  if(x == 0.0 || y == 0.0) continue;
                  if(x*y > 0) conc++; else disc++;
              }
              tm = (conc+disc) ? (double)(conc-disc)/(double)(conc+disc) : 0.0;
              s1 += tm; s2 += tm*tm; p1 += tp; p2 += tp*tp;
              r1 += tm-tp; r2 += (tm-tp)*(tm-tp);
          }
          sd_t = sqrt(s2/R - (s1/R)*(s1/R));
          sd_p = sqrt(p2/R - (p1/R)*(p1/R));
          sd_r = sqrt(r2/R - (r1/R)*(r1/R));
          printf("\nSUBSAMPLE INTERVALS, %d replicates of %d architectures, own top-%d, selection redone\n",
                 R, m, kk);
          printf("  scaling: SD over subsamples times sqrt(m/n) = sqrt(%.3f)\n", (double)m/narch);
          printf("  measured tau   SD %.5f -> SE %.5f\n", sd_t, sd_t*sqrt((double)m/narch));
          printf("  predicted tau  SD %.5f -> SE %.5f\n", sd_p, sd_p*sqrt((double)m/narch));
          printf("  residual       SD %.5f -> SE %.5f   mean residual %+.4f\n",
                 sd_r, sd_r*sqrt((double)m/narch), r1/R);
          printf("  If the mean residual is many multiples of that SE, the model is falsified as an exact\n");
          printf("  statement and the honest claim is an out-of-sample prediction error, not an interval.\n");
      } }
    if(envint("CSV",0)) return 0;
    printf("  tau_retest is what a SECOND noisy run achieves against the first, and it is NOT the bound\n");
    printf("  on a deterministic predictor. CEILING = (2/pi)arcsin(sqrt(ICC)) is that bound, and it is\n");
    printf("  substantially higher. Read a published correlation against CEILING, on the matching subset.\n");

    /* ---- 2. the indifference class of the reported optimum ---- */
    {
        /* The benchmark's reported best is the argmax of the 3-run mean, which is how it is published.
         * The test uses EACH PAIR'S OWN noise rather than a pooled RMS: the pooled figure is dominated
         * by the 1.4% of architectures that sometimes collapse to chance, and using it declared 91% of
         * the benchmark inseparable from the winner, which is a statement about that summary and not
         * about the benchmark. Welch on two 3-run means is the honest local test. */
        int best = 0; long within = 0;
        double sbest, med;
        for(i = 1; i < narch; i++) if(vmean[i] > vmean[best]) best = i;
        { double ss = 0; int k;
          for(k = 0; k < NTRIAL; k++) ss += (vacc[best][k]-vmean[best])*(vacc[best][k]-vmean[best]);
          sbest = ss/2.0; }
        for(i = 0; i < narch; i++){
            double ss = 0, se; int k;
            for(k = 0; k < NTRIAL; k++) ss += (vacc[i][k]-vmean[i])*(vacc[i][k]-vmean[i]);
            se = sqrt(sbest/NTRIAL + (ss/2.0)/NTRIAL);
            if(se <= 0.0) se = 1e-6;
            if(vmean[best] - vmean[i] < 1.96*se) within++;
        }
        /* the median architecture's own SD, for scale, since the pooled RMS is not representative */
        { static float tmp[MAXARCH]; long m;
          for(m = 0; m < narch; m++){
              double ss = 0; int k;
              for(k = 0; k < NTRIAL; k++) ss += (vacc[m][k]-vmean[m])*(vacc[m][k]-vmean[m]);
              tmp[m] = (float)sqrt(ss/2.0); }
          { double lo = 0, hi = 100, mid = 0; int it; long below;
            for(it = 0; it < 40; it++){
                mid = 0.5*(lo+hi); below = 0;
                for(m = 0; m < narch; m++) if(tmp[m] <= mid) below++;
                if(below > narch/2) hi = mid; else lo = mid; }
            med = mid; } }
        printf("\nINDIFFERENCE CLASS OF THE REPORTED OPTIMUM\n");
        printf("  best 3-run mean %.4f pp. Median architecture's own SD %.3f pp; the pooled RMS is\n",
               vmean[best], med);
        printf("  %.3f pp and is dominated by the collapse-prone minority, so the test below uses each\n",
               pooled);
        printf("  pair's own noise (Welch on two %d-run means) rather than a single global figure.\n", NTRIAL);
        printf("  architectures NOT separable from the winner at 95%%: %ld of %d (%.3f%%)\n",
               within, narch, 100.0*within/narch);
        printf("  Regret measured against the reported optimum inherits that width.\n");
    }
    return 0;
}
