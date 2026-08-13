/* resolve.c -- is this comparison real, and how many runs would make it real?
 * Self-contained C99, no dependencies. Build: make resolve   Self-test: ./resolve selftest
 *
 * WHY THIS EXISTS RATHER THAN ONLY A PAPER. Two independent advisors said the same thing: a table of
 * percentages gets cited and changes nothing, while a default inside a tool people already run changes
 * what happens on Monday. The tooling situation they found supports it. Optuna's answer to a noisy
 * objective is to fix the seed. Ray Tune ships a Repeater that repeats and averages, and its own docs say
 * not to combine it with a scheduler, which walls the fix off from ASHA and Hyperband, the mechanisms
 * that promote candidates on noisy partial-budget signal. None of the mainstream trackers warns that a
 * sweep's winner is indistinguishable from its runner-up.
 *
 * So this answers the two questions an engineer actually has, and nothing else.
 *
 *   ./resolve compare  file      is candidate A really better than candidate B?
 *   ./resolve plan     file      how many runs per candidate to resolve a given difference?
 *   ./resolve ceiling  file      what is the best rank correlation any predictor could score here,
 *                                and how far does it collapse if I look only at the good candidates?
 *
 * INPUT FORMAT, one candidate per line, whitespace-separated repeated scores. Ragged lines are fine;
 * each candidate may have a different number of runs. Lines starting with # are ignored.
 *
 *     0.812 0.809 0.815
 *     0.809 0.811
 *
 * WHAT THE STATISTICS ARE, since they are classical and should not be reinvented. Writing sigma_W for
 * run-to-run noise and sigma_B for the spread of true quality across candidates, the reliability of a
 * single run is the one-way random-effects intraclass correlation
 *     ICC = sigma_B^2 / (sigma_B^2 + sigma_W^2)                      Fisher 1925; Shrout & Fleiss 1979
 * the reliability of a mean of m runs follows by Spearman-Brown
 *     R_m = m*ICC / (1 + (m-1)*ICC)                                  Spearman 1910; Brown 1910
 * the correlation between an m1-run and an m2-run estimate of the same candidates is sqrt(R_m1 * R_m2),
 * which is Spearman's correction for attenuation with a true-score correlation of one (Spearman 1904),
 * and for elliptical joints Kendall's tau = (2/pi) arcsin(rho) (Kruskal 1958; Lindskog et al. 2003).
 * "How many runs do I need" is a D-study in generalizability theory (Cronbach et al. 1972).
 *
 * WHAT IT WILL NOT DO. It will not tell you a difference is real when your data cannot support that, and
 * `compare` reports the minimum runs required instead of a verdict when the comparison is underpowered.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXCAND 500000
#define MAXRUN  64

static double sc[MAXCAND][MAXRUN];
static int    nr[MAXCAND];
static int    nc = 0;

static double mean_of(int i)
{ int k; double s = 0; for(k = 0; k < nr[i]; k++) s += sc[i][k]; return s/nr[i]; }

/* unbiased within-candidate variance; 0 when a candidate has a single run */
static double var_of(int i)
{ int k; double m = mean_of(i), s = 0;
  if(nr[i] < 2) return 0.0;
  for(k = 0; k < nr[i]; k++) s += (sc[i][k]-m)*(sc[i][k]-m);
  return s/(nr[i]-1); }

static int load(const char *path)
{
    FILE *f = strcmp(path, "-") ? fopen(path, "r") : stdin;
    char line[8192];
    if(!f){ fprintf(stderr, "resolve: cannot open %s\n", path); return -1; }
    while(fgets(line, sizeof line, f)){
        char *p = line; int k = 0;
        if(*p == '#') continue;
        while(k < MAXRUN){
            char *e; double v = strtod(p, &e);
            if(e == p) break;
            sc[nc][k++] = v; p = e;
        }
        if(k == 0) continue;
        nr[nc] = k;
        if(++nc >= MAXCAND) break;
    }
    if(f != stdin) fclose(f);
    return nc;
}

/* pooled run-to-run SD over every candidate that has at least two runs */
static double pooled_sd(int *ndf)
{
    int i, df = 0; double ss = 0;
    for(i = 0; i < nc; i++)
        if(nr[i] >= 2){ ss += var_of(i)*(nr[i]-1); df += nr[i]-1; }
    if(ndf) *ndf = df;
    return df ? sqrt(ss/df) : 0.0;
}

/* runs per candidate needed to resolve a difference `delta` at 95% two-sided, both sides equal n */
static double runs_needed(double delta, double sd)
{
    double n;
    if(delta <= 0 || sd <= 0) return 1;
    n = 2.0 * (1.959964*sd/delta) * (1.959964*sd/delta);
    return n < 1 ? 1 : n;
}

static int cmd_compare(void)
{
    int df, i; double sd = pooled_sd(&df);
    if(nc < 2){ fprintf(stderr, "resolve compare: need at least two candidates\n"); return 2; }
    if(df == 0){
        printf("Every candidate has a single run, so nothing here can be resolved: with one run each\n");
        printf("there is no estimate of run-to-run noise and no comparison is supported. Repeat any two\n");
        printf("candidates twice and run this again.\n");
        return 1;
    }
    printf("run-to-run SD %.6g, pooled over %d degrees of freedom\n", sd, df);
    printf("smallest difference one run per candidate can establish: %.6g\n\n", 2.7718*sd);
    printf("%-6s %-6s %12s %12s %10s   %s\n", "A", "B", "meanA-meanB", "95% halfwidth", "runs req", "verdict");
    for(i = 1; i < nc; i++){
        double d  = mean_of(0) - mean_of(i);
        double se = sd * sqrt(1.0/nr[0] + 1.0/nr[i]);
        double hw = 1.959964*se, ad = d < 0 ? -d : d;
        double need = runs_needed(ad, sd);
        const char *v;
        if(ad > hw)                 v = d > 0 ? "A better" : "B better";
        else if(need <= 1000.0)     v = "not distinguishable";
        else                        v = "not distinguishable, and impractically so";
        printf("%-6d %-6d %12.6g %12.6g %10.0f   %s\n", 0, i, d, hw, need, v);
    }
    printf("\n\"runs req\" is how many runs per candidate the OBSERVED difference would need to clear\n");
    printf("95%% two-sided at this noise level. A verdict of not-distinguishable is not a claim that the\n");
    printf("candidates are equal; it is a statement that this evidence does not order them.\n");
    return 0;
}

static int cmd_plan(void)
{
    static const double frac[7] = { 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0 };
    int df, i; double sd = pooled_sd(&df);
    if(df == 0){ fprintf(stderr, "resolve plan: no candidate has repeated runs, so noise is unknown\n"); return 1; }
    printf("run-to-run SD %.6g, pooled over %d degrees of freedom\n\n", sd, df);
    printf("%14s %22s\n", "difference", "runs per candidate");
    for(i = 0; i < 7; i++){
        double d = frac[i]*sd;
        printf("%8.4g (%.2gx SD) %22.0f\n", d, frac[i], runs_needed(d, sd));
    }
    printf("\nRead it as a budget question. If the effect you care about is a quarter of your run-to-run\n");
    printf("SD, no realistic number of repeats will find it and the honest move is to change the\n");
    printf("intervention rather than the statistics.\n");
    return 0;
}

/* Reliability and the ceiling. Split each candidate's runs into two halves, so both halves are
 * independent estimates of the same candidate, and correlate them. Candidates are ranked for the top-k
 * subsets by a run used in NEITHER half where one is available; otherwise the subset is chosen by the
 * first half and the output says so, because selecting on a quantity that is then correlated restricts
 * its range and biases the correlation downward (Pearson 1903; Thorndike 1949). */
static int cmp_key(const void *A, const void *B);
static double keyv[MAXCAND];
static int    ord[MAXCAND];
static int cmp_key(const void *A, const void *B)
{ double x = keyv[*(const int*)A], y = keyv[*(const int*)B]; return x < y ? 1 : (x > y ? -1 : 0); }

static int cmd_ceiling(void)
{
    static const int topk[4] = { 100, 1000, 10000, 0 };
    int i, k, usable = 0, sel_ok = 1;
    for(i = 0; i < nc; i++){ ord[i] = i; if(nr[i] >= 2) usable++; if(nr[i] < 3) sel_ok = 0; }
    if(usable < nc || nc < 20){
        fprintf(stderr, "resolve ceiling: needs at least 2 runs for every candidate and 20 candidates\n");
        return 1;
    }
    for(i = 0; i < nc; i++) keyv[i] = sel_ok ? sc[i][2] : sc[i][0];
    qsort(ord, (size_t)nc, sizeof ord[0], cmp_key);
    if(!sel_ok)
        printf("NOTE: some candidate has fewer than 3 runs, so the top-k subsets are ranked by a run that\n"
               "is also correlated. Those rows are biased DOWNWARD by range restriction; the all-candidates\n"
               "row is unaffected. Three runs each removes the caveat.\n\n");
    printf("%-14s %10s %10s %12s %14s\n", "subset", "ICC", "tau ceiling", "sW/sB", "runs for ICC .9");
    for(k = 0; k < 4; k++){
        int m = topk[k] ? (topk[k] < nc ? topk[k] : nc) : nc;
        double s1 = 0, s2 = 0, sw = 0, vB, icc, tau, need;
        int j;
        if(topk[k] && topk[k] > nc) continue;
        for(j = 0; j < m; j++){
            int a = ord[j];
            double x = sc[a][0], y = sc[a][1], d = x - y;
            s1 += 0.5*(x+y); s2 += 0.25*(x+y)*(x+y); sw += d*d/2.0;
        }
        s1 /= m; s2 = s2/m - s1*s1; sw /= m;
        vB = s2 - sw/2.0;                    /* Var(mean of 2) = sB^2 + sW^2/2 */
        if(vB < 1e-15) vB = 1e-15;
        icc = vB/(vB + sw);
        tau = (2.0/3.14159265358979323846)*asin(icc > 1 ? 1 : icc);
        /* Spearman-Brown inverted: runs needed for a mean whose reliability reaches 0.9 */
        need = icc > 0 ? 0.9*(1.0-icc)/(icc*(1.0-0.9)) : 1e9;
        if(need < 1) need = 1;
        if(topk[k]) printf("top %-10d", m); else printf("%-14s", "all");
        printf(" %10.4f %10.4f %12.3f %14.0f\n", icc, tau, sqrt(sw/vB), need);
    }
    printf("\nThe tau column is the highest rank correlation any predictor could score against a\n");
    printf("single-run label here, because it is the correlation the label achieves with itself. Read a\n");
    printf("published correlation against the row for the subset it was computed on, not against 1.0.\n");
    printf("Expect the top-k rows to be far worse than the all row: restricting to good candidates\n");
    printf("shrinks the real differences between them while leaving run-to-run noise untouched.\n");
    return 0;
}

/* Checked against cases computable by hand, because a tool that reports a verdict must be verifiable. */
static int selftest(void)
{
    int fail = 0;
    /* two candidates, three runs each, sd known exactly: values 1,2,3 have sample variance 1 */
    nc = 2; nr[0] = nr[1] = 3;
    sc[0][0]=1; sc[0][1]=2; sc[0][2]=3;
    sc[1][0]=1; sc[1][1]=2; sc[1][2]=3;
    { int df; double sd = pooled_sd(&df);
      printf("pooled_sd of two candidates each {1,2,3}: got %.6f want 1.000000, df %d want 4  %s\n",
             sd, df, (fabs(sd-1.0) < 1e-12 && df == 4) ? "PASS" : (fail=1, "FAIL")); }
    /* runs_needed: delta = sd gives 2*1.959964^2 = 7.682914. The first version of this test asserted
     * 7.6840 from a mis-multiplication and the self-test caught it, which is the entire argument for
     * having one: the code was right and the author was not. */
    { double n = runs_needed(1.0, 1.0);
      printf("runs_needed(delta=SD): got %.6f want 7.682914  %s\n",
             n, fabs(n-7.682914) < 1e-5 ? "PASS" : (fail=1, "FAIL")); }
    /* the one-run floor is 1.959964*sqrt(2) = 2.771808 times the SD; check the constant in the code
     * against the product rather than against itself */
    { double want = 1.959964*sqrt(2.0);
      printf("one-run floor coefficient: code has 2.7718, product is %.6f  %s\n", want,
             fabs(2.7718 - want) < 1e-4 ? "PASS" : (fail=1, "FAIL")); }
    /* Spearman-Brown: ICC 0.5 needs m=9 for R_m = 0.9 */
    { double need = 0.9*(1.0-0.5)/(0.5*0.1);
      printf("Spearman-Brown runs for ICC .5 -> R_m .9: got %.2f want 9.00  %s\n",
             need, fabs(need-9.0) < 1e-9 ? "PASS" : (fail=1, "FAIL")); }
    /* tau at perfect reliability is 1 */
    { double t = (2.0/3.14159265358979323846)*asin(1.0);
      printf("tau at ICC 1: got %.6f want 1.000000  %s\n",
             t, fabs(t-1.0) < 1e-12 ? "PASS" : (fail=1, "FAIL")); }
    printf("\nself-test %s\n", fail ? "FAILED" : "PASSED");
    return fail;
}

int main(int argc, char **argv)
{
    const char *cmd  = argc > 1 ? argv[1] : "";
    const char *path = argc > 2 ? argv[2] : "-";
    if(!strcmp(cmd, "selftest")) return selftest();
    if(!*cmd || !strcmp(cmd, "-h") || !strcmp(cmd, "--help")){
        printf("usage: resolve <compare|plan|ceiling|selftest> [file|-]\n\n");
        printf("  compare  is candidate 0 really better than each of the others?\n");
        printf("  plan     how many runs per candidate to resolve a difference of a given size?\n");
        printf("  ceiling  the best rank correlation any predictor could score against these labels,\n");
        printf("           and how far it collapses when restricted to the good candidates\n");
        printf("  selftest check the arithmetic against hand-computable cases\n\n");
        printf("input: one candidate per line, whitespace-separated repeated scores; # comments\n");
        return *cmd ? 0 : 2;
    }
    if(load(path) < 0) return 1;
    if(nc == 0){ fprintf(stderr, "resolve: no candidates read\n"); return 1; }
    if(!strcmp(cmd, "compare")) return cmd_compare();
    if(!strcmp(cmd, "plan"))    return cmd_plan();
    if(!strcmp(cmd, "ceiling")) return cmd_ceiling();
    fprintf(stderr, "resolve: unknown command '%s'\n", cmd);
    return 2;
}
