/* nb201_extract.c -- NAS-Bench-201 per-seed accuracies -> the same text table nb101_trials.c emits.
 * Self-contained C99, no NumPy and no PyTorch. Build: make nb201_extract
 *
 * WHY. The flip measurement needs a benchmark's individual training runs, and NAS-Bench-201 has three
 * (seeds 777/888/999) for each of 15,625 architectures on each of three datasets. The original release
 * is distributed only through Google Drive and unpickles only with PyTorch; Syne Tune publishes a
 * converted copy on Hugging Face that keeps the per-seed values as a plain NumPy array, which needs
 * neither. See PROVENANCE_nas.md for the URL.
 *
 * THE FILE. A .npy is a short text header followed by a raw C-order block:
 *     \x93NUMPY <major> <minor> <hlen:2 bytes LE> "{'descr': '<f4', 'fortran_order': False, 'shape': (...)}"
 * Here the shape is (3, 15625, 3, 200, 7) of float32:
 *     axis 0  task       cifar10, cifar100, ImageNet16-120
 *     axis 1  architecture
 *     axis 2  seed       three independent training runs, which is what this is all for
 *     axis 3  epoch      1..200
 *     axis 4  objective  0 valid_error, 1 train_error, 2 runtime, 3 elapsed, 4 latency, 5 flops, 6 params
 * so the element at (t,a,s,e,o) sits at (((t*15625 + a)*3 + s)*200 + e)*7 + o.
 *
 * NOTE the metric is an error expressed as a FRACTION, not a percentage. Reading it as a percentage
 * gave architecture 0 a validation accuracy of 99.86% on CIFAR-10, where this benchmark's best is about
 * 91.6%, which is how the mistake was caught. Accuracy is therefore 100*(1 - error). We emit accuracy in
 * percentage points to match nb101_trials.c, so one flip probe reads both benchmarks, and the program
 * refuses to write a table whose maximum falls outside a plausible band for the task.
 *
 * CAVEAT carried from the converter, and it matters for a study about replicate noise: a few
 * (architecture, seed) cells were never evaluated in the original release, and those specific cells
 * hold the mean of the remaining seeds rather than a real run. Such a cell has artificially LOW
 * variance, so it biases any noise estimate downward. We cannot identify them from this file alone,
 * which is stated here rather than glossed; NAS-Bench-101 has no such imputation and is the cleaner
 * of the two for that reason.
 *
 * usage: ./nb201_extract nb201_objectives.npy <task 0|1|2> out.txt [epoch]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NTASK  3
#define NARCH  15625
#define NSEED  3
#define NEPOCH 200
#define NOBJ   7

int main(int argc, char **argv)
{
    const char *inpath  = argc > 1 ? argv[1] : "nb201_objectives.npy";
    int         task    = argc > 2 ? atoi(argv[2]) : 0;
    const char *outpath = argc > 3 ? argv[3] : "validation/nasbench201_trials.txt";
    int         epoch   = argc > 4 ? atoi(argv[4]) : NEPOCH - 1;
    static const char *tname[NTASK] = { "cifar10", "cifar100", "ImageNet16-120" };
    FILE *f = fopen(inpath, "rb"), *out = NULL;
    unsigned char magic[8];
    unsigned char hl[2];
    char header[512];
    size_t hlen, base;
    long a, s, written = 0;
    float *buf = NULL;
    double amax = -1e9, amin = 1e9;
    /* published best validation accuracy per task, used as a sanity band rather than an assertion of
     * the exact value: a table whose maximum lands far from this means the axis or the units are wrong,
     * which is precisely the mistake this check exists to catch. */
    /* Bands set from the published best 3-seed means, widened because a SINGLE seed can exceed the mean:
     * cifar10 94.37, cifar100 73.51, ImageNet16-120 47.31. The first band was originally set from
     * cifar10-VALID's 91.61 ceiling and the check duly refused the table, which is how we learned that
     * this task axis carries plain "cifar10" -- trained on the full training set and evaluated on TEST,
     * not on a held-out validation half. The objective is still named valid_error in the file. That
     * mislabelling does not affect a measurement about replicate noise, since the three values are still
     * three independent training runs of the same architecture, but it does affect what the numbers
     * should be CALLED, so it is recorded here. */
    static const double bestlo[NTASK] = { 93.0, 70.0, 44.0 };
    static const double besthi[NTASK] = { 96.0, 78.0, 52.0 };
    int rc = 1;

    if (!f) { fprintf(stderr, "nb201_extract: cannot open %s\n", inpath); return 1; }
    if (task < 0 || task >= NTASK || epoch < 0 || epoch >= NEPOCH) {
        fprintf(stderr, "nb201_extract: task must be 0..2 and epoch 0..%d\n", NEPOCH - 1);
        goto done;
    }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\223NUMPY", 6) != 0) {
        fprintf(stderr, "nb201_extract: not a .npy file\n");
        goto done;
    }
    if (fread(hl, 1, 2, f) != 2) goto done;
    hlen = (size_t)hl[0] | ((size_t)hl[1] << 8);
    if (hlen >= sizeof header) { fprintf(stderr, "nb201_extract: header too long\n"); goto done; }
    if (fread(header, 1, hlen, f) != hlen) goto done;
    header[hlen] = '\0';
    /* the layout this program indexes by hand is only correct for these three properties */
    if (!strstr(header, "'<f4'") || !strstr(header, "False")) {
        fprintf(stderr, "nb201_extract: expected little-endian float32 C-order, got: %s\n", header);
        goto done;
    }
    if (!strstr(header, "(3, 15625, 3, 200, 7)")) {
        fprintf(stderr, "nb201_extract: unexpected shape, refusing to guess: %s\n", header);
        goto done;
    }
    base = 10 + hlen;

    /* one architecture's slice is NSEED*NEPOCH*NOBJ floats; read it whole and pick out the epoch */
    buf = malloc(sizeof(float) * NSEED * NEPOCH * NOBJ);
    out = fopen(outpath, "w");
    if (!buf || !out) { fprintf(stderr, "nb201_extract: out of memory or cannot write\n"); goto done; }

    fprintf(out, "# NAS-Bench-201 %s, epoch %d, per-SEED accuracy from 100*(1-valid_error).\n",
            tname[task], epoch + 1);
    fprintf(out, "# For task cifar10 this benchmark trains on the full training set and evaluates on\n"
                 "# TEST despite the objective's name; cifar10-valid is the held-out-half variant and is\n"
                 "# not what this axis carries.\n");
    fprintf(out, "# one architecture per line, three per-seed accuracies in percentage points\n");

    for (a = 0; a < NARCH; a++) {
        size_t off = ((size_t)task * NARCH + (size_t)a) * NSEED * NEPOCH * NOBJ;
        int bad = 0;
        float acc[NSEED];
        if (fseek(f, (long)(base + off * sizeof(float)), SEEK_SET) != 0) break;
        if (fread(buf, sizeof(float), (size_t)NSEED * NEPOCH * NOBJ, f)
            != (size_t)NSEED * NEPOCH * NOBJ) break;
        for (s = 0; s < NSEED; s++) {
            float err = buf[((size_t)s * NEPOCH + (size_t)epoch) * NOBJ + 0];  /* objective 0, a fraction */
            acc[s] = 100.0f * (1.0f - err);
            if (err < 0.0f || err > 1.0f) bad = 1;
        }
        if (bad) continue;
        for (s = 0; s < NSEED; s++) {
            fprintf(out, "%s%.4f", s ? " " : "", acc[s]);
            if (acc[s] > amax) amax = acc[s];
            if (acc[s] < amin) amin = acc[s];
        }
        fprintf(out, "\n");
        written++;
    }
    if (amax < bestlo[task] || amax > besthi[task]) {
        fprintf(stderr, "nb201_extract: REFUSING this table. Best accuracy %.2f is outside the "
                        "plausible band %.0f-%.0f for %s, so the axis or the units are wrong.\n",
                amax, bestlo[task], besthi[task], tname[task]);
        goto done;
    }
    printf("nb201_extract: %ld architectures, accuracy range %.2f to %.2f\n", written, amin, amax);
    printf("nb201_extract: %s epoch %d -> %s\n", tname[task], epoch + 1, outpath);
    rc = 0;
done:
    if (out) fclose(out);
    free(buf);
    fclose(f);
    return rc;
}
