# nnet.R -- the same fit, by R's nnet, for comparison with ./bpnn.
#
# nnet is the standard single-hidden-layer network in R and has been since Venables and Ripley.
# It is not the same algorithm: it minimises by BFGS over the whole training set, where bpnn
# takes one stochastic gradient step per row. So this is not an implementation check of the sort
# `make r` is in linearr, where both programs solve the same equations and must agree to eleven
# digits. Two different optimisers on a non-convex surface have no reason to land in the same
# place. What it does check is whether the C program reaches the same QUALITY on held-out rows,
# which is the only claim it makes.
#
# To keep the comparison about the optimiser and not about preprocessing, this scales exactly as
# bpnn does: every term to [0, 1] over its own range, the response to [0.1, 0.9], the same
# fractions held out, and the same count of refits from different random starts.
#
#     Rscript bench/nnet.R FILE [hidden] [holdout] [seeds] [maxit] [decay]
#
# Prints one line per group: the held-out RMSE of the median refit, and the spread over refits.

suppressPackageStartupMessages(library(nnet))

args   <- commandArgs(trailingOnly = TRUE)
file   <- args[1]
hidden <- if (length(args) > 1) as.integer(args[2]) else 6L
holdout<- if (length(args) > 2) as.numeric(args[3]) else 0.25
seeds  <- if (length(args) > 3) as.integer(args[4]) else 5L
maxit  <- if (length(args) > 4) as.integer(args[5]) else 200L
decay  <- if (length(args) > 5) as.numeric(args[6]) else 0.0

d <- read.csv(file, comment.char = "#", stringsAsFactors = FALSE)
gcol <- names(d)[1]
ycol <- names(d)[2]
tcol <- names(d)[-(1:2)]

TLO <- 0.1; THI <- 0.9

for (g in unique(d[[gcol]])) {
    rows <- d[d[[gcol]] == g, , drop = FALSE]
    n <- nrow(rows)
    if (n < 4) next

    # bpnn's scaling: each term over its own range, the response into [TLO, THI]
    X <- as.matrix(rows[, tcol, drop = FALSE])
    lo <- apply(X, 2, min); hi <- apply(X, 2, max)
    hi[hi <= lo] <- lo[hi <= lo] + 1
    Xs <- sweep(sweep(X, 2, lo, "-"), 2, hi - lo, "/")
    y <- rows[[ycol]]
    tlo <- min(y); thi <- max(y); if (thi <= tlo) thi <- tlo + 1
    ys <- TLO + (THI - TLO) * (y - tlo) / (thi - tlo)

    ntr <- round((1 - holdout) * n)
    held <- numeric(seeds)
    for (s in seq_len(seeds)) {
        set.seed(1000 + 7919 * (s - 1))
        ord <- sample(n)
        tr <- ord[seq_len(ntr)]; te <- ord[-seq_len(ntr)]
        fit <- nnet(Xs[tr, , drop = FALSE], ys[tr], size = hidden, decay = decay,
                    maxit = maxit, trace = FALSE, linout = FALSE)
        p <- as.numeric(predict(fit, Xs[te, , drop = FALSE]))
        # back to the response's own units, which is what bpnn reports
        pu <- tlo + (p - TLO) / (THI - TLO) * (thi - tlo)
        held[s] <- sqrt(mean((pu - y[te])^2))
    }
    cat(sprintf("%-10s %6d %6d %10.5g %10.5g\n",
                g, n, n - ntr, median(held), sd(held)))
}
