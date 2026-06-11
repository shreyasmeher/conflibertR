# Active Learning Walkthrough: conflibertR
#
# Run line-by-line in RStudio, or `Rscript walkthrough.R`.
# If RStudio's Plots pane is short:  dev.new(width = 8, height = 8, noRStudioGD = TRUE)

if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all("/Users/apple/Documents/conflibertR")
} else {
  library(conflibertR)
}

# ---- Data ---------------------------------------------------------------
demo <- conflibert_example("active")
cat(sprintf("Seed %d  |  Pool %d  |  Dev %d  |  Oracle %d (simulation)\n",
            nrow(demo$seed), length(demo$pool),
            nrow(demo$dev),  length(demo$pool_labels)))

# ---- Round 1 ------------------------------------------------------------
session <- conflibert_active_start(
  seed       = demo$seed,
  pool       = demo$pool,
  dev        = demo$dev,
  task       = "binary",
  strategy   = "entropy",     # or "margin", "least_confidence"
  query_size = 8,
  epochs     = 3
)
print(session)
plot(session)

# ---- Rounds 2..N (oracle simulation) ------------------------------------
# Real workflow: labels <- conflibert_active_label(session)
n_rounds <- 5
for (i in seq_len(n_rounds)) {
  if (session$done) break
  labels  <- unname(demo$pool_labels[session$query$text])
  session <- conflibert_active_next(session, labels = labels)
  cat(sprintf("Round %d  labeled=%-3d  pool=%-3d  mean_unc=%.3f\n",
              session$round - 1L, session$labeled_n, session$pool_n,
              mean(session$query$uncertainty)))
}

plot(session)

# =========================================================================
# Diagnostics
# =========================================================================
m <- session$metrics
cat("\n=== Metrics by round ===\n"); print(m)

# Lift over the seed-only baseline (round 0).
delta <- function(col) tail(m[[col]], 1) - m[[col]][1]
cat(sprintf(
  "\nLift   F1: %+.3f   Acc: %+.3f   Prec: %+.3f   Recall: %+.3f\n",
  delta("f1"), delta("accuracy"), delta("precision"), delta("recall")
))
cat(sprintf("Labels added by AL: %d / pool=%d  (%.0f%% of pool)\n",
            session$labeled_n - nrow(demo$seed),
            length(demo$pool),
            100 * (session$labeled_n - nrow(demo$seed)) / length(demo$pool)))

# Class balance of the full labeled set; uncertainty sampling can drift
# toward one class, so it's worth a glance.
bal <- table(factor(session$.state$labeled_labels))
cat("\nLabeled-set class balance:\n"); print(bal)

op <- graphics::par(no.readonly = TRUE); on.exit(graphics::par(op), add = TRUE)
graphics::par(mar = c(4, 4, 3, 1))
graphics::barplot(bal,
  main = "Labeled set: class counts",
  xlab = "class", ylab = "n",
  col  = "steelblue", border = NA
)

# Dev predictions: round-trip the model through save/load and predict.
tmp_dir <- file.path(tempdir(), "al_walkthrough_model")
conflibert_active_save(session, tmp_dir)
clf  <- conflibert_load(tmp_dir)
pred <- predict(clf, session$.state$dev_texts)

truth <- session$.state$dev_labels
preds <- pred$class
hits  <- preds == truth

cat("\n=== Dev confusion matrix ===\n")
print(table(truth = truth, pred = preds))
cat(sprintf("Dev accuracy: %.3f  (%d / %d)\n",
            mean(hits), sum(hits), length(hits)))

# Confidence on dev, split by correct vs. wrong. Confident-and-wrong
# (right-side red bars) are the calibration problems worth chasing.
br   <- seq(0.5, 1, by = 0.05)
h_ok <- graphics::hist(pred$confidence[ hits], breaks = br, plot = FALSE)
h_no <- graphics::hist(pred$confidence[!hits], breaks = br, plot = FALSE)
graphics::par(mar = c(4, 4, 3, 1))
graphics::plot(h_ok,
  col = grDevices::adjustcolor("steelblue", 0.7), border = NA,
  xlim = c(0.5, 1), ylim = c(0, max(h_ok$counts, h_no$counts)),
  main = "Dev confidence: correct vs. wrong",
  xlab = "max class probability"
)
graphics::plot(h_no,
  col = grDevices::adjustcolor("firebrick", 0.7), border = NA,
  add = TRUE
)
graphics::legend("topleft", c("correct", "wrong"),
  fill = c("steelblue", "firebrick"), bty = "n"
)

# Persist the final model (optional):
# conflibert_active_save(session, "my_al_model")
