#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# End-to-end smoke test for the active-learning workflow.
#
# Runs a full simulated active-learning loop on the bundled example data:
# starts from a 20-text seed, queries 10 uncertain texts per round from a
# 61-text pool, uses the bundled oracle labels to answer, and plots the
# learning curve + uncertainty trend.
#
# Usage:
#   Rscript test_active_learning.R            # 3 rounds, ConfliBERT, 1 epoch
#   ROUNDS=5 Rscript test_active_learning.R   # override number of rounds
#
# Expect ~1–3 minutes per round on CPU with the defaults.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all(".", quiet = TRUE)
  } else {
    library(conflibertR)
  }
})

rounds     <- as.integer(Sys.getenv("ROUNDS", "3"))
epochs     <- as.integer(Sys.getenv("EPOCHS", "1"))
query_size <- as.integer(Sys.getenv("QUERY_SIZE", "10"))
model      <- Sys.getenv("MODEL", "ConfliBERT")
strategy   <- Sys.getenv("STRATEGY", "entropy")

cat("Settings:\n")
cat(sprintf("  rounds=%d  epochs=%d  query_size=%d  model=%s  strategy=%s\n\n",
            rounds, epochs, query_size, model, strategy))

data <- conflibert_example("active")
oracle <- data$pool_labels
stopifnot(!is.null(oracle), length(oracle) == length(data$pool))

cat(sprintf("Seed: %d labeled  |  Pool: %d unlabeled  |  Dev: %d\n\n",
            nrow(data$seed), length(data$pool), nrow(data$dev)))

# 1. Start the session — trains on the seed, returns the first query.
cat("=== Round 1: training on seed and querying... ===\n")
session <- conflibert_active_start(
  seed        = data$seed,
  pool        = data$pool,
  dev         = data$dev,
  model       = model,
  task        = "binary",
  strategy    = strategy,
  query_size  = query_size,
  epochs      = epochs
)
print(session)

# 2. Loop: use the oracle to "label" each query, then submit and continue.
for (r in seq_len(rounds - 1L)) {
  if (session$done) {
    cat("\nPool exhausted — stopping early.\n")
    break
  }
  labels <- unname(oracle[session$query$text])
  if (anyNA(labels)) {
    stop("Oracle could not find labels for some queried texts.", call. = FALSE)
  }
  cat(sprintf("\n=== Round %d: submitting %d labels, retraining... ===\n",
              r + 1L, length(labels)))
  session <- conflibert_active_next(session, labels = labels)
  print(session)
}

# 3. Final summary.
cat("\n\n=== Final metrics ===\n")
print(session$metrics, n = Inf)

# 4. Save plots side-by-side so you can eyeball the run without a display.
out_pdf <- "active_learning_demo.pdf"
grDevices::pdf(out_pdf, width = 8, height = 8)
plot(session)                       # default: metrics + uncertainty
grDevices::dev.off()
cat(sprintf("\nPlots written to %s\n", normalizePath(out_pdf)))

# 5. Save the trained model to a temp dir and report the path.
out_model <- tempfile("conflibert_al_model_")
conflibert_active_save(session, out_model)
cat(sprintf("Model checkpoint saved to %s\n", out_model))

cat("\nDone.\n")
