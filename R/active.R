#' Start an Active Learning Session
#'
#' Train a classifier on a small labeled seed and select the most
#' uncertain samples from an unlabeled pool for you to label. The
#' returned session object tracks the labeled set, the pool, the
#' current query, and metrics across rounds.
#'
#' The typical workflow is:
#' \enumerate{
#'   \item Call \code{conflibert_active_start()} with your seed and pool.
#'   \item Inspect \code{session$query} and assign labels.
#'   \item Pass the labels to \code{\link{conflibert_active_next}} to
#'     retrain and query the next batch.
#'   \item Repeat until the pool is exhausted or metrics plateau.
#'   \item Persist with \code{\link{conflibert_active_save}}.
#' }
#'
#' @param seed Labeled starter set. A data.frame with \code{text} and
#'   \code{label} columns.
#' @param pool Unlabeled pool. Either a character vector of texts, or a
#'   data.frame with a \code{text} column (other columns are ignored).
#' @param dev Optional validation set with \code{text} and \code{label}
#'   columns. When provided, metrics are recorded each round.
#' @param model Base model. See \code{\link{conflibert_models}}.
#'   Default: \code{"ConfliBERT"}.
#' @param task \code{"binary"} or \code{"multiclass"}. Default:
#'   \code{"binary"}.
#' @param strategy Uncertainty strategy: \code{"entropy"} (default),
#'   \code{"margin"}, or \code{"least_confidence"}.
#' @param query_size Samples queried per round. Default: 10.
#' @param epochs Training epochs per round. Default: 3.
#' @param batch_size Training batch size. Default: 8.
#' @param lr Learning rate. Default: 2e-5.
#' @param max_seq_len Max token sequence length. Default: 512.
#' @return An object of class \code{"conflibert_al_session"}: a list
#'   with \code{query} (tibble of texts to label), \code{metrics}
#'   (tibble of metrics across rounds), \code{round}, \code{labeled_n},
#'   \code{pool_n}, \code{done}, and internal state.
#' @export
#' @examples
#' \dontrun{
#' seed <- data.frame(
#'   text  = c("Troops advanced.", "Nice weather today."),
#'   label = c(1L, 0L)
#' )
#' pool <- c("Car bomb exploded.", "New coffee shop opened.", "...")
#'
#' session <- conflibert_active_start(
#'   seed = seed, pool = pool, query_size = 5, epochs = 1
#' )
#' session
#' session$query
#' }
conflibert_active_start <- function(
    seed, pool, dev = NULL,
    model = "ConfliBERT", task = c("binary", "multiclass"),
    strategy = c("entropy", "margin", "least_confidence"),
    query_size = 10,
    epochs = 3, batch_size = 8, lr = 2e-5, max_seq_len = 512
) {
  task <- match.arg(task)
  strategy <- match.arg(strategy)

  seed_texts <- .al_check_labeled(seed, "seed")
  seed_labels <- as.integer(seed$label)
  pool_texts <- .al_check_pool(pool)
  dev_texts <- NULL
  dev_labels <- NULL
  if (!is.null(dev)) {
    dev_texts <- .al_check_labeled(dev, "dev")
    dev_labels <- as.integer(dev$label)
  }

  num_labels <- if (task == "binary") 2L else {
    as.integer(max(c(seed_labels, dev_labels)) + 1L)
  }
  query_size <- as.integer(query_size)

  py <- .get_py()

  trained <- py$al_train(
    texts       = seed_texts,
    labels      = seed_labels,
    num_labels  = num_labels,
    model_name  = model,
    epochs      = as.integer(epochs),
    batch_size  = as.integer(batch_size),
    lr          = as.numeric(lr),
    max_seq_len = as.integer(max_seq_len),
    dev_texts   = dev_texts,
    dev_labels  = dev_labels,
    task        = task
  )

  query <- .al_pick_query(
    py, trained, pool_texts,
    available_idx = seq_along(pool_texts),
    strategy = strategy, query_size = query_size,
    max_seq_len = max_seq_len
  )
  round_metrics <- .al_round_row(
    0L, length(seed_texts), trained$metrics, query$tibble$uncertainty
  )

  session <- list(
    round       = 1L,
    query       = query$tibble,
    metrics     = round_metrics,
    labeled_n   = length(seed_texts),
    pool_n      = length(pool_texts) - length(query$indices),
    done        = FALSE,
    .state = list(
      model           = trained$model,
      tokenizer       = trained$tokenizer,
      labeled_texts   = seed_texts,
      labeled_labels  = seed_labels,
      pool_texts      = pool_texts,
      pool_available  = setdiff(seq_along(pool_texts), query$indices),
      query_indices   = query$indices,
      dev_texts       = dev_texts,
      dev_labels      = dev_labels,
      num_labels      = num_labels,
      params = list(
        model = model, task = task, strategy = strategy,
        query_size = query_size, epochs = as.integer(epochs),
        batch_size = as.integer(batch_size), lr = as.numeric(lr),
        max_seq_len = as.integer(max_seq_len)
      )
    )
  )
  class(session) <- "conflibert_al_session"
  session
}


#' Submit Labels and Query the Next Batch
#'
#' Takes a session from \code{\link{conflibert_active_start}} (or a
#' previous call to this function), incorporates your labels for the
#' current query, retrains the model on the full labeled set, and
#' selects the next uncertain batch from the remaining pool.
#'
#' @param session A \code{conflibert_al_session} object.
#' @param labels Integer (or coercible) vector of labels for
#'   \code{session$query}, in the same order. Length must match
#'   \code{nrow(session$query)}.
#' @return An updated \code{conflibert_al_session}. When the pool is
#'   exhausted, \code{session$done} is \code{TRUE} and
#'   \code{session$query} is empty.
#' @export
#' @examples
#' \dontrun{
#' session <- conflibert_active_start(seed, pool)
#' labels  <- my_labeling_fn(session$query$text)
#' session <- conflibert_active_next(session, labels)
#' session$metrics
#' }
conflibert_active_next <- function(session, labels) {
  stopifnot(inherits(session, "conflibert_al_session"))
  if (isTRUE(session$done)) {
    warning("Session is already complete; returning unchanged.")
    return(session)
  }

  labels <- suppressWarnings(as.integer(labels))
  expected <- nrow(session$query)
  if (length(labels) != expected) {
    stop(sprintf(
      "Expected %d labels (one per row of session$query), got %d.",
      expected, length(labels)
    ), call. = FALSE)
  }
  if (anyNA(labels)) {
    stop("Labels contain NA; every queried sample must be labeled.", call. = FALSE)
  }
  nl <- session$.state$num_labels
  if (any(labels < 0L | labels >= nl)) {
    stop(sprintf("Labels must be integers in [0, %d].", nl - 1L), call. = FALSE)
  }

  st <- session$.state
  st$labeled_texts  <- c(st$labeled_texts, session$query$text)
  st$labeled_labels <- c(st$labeled_labels, labels)

  py <- .get_py()
  p <- st$params

  trained <- py$al_train(
    texts       = st$labeled_texts,
    labels      = st$labeled_labels,
    num_labels  = st$num_labels,
    model_name  = p$model,
    epochs      = p$epochs,
    batch_size  = p$batch_size,
    lr          = p$lr,
    max_seq_len = p$max_seq_len,
    dev_texts   = st$dev_texts,
    dev_labels  = st$dev_labels,
    task        = p$task
  )

  if (length(st$pool_available) == 0L) {
    round_row <- .al_round_row(
      session$round, length(st$labeled_texts), trained$metrics, numeric(0)
    )
    metrics <- rbind(session$metrics, round_row)
    st$model <- trained$model
    st$tokenizer <- trained$tokenizer
    session$.state <- st
    session$metrics <- metrics
    session$labeled_n <- length(st$labeled_texts)
    session$pool_n <- 0L
    session$round <- session$round + 1L
    session$query <- tibble::tibble(text = character(0))
    session$done <- TRUE
    return(session)
  }

  query <- .al_pick_query(
    py, trained, st$pool_texts,
    available_idx = st$pool_available,
    strategy = p$strategy, query_size = p$query_size,
    max_seq_len = p$max_seq_len
  )
  round_row <- .al_round_row(
    session$round, length(st$labeled_texts), trained$metrics,
    query$tibble$uncertainty
  )
  metrics <- rbind(session$metrics, round_row)

  st$pool_available <- setdiff(st$pool_available, query$indices)
  st$query_indices <- query$indices
  st$model <- trained$model
  st$tokenizer <- trained$tokenizer

  session$.state <- st
  session$metrics <- metrics
  session$labeled_n <- length(st$labeled_texts)
  session$pool_n <- length(st$pool_available)
  session$round <- session$round + 1L
  session$query <- query$tibble
  session$done <- FALSE
  session
}


#' Save the Active Learning Model
#'
#' Write the current session's model and tokenizer to disk. The result
#' is a standard HuggingFace checkpoint that can be reloaded with any
#' \code{transformers} tool.
#'
#' @param session A \code{conflibert_al_session} object.
#' @param dir Directory to write the model into. Created if it does
#'   not exist.
#' @return The directory path, invisibly.
#' @export
#' @examples
#' \dontrun{
#' conflibert_active_save(session, "my_al_model")
#' }
conflibert_active_save <- function(session, dir) {
  stopifnot(inherits(session, "conflibert_al_session"))
  if (is.null(session$.state$model)) {
    stop("Session has no trained model to save.", call. = FALSE)
  }
  dir <- path.expand(dir)
  py <- .get_py()
  py$al_save(session$.state$model, session$.state$tokenizer, dir)
  invisible(dir)
}


#' @export
print.conflibert_al_session <- function(x, n = 5, ...) {
  header <- if (isTRUE(x$done)) {
    sprintf("Active learning session  (complete, %d rounds)", x$round - 1L)
  } else {
    sprintf("Active learning session  (round %d)", x$round)
  }
  cat(header, "\n", sep = "")
  p <- x$.state$params
  cat(sprintf(
    "  Model: %s  |  Task: %s  |  Strategy: %s\n",
    p$model, p$task, p$strategy
  ))
  cat(sprintf(
    "  Labeled: %d  |  Pool remaining: %d  |  Query size: %d\n",
    x$labeled_n, x$pool_n, p$query_size
  ))

  if (nrow(x$metrics) > 0L) {
    cat("\nMetrics by round:\n")
    print(x$metrics, n = Inf)
  }

  if (!isTRUE(x$done) && nrow(x$query) > 0L) {
    cat(sprintf(
      "\nCurrent query (%d samples, showing %d):\n",
      nrow(x$query), min(n, nrow(x$query))
    ))
    print(utils::head(x$query, n))
    cat("\nLabel these, then call:  session <- conflibert_active_next(session, labels)\n")
  } else if (isTRUE(x$done)) {
    cat("\nPool exhausted. Save with:  conflibert_active_save(session, dir)\n")
  }
  invisible(x)
}


#' Plot an Active Learning Session
#'
#' Visualize how the model is improving across rounds. The default
#' view is a two-panel plot: the learning curve on top (metrics vs
#' training set size) and the uncertainty trend on the bottom (mean
#' uncertainty of each round's queries). The uncertainty trend is a
#' useful signal: when it flattens, the model is no longer finding
#' informative samples.
#'
#' @param x A \code{conflibert_al_session} object.
#' @param which One of \code{"all"} (default, two panels),
#'   \code{"metrics"} (learning curve only), or \code{"uncertainty"}
#'   (uncertainty trend only).
#' @param ... Passed to the underlying plot call.
#' @return The session, invisibly.
#' @export
plot.conflibert_al_session <- function(
    x, which = c("all", "metrics", "uncertainty"), ...
) {
  which <- match.arg(which)
  m <- x$metrics
  if (nrow(m) == 0L) {
    message("No metrics to plot yet.")
    return(invisible(x))
  }

  base_cols <- c("round", "train_size", "uncertainty_mean", "uncertainty_max")
  metric_cols <- setdiff(names(m), base_cols)
  has_metrics <- length(metric_cols) > 0L &&
    any(!is.na(unlist(m[metric_cols])))
  has_uncert <- any(!is.na(m$uncertainty_mean))

  if (which == "metrics" && !has_metrics) {
    message("No dev metrics recorded. Pass `dev` to conflibert_active_start() ",
            "to track metrics across rounds.")
    return(invisible(x))
  }
  if (which == "uncertainty" && !has_uncert) {
    message("No uncertainty data recorded yet.")
    return(invisible(x))
  }

  oldpar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldpar), add = TRUE)

  show_metrics <- which %in% c("all", "metrics") && has_metrics
  show_uncert  <- which %in% c("all", "uncertainty") && has_uncert
  n_panels <- show_metrics + show_uncert
  if (n_panels == 0L) {
    message("Nothing to plot.")
    return(invisible(x))
  }
  if (n_panels == 2L) graphics::par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

  cols <- c("#ff6b35", "#3b82f6", "#10b981", "#8b5cf6", "#ef4444", "#64748b")
  xs <- m$train_size

  if (show_metrics) {
    plot(xs, m[[metric_cols[1]]], type = "n", ylim = c(0, 1),
         xlab = "Labeled examples", ylab = "Score",
         main = "Learning curve", ...)
    graphics::grid(col = "grey85", lty = 3)
    for (i in seq_along(metric_cols)) {
      vals <- m[[metric_cols[i]]]
      col <- cols[((i - 1L) %% length(cols)) + 1L]
      graphics::lines(xs, vals, col = col, lwd = 2)
      graphics::points(xs, vals, col = col, pch = 19, cex = 1.1)
    }
    graphics::legend(
      "bottomright", legend = metric_cols,
      col = cols[seq_along(metric_cols)], lwd = 2, pch = 19, bty = "n"
    )
  }

  if (show_uncert) {
    keep <- !is.na(m$uncertainty_mean)
    ux <- m$train_size[keep]
    umean <- m$uncertainty_mean[keep]
    umax <- m$uncertainty_max[keep]
    ylim <- range(c(umean, umax), na.rm = TRUE)
    if (diff(ylim) == 0) ylim <- ylim + c(-0.01, 0.01)
    plot(ux, umean, type = "n", ylim = ylim,
         xlab = "Labeled examples", ylab = "Query uncertainty",
         main = "Uncertainty of queried samples")
    graphics::grid(col = "grey85", lty = 3)
    graphics::polygon(
      c(ux, rev(ux)), c(umean, rev(umax)),
      col = grDevices::adjustcolor("#3b82f6", alpha.f = 0.15), border = NA
    )
    graphics::lines(ux, umax, col = "#3b82f6", lwd = 1, lty = 2)
    graphics::lines(ux, umean, col = "#3b82f6", lwd = 2)
    graphics::points(ux, umean, col = "#3b82f6", pch = 19, cex = 1.1)
    graphics::legend(
      "topright", legend = c("mean", "max"),
      col = "#3b82f6", lwd = c(2, 1), lty = c(1, 2), bty = "n"
    )
  }

  invisible(x)
}


# ---- internal helpers -------------------------------------------------

.al_check_labeled <- function(df, name) {
  if (!is.data.frame(df) || !all(c("text", "label") %in% names(df))) {
    stop(sprintf("`%s` must be a data.frame with `text` and `label` columns.",
                 name), call. = FALSE)
  }
  as.character(df$text)
}

.al_check_pool <- function(pool) {
  if (is.data.frame(pool)) {
    if (!"text" %in% names(pool)) {
      stop("`pool` data.frame must have a `text` column.", call. = FALSE)
    }
    return(as.character(pool$text))
  }
  if (!is.character(pool)) pool <- as.character(pool)
  if (length(pool) == 0L) stop("`pool` is empty.", call. = FALSE)
  pool
}

.al_round_row <- function(round, train_size, metrics, uncertainty) {
  row <- tibble::tibble(round = as.integer(round),
                        train_size = as.integer(train_size))
  for (k in names(metrics)) row[[k]] <- as.numeric(metrics[[k]])
  row$uncertainty_mean <- if (length(uncertainty)) {
    round(mean(uncertainty), 4)
  } else NA_real_
  row$uncertainty_max <- if (length(uncertainty)) {
    round(max(uncertainty), 4)
  } else NA_real_
  row
}

.al_pick_query <- function(py, trained, pool_texts, available_idx,
                            strategy, query_size, max_seq_len) {
  available_idx <- as.integer(available_idx)
  texts_sub <- pool_texts[available_idx]
  scores <- unlist(py$al_score(
    model = trained$model,
    tokenizer = trained$tokenizer,
    texts = texts_sub,
    strategy = strategy,
    max_seq_len = as.integer(max_seq_len)
  ))
  k <- min(as.integer(query_size), length(texts_sub))
  top_local <- order(scores, decreasing = TRUE)[seq_len(k)]
  top_idx <- available_idx[top_local]
  list(
    indices = top_idx,
    tibble = tibble::tibble(
      text = pool_texts[top_idx],
      uncertainty = round(as.numeric(scores[top_local]), 4)
    )
  )
}
