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
#' @param diverse If \code{TRUE}, cluster the top-scoring candidates in
#'   embedding space and pick the highest-scoring sample from each
#'   cluster. Prevents near-duplicates from dominating a batch.
#'   Default: \code{FALSE}.
#' @param diversity_candidates How many top-scoring candidates to
#'   cluster when \code{diverse = TRUE}. Default: \code{3 * query_size}.
#' @param query_size Samples queried per round. Default: 10.
#' @param epochs Training epochs per round. Default: 3.
#' @param batch_size Training batch size. Default: 8.
#' @param lr Learning rate. Default: 2e-5.
#' @param max_seq_len Max token sequence length. Default: 512.
#' @param use_lora If \code{TRUE}, train each round with a LoRA adapter
#'   (parameter-efficient). Default: \code{FALSE}.
#' @param lora_rank LoRA rank. Default: 8.
#' @param lora_alpha LoRA alpha. Default: 16.
#' @param random_seed Random seed for reproducibility (named to avoid a
#'   clash with the labeled \code{seed} set above). Seeds each round's
#'   model training and, when \code{diverse = TRUE}, the k-means
#'   clustering used to pick a diverse query, so re-running the whole
#'   session on the same hardware and package versions reproduces the
#'   same queries and metrics. It is stored in the session and reused by
#'   \code{\link{conflibert_active_next}}. Default: 42.
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
    diverse = FALSE, diversity_candidates = NULL,
    query_size = 10,
    epochs = 3, batch_size = 8, lr = 2e-5, max_seq_len = 512,
    use_lora = FALSE, lora_rank = 8, lora_alpha = 16,
    random_seed = 42
) {
  task <- match.arg(task)
  strategy <- match.arg(strategy)
  random_seed <- as.integer(random_seed)

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
    task        = task,
    use_lora    = isTRUE(use_lora),
    lora_rank   = as.integer(lora_rank),
    lora_alpha  = as.integer(lora_alpha),
    seed        = random_seed
  )

  dc <- if (is.null(diversity_candidates)) {
    3L * as.integer(query_size)
  } else as.integer(diversity_candidates)

  query <- .al_pick_query(
    py, trained, pool_texts,
    available_idx = seq_along(pool_texts),
    strategy = strategy, query_size = query_size,
    max_seq_len = max_seq_len,
    diverse = isTRUE(diverse), diversity_candidates = dc,
    seed = random_seed
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
        diverse = isTRUE(diverse), diversity_candidates = dc,
        query_size = query_size, epochs = as.integer(epochs),
        batch_size = as.integer(batch_size), lr = as.numeric(lr),
        max_seq_len = as.integer(max_seq_len),
        use_lora = isTRUE(use_lora),
        lora_rank = as.integer(lora_rank),
        lora_alpha = as.integer(lora_alpha),
        seed = random_seed
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
    task        = p$task,
    use_lora    = isTRUE(p$use_lora),
    lora_rank   = as.integer(p$lora_rank),
    lora_alpha  = as.integer(p$lora_alpha),
    seed        = as.integer(p$seed)
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
    max_seq_len = p$max_seq_len,
    diverse = isTRUE(p$diverse),
    diversity_candidates = as.integer(p$diversity_candidates),
    seed = as.integer(p$seed)
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


#' Label the Current Query Interactively
#'
#' Opens a small Shiny gadget that shows each text in
#' \code{session$query} alongside radio buttons for each class. Click
#' Done to submit; the labels are returned as an integer vector ready
#' to pass to \code{\link{conflibert_active_next}}.
#'
#' Requires the \pkg{shiny} and \pkg{miniUI} packages. In RStudio the
#' gadget opens as a modal dialog; elsewhere it opens in your browser.
#'
#' @param session A \code{conflibert_al_session} object with a
#'   non-empty \code{query}.
#' @param classes Optional named integer vector mapping display names
#'   to class values, e.g. \code{c("non-conflict" = 0, "conflict" = 1)}.
#'   If omitted, classes are inferred from the session.
#' @return An integer vector of labels (one per query row), or
#'   \code{NULL} if the user cancels.
#' @export
#' @examples
#' \dontrun{
#' labels  <- conflibert_active_label(session)
#' session <- conflibert_active_next(session, labels)
#' }
conflibert_active_label <- function(session, classes = NULL) {
  stopifnot(inherits(session, "conflibert_al_session"))
  if (isTRUE(session$done) || nrow(session$query) == 0L) {
    stop("No samples to label in this session.", call. = FALSE)
  }
  if (!requireNamespace("shiny", quietly = TRUE) ||
      !requireNamespace("miniUI", quietly = TRUE)) {
    stop(
      "The interactive labeler needs the shiny and miniUI packages:\n",
      "  install.packages(c(\"shiny\", \"miniUI\"))",
      call. = FALSE
    )
  }

  nl <- session$.state$num_labels
  if (is.null(classes)) {
    classes <- stats::setNames(seq_len(nl) - 1L, as.character(seq_len(nl) - 1L))
  }
  if (anyDuplicated(as.integer(classes)) ||
      any(is.na(match(seq_len(nl) - 1L, as.integer(classes))))) {
    stop(sprintf("`classes` must cover integers 0..%d.", nl - 1L), call. = FALSE)
  }
  class_values <- as.character(as.integer(classes))
  class_labels <- names(classes)

  query <- session$query
  n <- nrow(query)
  is_binary <- session$.state$params$task == "binary"

  ui <- miniUI::miniPage(
    miniUI::gadgetTitleBar(
      sprintf("Label %d samples (round %d)", n, session$round),
      right = miniUI::miniTitleBarButton("done", "Submit", primary = TRUE)
    ),
    shiny::tags$head(shiny::tags$style(shiny::HTML("
      .al-row { padding: 10px 14px; border-bottom: 1px solid #e5e7eb; }
      .al-row:hover { background: #f9fafb; }
      .al-idx { color: #9ca3af; font-weight: 600; min-width: 32px; }
      .al-text { font-size: 14px; line-height: 1.4; }
      .al-meta { color: #9ca3af; font-size: 11px; margin-top: 2px; }
      .al-progress { padding: 8px 14px; background: #f3f4f6;
                     border-bottom: 1px solid #e5e7eb;
                     font-size: 13px; color: #374151; position: sticky;
                     top: 0; z-index: 10; }
      .shiny-options-group { margin-top: 4px; }
    "))),
    miniUI::miniContentPanel(
      padding = 0,
      shiny::div(class = "al-progress", shiny::textOutput("progress", inline = TRUE)),
      shiny::uiOutput("rows")
    )
  )

  server <- function(input, output, sess) {
    get_labels <- function() {
      vapply(seq_len(n), function(i) {
        v <- input[[paste0("lbl_", i)]]
        if (is.null(v) || !nzchar(v)) NA_integer_ else as.integer(v)
      }, integer(1))
    }

    output$progress <- shiny::renderText({
      labs <- get_labels()
      done_n <- sum(!is.na(labs))
      sprintf("Labeled %d of %d. Click Submit when every row has a choice.",
              done_n, n)
    })

    output$rows <- shiny::renderUI({
      rows <- lapply(seq_len(n), function(i) {
        shiny::div(
          class = "al-row",
          shiny::fluidRow(
            shiny::column(
              8,
              shiny::div(
                style = "display: flex; gap: 10px;",
                shiny::span(class = "al-idx", paste0(i, ".")),
                shiny::div(
                  shiny::div(class = "al-text", query$text[i]),
                  shiny::div(class = "al-meta",
                             sprintf("uncertainty: %.3f", query$uncertainty[i]))
                )
              )
            ),
            shiny::column(
              4,
              shiny::radioButtons(
                inputId = paste0("lbl_", i),
                label = NULL,
                choiceNames = class_labels,
                choiceValues = class_values,
                selected = character(0),
                inline = is_binary
              )
            )
          )
        )
      })
      do.call(shiny::tagList, rows)
    })

    shiny::observeEvent(input$done, {
      labs <- get_labels()
      if (anyNA(labs)) {
        shiny::showNotification(
          sprintf("%d rows still unlabeled.", sum(is.na(labs))),
          type = "warning", duration = 3
        )
        return()
      }
      shiny::stopApp(labs)
    })

    shiny::observeEvent(input$cancel, shiny::stopApp(NULL))
  }

  viewer <- if (requireNamespace("rstudioapi", quietly = TRUE) &&
                rstudioapi::isAvailable()) {
    shiny::dialogViewer("Active Learning Labeler", width = 900, height = 700)
  } else {
    shiny::browserViewer()
  }
  shiny::runGadget(ui, server, viewer = viewer)
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
  right <- if (isTRUE(x$done)) {
    sprintf("complete, %d rounds", x$round - 1L)
  } else {
    sprintf("round %d", x$round)
  }
  cli::cat_rule(
    left = cli::style_bold("Active learning session"), right = right
  )
  p <- x$.state$params
  cli::cat_line(sprintf(
    "Model: %s  |  Task: %s  |  Strategy: %s",
    cli::col_blue(p$model), cli::col_blue(p$task), cli::col_blue(p$strategy)
  ))
  cli::cat_line(sprintf(
    "Labeled: %d  |  Pool remaining: %d  |  Query size: %d",
    x$labeled_n, x$pool_n, p$query_size
  ))

  if (nrow(x$metrics) > 0L) {
    cli::cat_line()
    cli::cat_line(cli::style_bold("Metrics by round:"))
    print(tibble::as_tibble(x$metrics), n = Inf)
  }

  if (!isTRUE(x$done) && nrow(x$query) > 0L) {
    cli::cat_line()
    cli::cat_line(cli::style_bold(sprintf(
      "Current query (%d %s, showing %d):",
      nrow(x$query), .cb_plural(nrow(x$query), "sample"),
      min(n, nrow(x$query))
    )))
    width <- cli::console_width()
    show <- utils::head(x$query, n)
    for (i in seq_len(nrow(show))) {
      cat(sprintf(
        "%3d. %s %s\n", i,
        cli::col_grey(sprintf("[%.3f]", show$uncertainty[i])),
        .cb_trim(show$text[i], width - 14L)
      ))
    }
    cli::cat_line()
    cli::cat_line(cli::col_grey(
      "# label these, then: session <- conflibert_active_next(session, labels)"
    ))
    cli::cat_line(cli::col_grey(
      "# or point and click:  labels <- conflibert_active_label(session)"
    ))
  } else if (isTRUE(x$done)) {
    cli::cat_line()
    cli::cat_line(cli::col_grey(
      "# pool exhausted; save with conflibert_active_save(session, dir)"
    ))
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
  # `loss` lives on a different scale than the 0..1 score metrics; keep
  # it in the metrics tibble but leave it out of the score panel.
  metric_cols <- setdiff(names(m), c(base_cols, "loss"))
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

  show_metrics <- which %in% c("all", "metrics") && has_metrics
  show_uncert  <- which %in% c("all", "uncertainty") && has_uncert
  n_panels <- show_metrics + show_uncert
  if (n_panels == 0L) {
    message("Nothing to plot.")
    return(invisible(x))
  }

  oldpar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldpar), add = TRUE)
  graphics::par(
    family   = "sans",
    bg       = "white",
    col.axis = "#475569",
    col.lab  = "#1f2937",
    col.main = "#0f172a",
    cex.main = 1.15, font.main = 2,
    cex.lab  = 0.95,
    cex.axis = 0.85,
    tcl = -0.25, mgp = c(2.5, 0.5, 0), las = 1
  )
  if (n_panels == 2L) {
    graphics::par(mfrow = c(2, 1), mar = c(4.2, 4.4, 3.2, 1.5))
  } else {
    graphics::par(mar = c(4.2, 4.4, 3.2, 1.5))
  }

  pal <- c("#0ea5e9", "#f59e0b", "#10b981", "#8b5cf6",
           "#ef4444", "#64748b")
  grid_col <- "#eef1f6"

  panel_frame <- function(xs, ylim, title, xlab, ylab) {
    graphics::plot.new()
    graphics::plot.window(xlim = range(xs), ylim = ylim)
    yt <- pretty(ylim, n = 5)
    graphics::abline(h = yt, col = grid_col, lwd = 1)
    graphics::axis(1, col = NA, col.ticks = NA)
    graphics::axis(2, at = yt, col = NA, col.ticks = NA)
    graphics::title(main = title, adj = 0, line = 1.4)
    graphics::title(xlab = xlab, line = 2.4)
    graphics::title(ylab = ylab, line = 2.8)
  }

  xs <- m$train_size

  if (show_metrics) {
    panel_frame(xs, ylim = c(0, 1),
                title = "Learning curve",
                xlab = "Labeled examples",
                ylab = "Score")
    for (i in seq_along(metric_cols)) {
      vals <- m[[metric_cols[i]]]
      col <- pal[((i - 1L) %% length(pal)) + 1L]
      graphics::lines(xs, vals, col = col, lwd = 2.4)
      graphics::points(xs, vals, pch = 19, col = col, cex = 1.2)
    }
    graphics::legend(
      "bottomright", legend = metric_cols,
      col   = pal[seq_along(metric_cols)],
      pch = 19, pt.cex = 1.1, lwd = 2,
      bty = "n", text.col = "#334155",
      cex = 0.85, inset = c(0.02, 0.02), seg.len = 1.6
    )
  }

  if (show_uncert) {
    keep <- !is.na(m$uncertainty_mean)
    ux <- m$train_size[keep]
    umean <- m$uncertainty_mean[keep]
    umax  <- m$uncertainty_max[keep]
    ylim <- range(c(umean, umax), na.rm = TRUE)
    if (diff(ylim) == 0) ylim <- ylim + c(-0.01, 0.01)
    pad <- 0.08 * diff(ylim)
    ylim <- c(max(0, ylim[1] - pad), ylim[2] + pad)

    panel_frame(ux, ylim = ylim,
                title = "Query uncertainty",
                xlab = "Labeled examples",
                ylab = "Uncertainty")

    if (length(ux) >= 2L) {
      graphics::polygon(
        c(ux, rev(ux)), c(umean, rev(umax)),
        col = grDevices::adjustcolor("#0ea5e9", alpha.f = 0.14),
        border = NA
      )
    }
    graphics::lines(ux, umax,  col = "#0ea5e9", lwd = 1,   lty = 3)
    graphics::lines(ux, umean, col = "#0ea5e9", lwd = 2.4)
    graphics::points(ux, umean, pch = 19, col = "#0ea5e9", cex = 1.2)
    graphics::legend(
      "topright", legend = c("mean", "max"),
      col = "#0ea5e9", lwd = c(2.2, 1), lty = c(1, 3),
      bty = "n", text.col = "#334155",
      cex = 0.85, inset = c(0.02, 0.02), seg.len = 1.6
    )
  }

  invisible(x)
}


# ---- internal helpers -------------------------------------------------

# Evaluate `expr` with the RNG seeded to `seed`, then restore whatever
# RNG state the caller had (or remove .Random.seed if there was none).
# When `seed` is NULL, `expr` runs untouched. Used to make seeded steps
# (e.g. k-means) reproducible without clobbering the user's random stream.
.al_with_seed <- function(seed, expr) {
  if (is.null(seed)) {
    return(expr)
  }
  if (exists(".Random.seed", envir = .GlobalEnv)) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv)
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
  } else {
    on.exit(
      suppressWarnings(rm(".Random.seed", envir = .GlobalEnv)),
      add = TRUE
    )
  }
  set.seed(as.integer(seed))
  expr
}

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
  # HF Trainer reports training throughput on every evaluate(); these
  # aren't learning signals, so keep them out of the round-by-round table.
  drop <- c("runtime", "samples_per_second", "steps_per_second")
  for (k in setdiff(names(metrics), drop)) {
    row[[k]] <- as.numeric(metrics[[k]])
  }
  row$uncertainty_mean <- if (length(uncertainty)) {
    round(mean(uncertainty), 4)
  } else NA_real_
  row$uncertainty_max <- if (length(uncertainty)) {
    round(max(uncertainty), 4)
  } else NA_real_
  row
}

.al_pick_query <- function(py, trained, pool_texts, available_idx,
                            strategy, query_size, max_seq_len,
                            diverse = FALSE, diversity_candidates = NULL,
                            seed = NULL) {
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
  top_local <- if (!isTRUE(diverse) || k >= length(texts_sub)) {
    order(scores, decreasing = TRUE)[seq_len(k)]
  } else {
    .al_pick_diverse(
      py, trained, texts_sub, scores, k,
      as.integer(diversity_candidates), as.integer(max_seq_len),
      seed = seed
    )
  }

  top_idx <- available_idx[top_local]
  list(
    indices = top_idx,
    tibble = tibble::tibble(
      text = pool_texts[top_idx],
      uncertainty = round(as.numeric(scores[top_local]), 4)
    )
  )
}


.al_pick_diverse <- function(py, trained, texts_sub, scores, k,
                              diversity_candidates, max_seq_len,
                              seed = NULL) {
  n_cand <- min(max(diversity_candidates, k), length(texts_sub))
  cand_local <- order(scores, decreasing = TRUE)[seq_len(n_cand)]
  if (n_cand == k) return(cand_local)

  emb <- tryCatch(
    py$al_embed(
      model = trained$model,
      tokenizer = trained$tokenizer,
      texts = texts_sub[cand_local],
      max_seq_len = max_seq_len
    ),
    error = function(e) NULL
  )
  if (is.null(emb)) {
    warning("Embedding step failed; falling back to top-uncertainty picks.",
            call. = FALSE)
    return(cand_local[seq_len(k)])
  }
  if (!is.matrix(emb)) emb <- as.matrix(emb)
  if (nrow(emb) < k) return(cand_local[seq_len(nrow(emb))])

  # k-means uses random initialization; seed it so diverse query
  # selection is reproducible. .al_with_seed restores the caller's RNG
  # state afterwards, so the user's global random stream is untouched.
  km <- .al_with_seed(seed, tryCatch(
    stats::kmeans(emb, centers = k, nstart = 5, iter.max = 50),
    error = function(e) NULL
  ))
  if (is.null(km)) {
    warning("k-means failed; falling back to top-uncertainty picks.",
            call. = FALSE)
    return(cand_local[seq_len(k)])
  }

  clus_local <- vapply(seq_len(k), function(j) {
    idx <- which(km$cluster == j)
    if (length(idx) == 0L) return(NA_integer_)
    idx[which.max(scores[cand_local[idx]])]
  }, integer(1))

  # replace any empty-cluster NAs with the next-best top-scorers
  if (anyNA(clus_local)) {
    used <- clus_local[!is.na(clus_local)]
    fill <- setdiff(seq_along(cand_local), used)[seq_len(sum(is.na(clus_local)))]
    clus_local[is.na(clus_local)] <- fill
  }
  cand_local[clus_local]
}
