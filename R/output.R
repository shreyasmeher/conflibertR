# Shared output helpers: palette, themed printing, progress feedback.

# Package palette, used by both base-graphics and ggplot2 plots so every
# visual the package produces looks like it came from the same place.
.cb_pal <- c(
  blue   = "#0ea5e9",
  amber  = "#f59e0b",
  green  = "#10b981",
  purple = "#8b5cf6",
  red    = "#ef4444",
  slate  = "#64748b",
  ink    = "#0f172a",
  muted  = "#94a3b8"
)

.cb_series <- unname(.cb_pal[c("blue", "amber", "green", "purple", "red", "slate")])

# cli styling for NER entity types
.cb_entity_style <- function(label) {
  switch(label,
    "Person"       = cli::col_blue,
    "Organisation" = cli::col_magenta,
    "Organization" = cli::col_magenta,
    "Location"     = cli::col_green,
    "Weapon"       = cli::col_red,
    "Temporal"     = cli::col_cyan,
    cli::col_yellow
  )
}

# A small horizontal bar for probabilities in console output.
.cb_bar <- function(p, width = 10L) {
  p <- max(0, min(1, p))
  filled <- round(p * width)
  paste0(
    cli::col_silver(strrep("\u2588", filled)),
    cli::col_grey(strrep("\u2591", width - filled))
  )
}

.cb_trim <- function(x, width) {
  cli::ansi_strtrim(x, width = max(20L, width))
}

# Announce a first-time model load (the HuggingFace download can take a
# while and used to happen in complete silence).
.cb_announce_load <- function(task, label) {
  loaded <- tryCatch(
    isTRUE(.get_py()$is_loaded(task)),
    error = function(e) TRUE
  )
  if (!loaded) {
    cli::cli_alert_info(
      "Loading the {label} model (first call downloads it from HuggingFace)."
    )
  }
  invisible(NULL)
}

# Run `fn` over `texts` in chunks with a progress bar when the input is
# large. Returns a list of per-chunk results for the caller to combine.
.cb_chunked <- function(texts, fn, label, chunk_size = 64L) {
  idx <- split(seq_along(texts), ceiling(seq_along(texts) / chunk_size))
  if (length(idx) <= 1L) return(list(fn(texts)))
  out <- vector("list", length(idx))
  cli::cli_progress_bar(label, total = length(texts))
  for (i in seq_along(idx)) {
    out[[i]] <- fn(texts[idx[[i]]])
    cli::cli_progress_update(inc = length(idx[[i]]))
  }
  cli::cli_progress_done()
  out
}

# Flatten one named column out of a list of per-chunk columnar results.
.cb_col <- function(chunks, name) {
  unlist(lapply(chunks, function(r) r[[name]]), use.names = FALSE)
}

# Subclass a tibble so it gets a themed print method while remaining a
# perfectly ordinary tibble for everything else.
.cb_result <- function(df, subclass) {
  class(df) <- c(subclass, class(df))
  df
}

.cb_unclass <- function(x) {
  class(x) <- setdiff(class(x), c(
    "conflibert_classify", "conflibert_ner",
    "conflibert_multilabel", "conflibert_comparison"
  ))
  x
}


# ---- print methods ------------------------------------------------------
# All output goes through cli::cat_*() so it lands on stdout, where
# print() output belongs.

.cb_plural <- function(n, singular, plural = paste0(singular, "s")) {
  if (n == 1) singular else plural
}

.cb_footer <- function(x) {
  cli::cat_line(cli::col_grey(sprintf(
    "# A tibble underneath: columns %s", paste(names(x), collapse = ", ")
  )))
}

#' @export
print.conflibert_classify <- function(x, n = 10, ...) {
  total <- nrow(x)
  cli::cat_rule(
    left = cli::style_bold("ConfliBERT classification"),
    right = sprintf("%d %s", total, .cb_plural(total, "text"))
  )
  width <- cli::console_width()
  show <- utils::head(x, n)
  for (i in seq_len(nrow(show))) {
    lab <- show$label[i]
    badge <- if (identical(lab, "Positive")) {
      cli::col_red(cli::style_bold("Positive"))
    } else {
      cli::col_grey("Negative")
    }
    cat(sprintf("%3d. %s\n", i, .cb_trim(show$text[i], width - 6L)))
    cat(sprintf(
      "     %s %s %.2f\n",
      cli::ansi_align(badge, 9), .cb_bar(show$confidence[i]),
      show$confidence[i]
    ))
  }
  if (total > nrow(show)) {
    cli::cat_line(cli::col_grey(sprintf(
      "# ... %d more %s", total - nrow(show),
      .cb_plural(total - nrow(show), "row")
    )))
  }
  .cb_footer(x)
  invisible(x)
}

#' @export
print.conflibert_ner <- function(x, n = 10, ...) {
  total_docs <- if ("doc_id" %in% names(x)) length(unique(x$doc_id)) else 1L
  cli::cat_rule(
    left = cli::style_bold("ConfliBERT entities"),
    right = sprintf(
      "%d %s in %d %s",
      nrow(x), .cb_plural(nrow(x), "entity", "entities"),
      total_docs, .cb_plural(total_docs, "text")
    )
  )
  if (nrow(x) == 0L) {
    cli::cat_line(cli::col_grey("No entities found."))
    return(invisible(x))
  }

  texts <- attr(x, "texts", exact = TRUE)
  doc_ids <- if ("doc_id" %in% names(x)) x$doc_id else rep(1L, nrow(x))
  shown_docs <- utils::head(unique(doc_ids), n)
  width <- cli::console_width()

  for (d in shown_docs) {
    rows <- which(doc_ids == d)
    # highlighted source text when we have offsets
    if (!is.null(texts) && d <= length(texts) &&
        all(c("start", "end") %in% names(x))) {
      txt <- texts[[d]]
      ord <- rows[order(x$start[rows], decreasing = TRUE)]
      for (r in ord) {
        s <- x$start[r]; e <- x$end[r]
        if (is.na(s) || is.na(e) || s < 1 || e > nchar(txt)) next
        styled <- .cb_entity_style(x$label[r])(
          cli::style_underline(substr(txt, s, e))
        )
        txt <- paste0(substr(txt, 1, s - 1), styled,
                      substr(txt, e + 1, nchar(txt)))
      }
      cat(sprintf("%3d. %s\n", d, .cb_trim(txt, width - 6L)))
    } else {
      cat(sprintf("%3d.\n", d))
    }
    for (r in rows) {
      sc <- if ("score" %in% names(x)) sprintf("  %.2f", x$score[r]) else ""
      cat(sprintf(
        "     %s %s%s\n",
        cli::ansi_align(.cb_entity_style(x$label[r])(x$label[r]), 14),
        x$entity[r],
        cli::col_grey(sc)
      ))
    }
  }
  if (length(unique(doc_ids)) > length(shown_docs)) {
    extra <- length(unique(doc_ids)) - length(shown_docs)
    cli::cat_line(cli::col_grey(sprintf(
      "# ... %d more %s", extra, .cb_plural(extra, "text")
    )))
  }
  .cb_footer(x)
  invisible(x)
}

#' @export
print.conflibert_multilabel <- function(x, n = 5, ...) {
  doc_ids <- if ("doc_id" %in% names(x)) x$doc_id else rep(1L, nrow(x))
  docs <- unique(doc_ids)
  cli::cat_rule(
    left = cli::style_bold("ConfliBERT event types"),
    right = sprintf("%d %s", length(docs), .cb_plural(length(docs), "text"))
  )
  width <- cli::console_width()
  shown <- utils::head(docs, n)
  for (d in shown) {
    rows <- which(doc_ids == d)
    cat(sprintf("%3d. %s\n", d, .cb_trim(x$text[rows[1]], width - 6L)))
    for (r in rows) {
      mark <- if (isTRUE(x$predicted[r])) cli::col_green("\u2713") else " "
      lab <- if (isTRUE(x$predicted[r])) {
        cli::style_bold(x$label[r])
      } else {
        cli::col_grey(x$label[r])
      }
      cat(sprintf(
        "     %s %s %s %.2f\n",
        mark, cli::ansi_align(lab, 21), .cb_bar(x$probability[r]),
        x$probability[r]
      ))
    }
  }
  if (length(docs) > length(shown)) {
    extra <- length(docs) - length(shown)
    cli::cat_line(cli::col_grey(sprintf(
      "# ... %d more %s", extra, .cb_plural(extra, "text")
    )))
  }
  .cb_footer(x)
  invisible(x)
}

#' @export
print.conflibert_comparison <- function(x, ...) {
  cli::cat_rule(
    left = cli::style_bold("Model comparison"),
    right = sprintf("%d %s", nrow(x), .cb_plural(nrow(x), "model"))
  )
  key <- intersect(c("f1", "f1_macro", "accuracy"), names(x))[1]
  if (!is.na(key) && nrow(x) > 0L) {
    ord <- order(x[[key]], decreasing = TRUE)
    best <- ord[1]
    metric_cols <- setdiff(
      names(x)[vapply(.cb_unclass(x), is.numeric, logical(1))], "runtime"
    )
    for (i in ord) {
      star <- if (i == best) cli::col_yellow("\u2605") else " "
      name <- if (i == best) cli::style_bold(x$model[i]) else x$model[i]
      vals <- vapply(metric_cols, function(mc) {
        sprintf("%s %.3f", mc, x[[mc]][i])
      }, character(1))
      rt <- if ("runtime" %in% names(x) && !is.na(x$runtime[i])) {
        cli::col_grey(sprintf("  (%.0fs)", x$runtime[i]))
      } else ""
      cat(sprintf(
        " %s %s\n     %s%s\n",
        star, cli::ansi_align(name, 24),
        cli::col_grey(paste(vals, collapse = "   ")), rt
      ))
    }
    cli::cat_line(cli::col_grey(sprintf(
      "# ranked by %s; tibble underneath with columns %s",
      key, paste(names(x), collapse = ", ")
    )))
  } else {
    print(tibble::as_tibble(.cb_unclass(x)))
  }
  invisible(x)
}

#' @export
print.conflibert_finetune <- function(x, ...) {
  cli::cat_rule(left = cli::style_bold("Fine-tuned model"))
  cli::cat_line(sprintf(
    "Model: %s  |  Task: %s",
    cli::col_blue(x$model %||% "unknown"), cli::col_blue(x$task %||% "unknown")
  ))
  if (!is.null(x$runtime)) {
    cli::cat_line(sprintf("Training time: %.1fs", x$runtime))
  }
  if (!is.null(x$model_dir)) {
    cli::cat_line("Checkpoint: ", cli::col_blue(x$model_dir))
  }
  if (!is.null(x$metrics) && nrow(x$metrics) > 0L) {
    cli::cat_line(cli::style_bold("Test metrics:"))
    m <- x$metrics
    num <- names(m)[vapply(m, is.numeric, logical(1))]
    num <- setdiff(num, "loss")
    for (k in num) {
      cat(sprintf(
        "  %s %s %.3f\n",
        cli::ansi_align(k, 18), .cb_bar(m[[k]][1], 16), m[[k]][1]
      ))
    }
    if ("loss" %in% names(m)) {
      cat(sprintf("  %s %.4f\n", cli::ansi_align("loss", 18), m$loss[1]))
    }
  }
  cli::cat_line(cli::col_grey(
    "# plot() for the confusion matrix; $metrics, $predictions, $probabilities as before"
  ))
  invisible(x)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
