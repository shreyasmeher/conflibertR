# Plot methods: a shared ggplot2 theme + autoplot() methods for every
# result object, with base-graphics plot() fallbacks that share the same
# palette so everything the package draws looks consistent.

utils::globalVariables(c(".data", "panel", "doc"))

.cb_require_ggplot <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "This plot needs the ggplot2 package:\n",
      "  install.packages(\"ggplot2\")",
      call. = FALSE
    )
  }
}

#' ConfliBERT ggplot2 Theme
#'
#' A modern, flat ggplot2 theme used by all of the package's
#' \code{autoplot()} methods: no tick marks, no panel box, hairline
#' grid lines only along the data axis, bold left-aligned titles, and
#' generous whitespace. Use it on your own plots to match.
#'
#' @param base_size Base font size. Default: 12.
#' @param base_family Base font family. Default: system sans.
#' @param grid Which grid lines to keep: \code{"xy"} (default),
#'   \code{"x"} (vertical only, good for horizontal bars), \code{"y"}
#'   (horizontal only, good for line/column charts), or \code{"none"}.
#' @return A ggplot2 theme object.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   ggplot2::ggplot(mtcars, ggplot2::aes(wt, mpg)) +
#'     ggplot2::geom_point(colour = "#0ea5e9", size = 3) +
#'     theme_conflibert(grid = "y")
#' }
#' }
theme_conflibert <- function(base_size = 12, base_family = "",
                             grid = c("xy", "x", "y", "none")) {
  .cb_require_ggplot()
  grid <- match.arg(grid)
  grid_line <- ggplot2::element_line(colour = "#eef1f6", linewidth = 0.5)

  th <- ggplot2::theme_minimal(
    base_size = base_size, base_family = base_family
  ) +
    ggplot2::theme(
      text = ggplot2::element_text(colour = "#334155"),
      plot.title = ggplot2::element_text(
        face = "bold", size = base_size * 1.3, colour = .cb_pal[["ink"]],
        margin = ggplot2::margin(b = 4)
      ),
      plot.subtitle = ggplot2::element_text(
        colour = .cb_pal[["muted"]], size = base_size * 0.95,
        margin = ggplot2::margin(b = 14)
      ),
      plot.title.position = "plot",
      plot.caption = ggplot2::element_text(
        colour = .cb_pal[["muted"]], hjust = 0
      ),
      plot.caption.position = "plot",
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = if (grid %in% c("xy", "x")) {
        grid_line
      } else {
        ggplot2::element_blank()
      },
      panel.grid.major.y = if (grid %in% c("xy", "y")) {
        grid_line
      } else {
        ggplot2::element_blank()
      },
      axis.ticks = ggplot2::element_blank(),
      axis.title = ggplot2::element_text(
        colour = "#64748b", size = base_size * 0.9
      ),
      axis.text = ggplot2::element_text(
        colour = "#64748b", size = base_size * 0.85
      ),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        face = "bold", colour = .cb_pal[["ink"]], hjust = 0,
        size = base_size * 0.95,
        margin = ggplot2::margin(t = 10, b = 4)
      ),
      panel.spacing = ggplot2::unit(14, "pt"),
      legend.position = "bottom",
      legend.justification = "left",
      legend.title = ggplot2::element_blank(),
      legend.key.size = ggplot2::unit(12, "pt"),
      plot.margin = ggplot2::margin(16, 18, 12, 16)
    )
  th
}


# ---- autoplot methods ---------------------------------------------------

#' Plot Active Learning Progress with ggplot2
#'
#' ggplot2 version of \code{\link{plot.conflibert_al_session}}: the
#' learning curve and the query-uncertainty trend across rounds.
#'
#' @param object A \code{conflibert_al_session}.
#' @param which \code{"all"} (default), \code{"metrics"}, or
#'   \code{"uncertainty"}.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_al_session <- function(
    object, which = c("all", "metrics", "uncertainty"), ...
) {
  .cb_require_ggplot()
  which <- match.arg(which)
  m <- object$metrics
  if (nrow(m) == 0L) stop("No metrics to plot yet.", call. = FALSE)

  base_cols <- c("round", "train_size", "uncertainty_mean", "uncertainty_max")
  metric_cols <- setdiff(names(m), c(base_cols, "loss"))

  parts <- list()
  if (which %in% c("all", "metrics") && length(metric_cols) > 0L) {
    for (mc in metric_cols) {
      if (all(is.na(m[[mc]]))) next
      parts[[length(parts) + 1L]] <- data.frame(
        panel = "Learning curve", x = m$train_size,
        series = mc, y = m[[mc]]
      )
    }
  }
  if (which %in% c("all", "uncertainty") && any(!is.na(m$uncertainty_mean))) {
    parts[[length(parts) + 1L]] <- data.frame(
      panel = "Query uncertainty", x = m$train_size,
      series = "mean", y = m$uncertainty_mean
    )
    parts[[length(parts) + 1L]] <- data.frame(
      panel = "Query uncertainty", x = m$train_size,
      series = "max", y = m$uncertainty_max
    )
  }
  if (length(parts) == 0L) stop("Nothing to plot.", call. = FALSE)
  df <- do.call(rbind, parts)
  df <- df[!is.na(df$y), ]

  ggplot2::ggplot(df, ggplot2::aes(.data$x, .data$y, colour = .data$series)) +
    ggplot2::geom_line(linewidth = 1.1) +
    ggplot2::geom_point(size = 2.6) +
    ggplot2::facet_wrap(~panel, ncol = 1, scales = "free_y") +
    ggplot2::scale_colour_manual(values = .cb_series) +
    ggplot2::scale_x_continuous(
      breaks = function(lims) unique(round(pretty(lims)))
    ) +
    ggplot2::labs(
      title = "Active learning progress",
      subtitle = sprintf(
        "%d round%s, %d labeled examples",
        max(m$round), if (max(m$round) == 1) "" else "s",
        max(m$train_size)
      ),
      x = "Labeled examples", y = NULL
    ) +
    theme_conflibert(grid = "y")
}

#' Plot Classification Results with ggplot2
#'
#' Aggregated view of a \code{\link{conflibert_classify}} result: how
#' many texts landed in each class, annotated with the average
#' confidence.
#'
#' @param object A result from \code{\link{conflibert_classify}}.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_classify <- function(object, ...) {
  .cb_require_ggplot()
  df <- as.data.frame(.cb_unclass(object))
  agg <- do.call(rbind, lapply(split(df, df$label), function(g) {
    data.frame(label = g$label[1], n = nrow(g),
               confidence = mean(g$confidence))
  }))
  agg$label <- factor(agg$label, levels = c("Negative", "Positive"))
  agg$note <- sprintf("%d  (avg conf %.2f)", agg$n, agg$confidence)

  ggplot2::ggplot(agg, ggplot2::aes(.data$n, .data$label,
                                    fill = .data$label)) +
    ggplot2::geom_col(width = 0.55) +
    ggplot2::geom_text(
      ggplot2::aes(label = .data$note), hjust = -0.08,
      colour = "#475569", size = 3.4
    ) +
    ggplot2::scale_fill_manual(values = c(
      "Negative" = "#cbd5e1", "Positive" = .cb_pal[["red"]]
    ), guide = "none") +
    ggplot2::scale_x_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.25))
    ) +
    ggplot2::labs(
      title = "Classification results",
      subtitle = sprintf("%d texts, %d flagged as conflict-related",
                         nrow(df), sum(df$class == 1L)),
      x = "Texts", y = NULL
    ) +
    theme_conflibert(grid = "x")
}

#' Plot NER Results with ggplot2
#'
#' Aggregated view of a \code{\link{conflibert_ner}} result:
#' \code{type = "types"} (default) counts entities per entity type;
#' \code{type = "entities"} shows the most frequent individual
#' entities, colored by type.
#'
#' @param object A result from \code{\link{conflibert_ner}}.
#' @param type \code{"types"} or \code{"entities"}.
#' @param top_n How many entities to show when
#'   \code{type = "entities"}. Default: 12.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_ner <- function(object, type = c("types", "entities"),
                                    top_n = 12, ...) {
  .cb_require_ggplot()
  type <- match.arg(type)
  df <- as.data.frame(.cb_unclass(object))
  if (nrow(df) == 0L) stop("No entities to plot.", call. = FALSE)
  n_docs <- if ("doc_id" %in% names(df)) length(unique(df$doc_id)) else 1L

  if (type == "types") {
    tab <- as.data.frame(table(label = df$label), responseName = "n")
    tab <- tab[order(tab$n), ]
    tab$label <- factor(tab$label, levels = tab$label)
    p <- ggplot2::ggplot(tab, ggplot2::aes(.data$n, .data$label,
                                           fill = .data$label)) +
      ggplot2::geom_col(width = 0.6) +
      ggplot2::geom_text(
        ggplot2::aes(label = .data$n), hjust = -0.3,
        colour = "#475569", size = 3.4
      ) +
      ggplot2::scale_fill_manual(
        values = .cb_entity_fill(levels(tab$label)), guide = "none"
      ) +
      ggplot2::labs(
        title = "Entities by type",
        subtitle = sprintf("%d entities across %d text%s", nrow(df),
                           n_docs, if (n_docs == 1) "" else "s"),
        x = "Entities", y = NULL
      )
  } else {
    tab <- as.data.frame(
      table(entity = df$entity, label = df$label), responseName = "n"
    )
    tab <- tab[tab$n > 0, ]
    tab <- tab[order(-tab$n), ]
    tab <- utils::head(tab, top_n)
    tab <- tab[order(tab$n), ]
    tab$entity <- factor(tab$entity, levels = unique(tab$entity))
    p <- ggplot2::ggplot(tab, ggplot2::aes(.data$n, .data$entity,
                                           fill = .data$label)) +
      ggplot2::geom_col(width = 0.6) +
      ggplot2::scale_fill_manual(values = .cb_entity_fill(
        sort(unique(as.character(tab$label)))
      )) +
      ggplot2::labs(
        title = "Most frequent entities",
        subtitle = sprintf("top %d of %d entities", nrow(tab),
                           length(unique(df$entity))),
        x = "Mentions", y = NULL
      )
  }
  p +
    ggplot2::scale_x_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15)),
      breaks = function(lims) unique(floor(pretty(lims)))
    ) +
    theme_conflibert(grid = "x")
}

#' Plot Multilabel Results with ggplot2
#'
#' Plots a \code{\link{conflibert_multilabel}} result. With several
#' texts it aggregates: the share of texts flagged for each event
#' category. With a single text it shows that text's category
#' probabilities against the 0.5 decision threshold.
#'
#' @param object A result from \code{\link{conflibert_multilabel}}.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_multilabel <- function(object, ...) {
  .cb_require_ggplot()
  df <- as.data.frame(.cb_unclass(object))
  if (!"doc_id" %in% names(df)) df$doc_id <- 1L
  n_docs <- length(unique(df$doc_id))

  if (n_docs > 1L) {
    agg <- do.call(rbind, lapply(split(df, df$label), function(g) {
      data.frame(label = g$label[1], n = sum(g$predicted),
                 share = mean(g$predicted))
    }))
    agg <- agg[order(agg$share), ]
    agg$label <- factor(agg$label, levels = agg$label)
    return(
      ggplot2::ggplot(agg, ggplot2::aes(.data$share, .data$label)) +
        ggplot2::geom_col(width = 0.55, fill = .cb_pal[["blue"]]) +
        ggplot2::geom_text(
          ggplot2::aes(label = sprintf("%d of %d", .data$n, n_docs)),
          hjust = -0.12, colour = "#475569", size = 3.4
        ) +
        ggplot2::scale_x_continuous(
          labels = function(x) paste0(round(100 * x), "%"),
          expand = ggplot2::expansion(mult = c(0, 0.2))
        ) +
        ggplot2::labs(
          title = "Event types across texts",
          subtitle = sprintf("share of %d texts flagged per category",
                             n_docs),
          x = "Texts flagged", y = NULL
        ) +
        theme_conflibert(grid = "x")
    )
  }

  df <- df[order(df$probability), ]
  df$label <- factor(df$label, levels = df$label)
  ggplot2::ggplot(
    df,
    ggplot2::aes(.data$probability, .data$label, fill = .data$predicted)
  ) +
    ggplot2::geom_col(width = 0.55) +
    ggplot2::geom_vline(
      xintercept = 0.5, linetype = "dashed", colour = .cb_pal[["muted"]]
    ) +
    ggplot2::scale_fill_manual(
      values = c("FALSE" = "#cbd5e1", "TRUE" = .cb_pal[["blue"]]),
      labels = c("FALSE" = "below threshold", "TRUE" = "predicted"),
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, 1), expand = ggplot2::expansion(mult = c(0, 0.02))
    ) +
    ggplot2::labs(
      title = "Event type probabilities",
      subtitle = .cb_clip(df$text[1], 70),
      x = "Probability", y = NULL
    ) +
    theme_conflibert(grid = "x")
}

#' Plot a Model Comparison with ggplot2
#'
#' Dot chart of every metric for every model in a
#' \code{\link{conflibert_compare}} result, models ordered by their
#' primary metric.
#'
#' @param object A result from \code{\link{conflibert_compare}}.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_comparison <- function(object, ...) {
  .cb_require_ggplot()
  df <- as.data.frame(.cb_unclass(object))
  metric_cols <- setdiff(
    names(df)[vapply(df, is.numeric, logical(1))], "runtime"
  )
  if (length(metric_cols) == 0L) {
    stop("No metric columns to plot.", call. = FALSE)
  }
  key <- intersect(c("f1", "f1_macro", "accuracy"), metric_cols)[1]
  if (is.na(key)) key <- metric_cols[1]

  long <- do.call(rbind, lapply(metric_cols, function(mc) {
    data.frame(model = df$model, metric = mc, value = df[[mc]])
  }))
  long <- long[!is.na(long$value), ]
  long$model <- factor(long$model, levels = df$model[order(df[[key]])])

  ggplot2::ggplot(
    long, ggplot2::aes(.data$value, .data$model, colour = .data$metric)
  ) +
    ggplot2::geom_line(
      ggplot2::aes(group = .data$model), colour = "#eef1f6", linewidth = 2.5
    ) +
    ggplot2::geom_point(size = 3.4) +
    ggplot2::scale_colour_manual(values = .cb_series) +
    ggplot2::labs(
      title = "Model comparison",
      subtitle = sprintf("ranked by %s", key),
      x = "Score", y = NULL
    ) +
    theme_conflibert(grid = "x")
}

#' Plot a Fine-tuning Confusion Matrix with ggplot2
#'
#' Test-set confusion matrix heatmap for a
#' \code{\link{conflibert_finetune}} result.
#'
#' @param object A \code{conflibert_finetune} result.
#' @param ... Ignored.
#' @return A ggplot object.
#' @exportS3Method ggplot2::autoplot
autoplot.conflibert_finetune <- function(object, ...) {
  .cb_require_ggplot()
  df <- .cb_confusion_df(object)
  ggplot2::ggplot(
    df, ggplot2::aes(.data$predicted, .data$true, fill = .data$n)
  ) +
    ggplot2::geom_tile(colour = "white", linewidth = 1.2) +
    ggplot2::geom_text(
      ggplot2::aes(label = .data$n), colour = .cb_pal[["ink"]],
      fontface = "bold", size = 4.2
    ) +
    ggplot2::scale_fill_gradient(
      low = "#f8fafc", high = .cb_pal[["blue"]], guide = "none"
    ) +
    ggplot2::scale_y_discrete(limits = rev(levels(df$true))) +
    ggplot2::labs(
      title = "Confusion matrix",
      subtitle = sprintf(
        "%s on the test set (n = %d)",
        object$model %||% "model", length(object$true_labels)
      ),
      x = "Predicted class", y = "True class"
    ) +
    theme_conflibert(grid = "none")
}


# ---- base-graphics plot methods ----------------------------------------

#' Plot a Fine-tuning Confusion Matrix
#'
#' Base-graphics test-set confusion matrix for a
#' \code{\link{conflibert_finetune}} result. For a ggplot2 version use
#' \code{ggplot2::autoplot()}.
#'
#' @param x A \code{conflibert_finetune} result.
#' @param ... Ignored.
#' @return The object, invisibly.
#' @export
plot.conflibert_finetune <- function(x, ...) {
  df <- .cb_confusion_df(x)
  k <- nlevels(df$true)
  old <- .cb_par(mar = c(4.2, 4.4, 3.2, 1.5))
  on.exit(graphics::par(old), add = TRUE)

  graphics::plot.new()
  graphics::plot.window(xlim = c(0.5, k + 0.5), ylim = c(0.5, k + 0.5),
                        asp = 1)
  max_n <- max(df$n, 1)
  ramp <- grDevices::colorRamp(c("#f8fafc", .cb_pal[["blue"]]))
  for (i in seq_len(nrow(df))) {
    px <- as.integer(df$predicted[i])
    py <- k + 1 - as.integer(df$true[i])
    rgbv <- ramp(df$n[i] / max_n)
    graphics::rect(
      px - 0.47, py - 0.47, px + 0.47, py + 0.47,
      col = grDevices::rgb(rgbv[1], rgbv[2], rgbv[3], maxColorValue = 255),
      border = "white", lwd = 2
    )
    graphics::text(px, py, df$n[i], font = 2, col = .cb_pal[["ink"]])
  }
  graphics::axis(1, at = seq_len(k), labels = levels(df$predicted),
                 col = NA, col.ticks = NA)
  graphics::axis(2, at = rev(seq_len(k)), labels = levels(df$true),
                 col = NA, col.ticks = NA)
  graphics::title(main = "Confusion matrix", adj = 0, line = 1.4)
  graphics::title(xlab = "Predicted class", ylab = "True class")
  invisible(x)
}

#' Plot a Model Comparison
#'
#' Base-graphics dot chart of metrics per model for a
#' \code{\link{conflibert_compare}} result. For a ggplot2 version use
#' \code{ggplot2::autoplot()}.
#'
#' @param x A \code{conflibert_comparison} tibble.
#' @param ... Ignored.
#' @return The object, invisibly.
#' @export
plot.conflibert_comparison <- function(x, ...) {
  df <- as.data.frame(.cb_unclass(x))
  metric_cols <- setdiff(
    names(df)[vapply(df, is.numeric, logical(1))], "runtime"
  )
  if (length(metric_cols) == 0L) {
    message("No metric columns to plot.")
    return(invisible(x))
  }
  key <- intersect(c("f1", "f1_macro", "accuracy"), metric_cols)[1]
  if (is.na(key)) key <- metric_cols[1]
  df <- df[order(df[[key]]), ]
  nm <- nrow(df)

  old <- .cb_par(mar = c(4.2, 11, 3.2, 1.5))
  on.exit(graphics::par(old), add = TRUE)

  vals <- unlist(df[metric_cols])
  xlim <- range(vals, na.rm = TRUE)
  pad <- max(0.05 * diff(xlim), 0.02)
  xlim <- c(max(0, xlim[1] - pad), min(1, xlim[2] + pad))

  graphics::plot.new()
  graphics::plot.window(xlim = xlim, ylim = c(0.5, nm + 0.5))
  graphics::abline(h = seq_len(nm), col = "#eef1f6", lwd = 5)
  for (j in seq_along(metric_cols)) {
    col <- .cb_series[((j - 1L) %% length(.cb_series)) + 1L]
    graphics::points(df[[metric_cols[j]]], seq_len(nm), pch = 19,
                     col = col, cex = 1.6)
  }
  graphics::axis(1, col = NA, col.ticks = NA)
  graphics::axis(2, at = seq_len(nm), labels = df$model, col = NA,
                 col.ticks = NA, las = 1)
  graphics::title(main = "Model comparison", adj = 0, line = 1.4)
  graphics::title(xlab = "Score", line = 2.4)
  graphics::legend(
    "bottomright", legend = metric_cols,
    col = .cb_series[seq_along(metric_cols)], pch = 19,
    bty = "n", text.col = "#334155", cex = 0.85,
    inset = c(0.02, 0.02)
  )
  invisible(x)
}

#' Plot Classification Results
#'
#' Base-graphics aggregated view of a \code{\link{conflibert_classify}}
#' result: texts per class with average confidence. For a ggplot2
#' version use \code{ggplot2::autoplot()}.
#'
#' @param x A \code{conflibert_classify} tibble.
#' @param ... Ignored.
#' @return The object, invisibly.
#' @export
plot.conflibert_classify <- function(x, ...) {
  df <- as.data.frame(.cb_unclass(x))
  labs <- c("Negative", "Positive")
  n <- vapply(labs, function(l) sum(df$label == l), numeric(1))
  conf <- vapply(labs, function(l) {
    if (n[[l]] > 0) mean(df$confidence[df$label == l]) else NA_real_
  }, numeric(1))
  cols <- c("#cbd5e1", .cb_pal[["red"]])

  .cb_barh(
    values = n, labels = labs, fills = cols,
    notes = ifelse(is.na(conf), "0",
                   sprintf("%d  (avg conf %.2f)", n, conf)),
    main = "Classification results",
    xlab = "Texts"
  )
  invisible(x)
}

#' Plot NER Results
#'
#' Base-graphics aggregated view of a \code{\link{conflibert_ner}}
#' result: entity counts per type. For a ggplot2 version (including a
#' top-entities view) use \code{ggplot2::autoplot()}.
#'
#' @param x A \code{conflibert_ner} tibble.
#' @param ... Ignored.
#' @return The object, invisibly.
#' @export
plot.conflibert_ner <- function(x, ...) {
  df <- as.data.frame(.cb_unclass(x))
  if (nrow(df) == 0L) {
    message("No entities to plot.")
    return(invisible(x))
  }
  tab <- sort(table(df$label))
  fills <- .cb_entity_fill(names(tab))

  .cb_barh(
    values = as.numeric(tab), labels = names(tab), fills = fills,
    notes = as.character(as.numeric(tab)),
    main = "Entities by type",
    xlab = "Entities"
  )
  invisible(x)
}

#' Plot Multilabel Results
#'
#' Base-graphics version of the multilabel plot. With several texts it
#' aggregates (share of texts flagged per category); with a single text
#' it shows that text's category probabilities. For a ggplot2 version
#' use \code{ggplot2::autoplot()}.
#'
#' @param x A \code{conflibert_multilabel} tibble.
#' @param ... Ignored.
#' @return The object, invisibly.
#' @export
plot.conflibert_multilabel <- function(x, ...) {
  df <- as.data.frame(.cb_unclass(x))
  if (!"doc_id" %in% names(df)) df$doc_id <- 1L
  n_docs <- length(unique(df$doc_id))

  if (n_docs > 1L) {
    agg <- do.call(rbind, lapply(split(df, df$label), function(g) {
      data.frame(label = g$label[1], n = sum(g$predicted),
                 share = mean(g$predicted))
    }))
    agg <- agg[order(agg$share), ]
    .cb_barh(
      values = agg$share, labels = agg$label,
      fills = rep(.cb_pal[["blue"]], nrow(agg)),
      notes = sprintf("%d of %d", agg$n, n_docs),
      main = "Event types across texts",
      xlab = "Share of texts flagged",
      xaxt_percent = TRUE
    )
    return(invisible(x))
  }

  df <- df[order(df$probability), ]
  .cb_barh(
    values = df$probability, labels = df$label,
    fills = ifelse(df$predicted, .cb_pal[["blue"]], "#cbd5e1"),
    notes = sprintf("%.2f", df$probability),
    main = "Event type probabilities",
    xlab = "Probability",
    vline = 0.5
  )
  invisible(x)
}


# ---- shared plot helpers ------------------------------------------------

.cb_par <- function(...) {
  old <- graphics::par(no.readonly = TRUE)
  graphics::par(
    family = "sans", bg = "white",
    col.axis = "#475569", col.lab = "#1f2937", col.main = .cb_pal[["ink"]],
    cex.main = 1.15, font.main = 2, cex.lab = 0.95, cex.axis = 0.85,
    tcl = -0.25, mgp = c(2.5, 0.5, 0), las = 1, ...
  )
  old
}

# flat horizontal bar chart used by the base-graphics plot methods
.cb_barh <- function(values, labels, fills, notes = NULL,
                     main = "", xlab = "", xaxt_percent = FALSE,
                     vline = NULL) {
  old <- .cb_par(mar = c(4.2, 11, 3.2, 4))
  on.exit(graphics::par(old), add = TRUE)

  k <- length(values)
  xmax <- max(values, na.rm = TRUE)
  if (!is.finite(xmax) || xmax <= 0) xmax <- 1
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, xmax * 1.18), ylim = c(0.4, k + 0.6))

  ticks <- pretty(c(0, xmax), n = 4)
  ticks <- ticks[ticks <= xmax * 1.02]
  graphics::abline(v = ticks, col = "#eef1f6", lwd = 1)
  tick_labels <- if (xaxt_percent) paste0(round(100 * ticks), "%") else ticks
  graphics::axis(1, at = ticks, labels = tick_labels,
                 col = NA, col.ticks = NA)

  if (!is.null(vline)) {
    graphics::abline(v = vline, lty = 3, col = .cb_pal[["muted"]])
  }
  yy <- seq_len(k)
  graphics::rect(0, yy - 0.28, values, yy + 0.28, col = fills, border = NA)
  graphics::axis(2, at = yy, labels = labels, col = NA, col.ticks = NA,
                 las = 1)
  if (!is.null(notes)) {
    graphics::text(values, yy, labels = notes, pos = 4,
                   col = "#475569", cex = 0.85, xpd = NA)
  }
  graphics::title(main = main, adj = 0, line = 1.4)
  graphics::title(xlab = xlab, line = 2.4)
}

# hex fills for entity types, matching the console colors
.cb_entity_fill <- function(labels) {
  map <- c(
    Person = .cb_pal[["blue"]], Organisation = .cb_pal[["purple"]],
    Organization = .cb_pal[["purple"]], Location = .cb_pal[["green"]],
    Weapon = .cb_pal[["red"]], Temporal = .cb_pal[["amber"]]
  )
  out <- map[labels]
  out[is.na(out)] <- .cb_pal[["slate"]]
  stats::setNames(unname(out), labels)
}

.cb_clip <- function(x, width) {
  ifelse(nchar(x) > width, paste0(substr(x, 1, width - 1), "\u2026"), x)
}

.cb_confusion_df <- function(object) {
  truth <- as.integer(object$true_labels)
  pred <- as.integer(object$predictions)
  if (length(truth) == 0L || length(pred) != length(truth)) {
    stop("This result has no test-set predictions to plot.", call. = FALSE)
  }
  classes <- sort(unique(c(truth, pred)))
  tab <- table(
    true = factor(truth, levels = classes),
    predicted = factor(pred, levels = classes)
  )
  df <- as.data.frame(tab, responseName = "n")
  df
}
