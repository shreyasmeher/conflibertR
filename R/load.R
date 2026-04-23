#' Load a Fine-tuned Classifier
#'
#' Load a classifier saved by \code{\link{conflibert_finetune}}
#' (when \code{save_dir} was specified) or
#' \code{\link{conflibert_active_save}}. Returns a reusable classifier
#' object you can pass to \code{\link[=predict.conflibert_classifier]{predict}}.
#'
#' @param dir Directory containing a HuggingFace checkpoint
#'   (\code{config.json}, model weights, tokenizer files).
#' @return An object of class \code{"conflibert_classifier"}: a list
#'   with \code{model}, \code{tokenizer}, \code{num_labels}, and
#'   \code{dir}.
#' @export
#' @examples
#' \dontrun{
#' # Save during fine-tuning
#' conflibert_finetune(train, dev, test, save_dir = "my_model")
#'
#' # Reload later and predict
#' clf <- conflibert_load("my_model")
#' predict(clf, c("Troops advanced on the capital.", "Nice weather."))
#' }
conflibert_load <- function(dir) {
  dir <- path.expand(dir)
  if (!dir.exists(dir)) {
    stop("Directory does not exist: ", dir, call. = FALSE)
  }
  py <- .get_py()
  loaded <- py$load_classifier(dir)
  obj <- list(
    model      = loaded$model,
    tokenizer  = loaded$tokenizer,
    num_labels = as.integer(loaded$num_labels),
    dir        = dir
  )
  class(obj) <- "conflibert_classifier"
  obj
}


#' Predict with a Loaded Classifier
#'
#' Run batched inference with a classifier loaded via
#' \code{\link{conflibert_load}}. Returns a tibble with predicted class,
#' confidence, and one \code{prob_*} column per class.
#'
#' @param object A \code{conflibert_classifier}.
#' @param text Character vector of texts to classify.
#' @param batch_size Inference batch size. Default: 32.
#' @param max_seq_len Max token sequence length. Default: 512.
#' @param ... Ignored.
#' @return A tibble with \code{text}, \code{class}, \code{confidence},
#'   and \code{prob_0..prob_{K-1}} columns.
#' @export
#' @examples
#' \dontrun{
#' clf <- conflibert_load("my_model")
#' predict(clf, c("Bomb exploded.", "Stock market rose."))
#' }
predict.conflibert_classifier <- function(
    object, text, batch_size = 32, max_seq_len = 512, ...
) {
  text <- as.character(text)
  if (length(text) == 0L) {
    stop("`text` is empty.", call. = FALSE)
  }
  py <- .get_py()
  r <- py$predict_classifier(
    model       = object$model,
    tokenizer   = object$tokenizer,
    texts       = text,
    max_seq_len = as.integer(max_seq_len),
    batch_size  = as.integer(batch_size)
  )
  probs <- do.call(rbind, r$probabilities)
  preds <- as.integer(unlist(r$predictions))
  out <- tibble::tibble(
    text       = text,
    class      = preds,
    confidence = vapply(seq_len(nrow(probs)),
                        function(i) max(probs[i, ]), numeric(1))
  )
  for (j in seq_len(ncol(probs))) {
    out[[paste0("prob_", j - 1L)]] <- probs[, j]
  }
  out
}


#' @export
print.conflibert_classifier <- function(x, ...) {
  cat("conflibert_classifier\n")
  cat("  Source:  ", x$dir, "\n", sep = "")
  cat("  Classes: ", x$num_labels, "\n", sep = "")
  cat("\nUse predict(model, text) to classify new texts.\n")
  invisible(x)
}
