#' Binary Classification
#'
#' Classify text as conflict-related (Positive) or not (Negative) using
#' the pretrained ConfliBERT binary classifier. Inference is batched, so
#' long vectors are fast; a progress bar appears for large inputs.
#'
#' @param text A character vector of one or more texts.
#' @return A tibble (with a themed print method) with columns:
#'   \describe{
#'     \item{text}{The input text.}
#'     \item{label}{"Positive" or "Negative".}
#'     \item{class}{Integer (1 = positive, 0 = negative).}
#'     \item{confidence}{Probability of the predicted class.}
#'     \item{prob_negative}{Probability of the negative class.}
#'     \item{prob_positive}{Probability of the positive class.}
#'   }
#' @export
#' @examplesIf conflibert_available()
#' conflibert_classify("A bomb exploded in the market.")
#'
#' conflibert_classify(c(
#'   "Government troops clashed with rebels.",
#'   "The weather was sunny and warm."
#' ))
conflibert_classify <- function(text) {
  stopifnot(is.character(text), length(text) >= 1)
  py <- .get_py()
  .cb_announce_load("classify", "binary classification")

  chunks <- .cb_chunked(
    text, function(ts) py$classify_batch(as.list(ts)), "Classifying"
  )
  out <- tibble::tibble(
    text          = text,
    label         = as.character(.cb_col(chunks, "label")),
    class         = as.integer(.cb_col(chunks, "class")),
    confidence    = as.numeric(.cb_col(chunks, "confidence")),
    prob_negative = as.numeric(.cb_col(chunks, "prob_negative")),
    prob_positive = as.numeric(.cb_col(chunks, "prob_positive"))
  )
  .cb_result(out, "conflibert_classify")
}


#' Multilabel Classification
#'
#' Score text against four event categories: Armed Assault,
#' Bombing or Explosion, Kidnapping, and Other. Each category is
#' scored independently. Inference is batched.
#'
#' @param text A character vector of one or more texts.
#' @return A tibble (with themed print and plot methods) with columns:
#'   \describe{
#'     \item{doc_id}{Integer index of the input text
#'       (only when \code{length(text) > 1}).}
#'     \item{text}{The input text.}
#'     \item{label}{Event category name.}
#'     \item{probability}{Score between 0 and 1.}
#'     \item{predicted}{Logical, TRUE if probability >= 0.5.}
#'   }
#' @export
#' @examplesIf conflibert_available()
#' conflibert_multilabel("Insurgents kidnapped two aid workers near the border.")
conflibert_multilabel <- function(text) {
  stopifnot(is.character(text), length(text) >= 1)
  py <- .get_py()
  .cb_announce_load("multilabel", "multilabel classification")

  chunks <- .cb_chunked(
    text, function(ts) py$multilabel_batch(as.list(ts)), "Scoring"
  )
  categories <- as.character(chunks[[1]]$categories)
  probs <- do.call(rbind, lapply(chunks, function(r) {
    p <- r$probabilities
    if (is.matrix(p)) p else do.call(rbind, lapply(p, as.numeric))
  }))

  k <- length(categories)
  n <- length(text)
  prob_vec <- as.numeric(t(probs))
  result <- tibble::tibble(
    doc_id      = rep(seq_len(n), each = k),
    text        = rep(text, each = k),
    label       = rep(categories, times = n),
    probability = prob_vec,
    predicted   = prob_vec >= 0.5
  )
  if (length(text) == 1) result$doc_id <- NULL
  .cb_result(result, "conflibert_multilabel")
}
