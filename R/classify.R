#' Binary Classification
#'
#' Classify text as conflict-related (Positive) or not (Negative) using
#' the pretrained ConfliBERT binary classifier.
#'
#' @param text A character vector of one or more texts.
#' @return A tibble with columns:
#'   \describe{
#'     \item{text}{The input text.}
#'     \item{label}{"Positive" or "Negative".}
#'     \item{class}{Integer (1 = positive, 0 = negative).}
#'     \item{confidence}{Probability of the predicted class.}
#'     \item{prob_negative}{Probability of the negative class.}
#'     \item{prob_positive}{Probability of the positive class.}
#'   }
#' @export
#' @examples
#' \dontrun{
#' conflibert_classify("A bomb exploded in the market.")
#'
#' conflibert_classify(c(
#'   "Government troops clashed with rebels.",
#'   "The weather was sunny and warm."
#' ))
#' }
conflibert_classify <- function(text) {
  stopifnot(is.character(text), length(text) >= 1)
  py <- .get_py()

  parts <- lapply(text, function(t) {
    r <- py$classify(t)
    tibble::tibble(
      text         = t,
      label        = r$label,
      class        = as.integer(r[["class"]]),
      confidence   = r$confidence,
      prob_negative = r$prob_negative,
      prob_positive = r$prob_positive
    )
  })
  do.call(rbind, parts)
}


#' Multilabel Classification
#'
#' Score text against four event categories: Armed Assault,
#' Bombing or Explosion, Kidnapping, and Other. Each category is
#' scored independently.
#'
#' @param text A character vector of one or more texts.
#' @return A tibble with columns:
#'   \describe{
#'     \item{doc_id}{Integer index of the input text
#'       (only when \code{length(text) > 1}).}
#'     \item{text}{The input text.}
#'     \item{label}{Event category name.}
#'     \item{probability}{Score between 0 and 1.}
#'     \item{predicted}{Logical, TRUE if probability >= 0.5.}
#'   }
#' @export
#' @examples
#' \dontrun{
#' conflibert_multilabel("Insurgents kidnapped two aid workers near the border.")
#' }
conflibert_multilabel <- function(text) {
  stopifnot(is.character(text), length(text) >= 1)
  py <- .get_py()

  parts <- lapply(seq_along(text), function(i) {
    r <- py$multilabel(text[i])
    tibble::tibble(
      doc_id      = i,
      text        = text[i],
      label       = vapply(r, function(x) x$label, character(1)),
      probability = vapply(r, function(x) x$probability, numeric(1)),
      predicted   = vapply(r, function(x) x$predicted, logical(1))
    )
  })

  result <- do.call(rbind, parts)
  if (length(text) == 1) result$doc_id <- NULL
  result
}
