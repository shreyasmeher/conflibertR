#' Benchmark the Pretrained Classifier
#'
#' Evaluate the pretrained ConfliBERT binary classifier against labeled
#' data and compute accuracy, precision, recall, and F1.
#'
#' @param texts Character vector of texts.
#' @param labels Integer vector of true labels (0 or 1).
#' @return A tibble with one row and columns: accuracy, precision, recall, f1, n.
#' @export
#' @examplesIf conflibert_available()
#' conflibert_benchmark(
#'   texts  = c("A bomb exploded.", "The weather was nice."),
#'   labels = c(1L, 0L)
#' )
conflibert_benchmark <- function(texts, labels) {
  stopifnot(is.character(texts), length(texts) == length(labels))
  py <- .get_py()
  r <- py$benchmark(texts, as.integer(labels))
  tibble::tibble(
    accuracy  = r$accuracy,
    precision = r$precision,
    recall    = r$recall,
    f1        = r$f1,
    n         = as.integer(r$n)
  )
}
