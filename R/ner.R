#' Named Entity Recognition
#'
#' Identify persons, organizations, locations, weapons, and other entity
#' types in text using the pretrained ConfliBERT NER model. Inference is
#' batched, and printing highlights each entity in its source sentence.
#'
#' @param text A character vector of one or more texts to analyze.
#' @return A tibble (with a themed print method) with columns:
#'   \describe{
#'     \item{doc_id}{Integer. Which input text this entity came from
#'       (only present when \code{length(text) > 1}).}
#'     \item{entity}{Character. The entity text.}
#'     \item{label}{Character. Entity type (Person, Organisation,
#'       Location, Weapon, etc.).}
#'     \item{score}{Numeric. Mean model confidence for the entity span.}
#'     \item{start, end}{Integer. 1-based inclusive character offsets of
#'       the entity in the input text (use with \code{substr()}).}
#'   }
#' @export
#' @examples
#' \dontrun{
#' conflibert_ner("The soldiers attacked the village near Kabul.")
#'
#' conflibert_ner(c(
#'   "NATO forces were deployed to the region.",
#'   "The UN Security Council met in New York."
#' ))
#' }
conflibert_ner <- function(text) {
  stopifnot(is.character(text), length(text) >= 1)
  py <- .get_py()
  .cb_announce_load("ner", "named entity recognition")

  idx <- split(seq_along(text), ceiling(seq_along(text) / 64L))
  show_progress <- length(idx) > 1L
  if (show_progress) {
    cli::cli_progress_bar("Extracting entities", total = length(text))
  }

  parts <- vector("list", length(idx))
  for (i in seq_along(idx)) {
    r <- py$ner_batch(as.list(text[idx[[i]]]))
    # python doc_ids are relative to the chunk; shift to global indices
    doc_id <- as.integer(unlist(r$doc_id, use.names = FALSE))
    parts[[i]] <- tibble::tibble(
      doc_id = idx[[i]][doc_id],
      entity = as.character(unlist(r$entity, use.names = FALSE)),
      label  = as.character(unlist(r$label, use.names = FALSE)),
      score  = as.numeric(unlist(r$score, use.names = FALSE)),
      start  = as.integer(unlist(r$start, use.names = FALSE)),
      end    = as.integer(unlist(r$end, use.names = FALSE))
    )
    if (show_progress) cli::cli_progress_update(inc = length(idx[[i]]))
  }
  if (show_progress) cli::cli_progress_done()

  result <- do.call(rbind, parts)
  if (is.null(result) || nrow(result) == 0L) {
    result <- tibble::tibble(
      doc_id = integer(), entity = character(), label = character(),
      score = numeric(), start = integer(), end = integer()
    )
  }
  if (length(text) == 1) result$doc_id <- NULL
  attr(result, "texts") <- text
  .cb_result(result, "conflibert_ner")
}
