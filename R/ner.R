#' Named Entity Recognition
#'
#' Identify persons, organizations, locations, weapons, and other entity
#' types in text using the pretrained ConfliBERT NER model.
#'
#' @param text A character vector of one or more texts to analyze.
#' @return A tibble with columns:
#'   \describe{
#'     \item{doc_id}{Integer. Which input text this entity came from
#'       (only present when \code{length(text) > 1}).}
#'     \item{entity}{Character. The entity text.}
#'     \item{label}{Character. Entity type (Person, Organisation,
#'       Location, Weapon, etc.).}
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

  parts <- lapply(seq_along(text), function(i) {
    entities <- py$ner(text[i])
    if (length(entities) == 0) return(NULL)
    tibble::tibble(
      doc_id = i,
      entity = vapply(entities, function(e) e$entity, character(1)),
      label  = vapply(entities, function(e) e$label, character(1))
    )
  })

  result <- do.call(rbind, Filter(Negate(is.null), parts))
  if (is.null(result)) {
    return(tibble::tibble(doc_id = integer(), entity = character(), label = character()))
  }
  if (length(text) == 1) result$doc_id <- NULL
  result
}
