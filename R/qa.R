#' Question Answering
#'
#' Extract an answer from a context passage given a question.
#'
#' @param context Character string containing the passage.
#' @param question Character string with the question.
#' @return A character string with the extracted answer.
#' @export
#' @examples
#' \dontrun{
#' conflibert_qa(
#'   context  = "The ceasefire was signed in Geneva on March 15th.",
#'   question = "Where was the ceasefire signed?"
#' )
#' }
conflibert_qa <- function(context, question) {
  stopifnot(
    is.character(context), length(context) == 1,
    is.character(question), length(question) == 1
  )
  py <- .get_py()
  result <- py$qa(context, question)
  result$answer
}
