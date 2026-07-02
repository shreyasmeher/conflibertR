#' Question Answering
#'
#' Extract an answer from a context passage given a question. Both
#' arguments are vectorized: pass equal-length vectors to answer many
#' questions in one call, or a single context with several questions
#' (scalars are recycled).
#'
#' The published question-answering checkpoint ships only 'TensorFlow'
#' weights. The first call converts them to 'PyTorch' once and caches the
#' result under \code{~/.cache/conflibertR} (or \code{$XDG_CACHE_HOME} when
#' set); subsequent calls reuse the cache and never touch 'TensorFlow'
#' again. Downloaded 'Hugging Face' models are cached under \code{$HF_HOME}
#' (default \code{~/.cache/huggingface}). Both caches are written only on
#' explicit, network-enabled calls, and can be relocated by setting those
#' environment variables.
#'
#' @param context Character vector of passages.
#' @param question Character vector of questions.
#' @param details If \code{TRUE}, return a tibble with the answer plus
#'   the model's confidence score and the character span of the answer
#'   inside the context. Default \code{FALSE}.
#' @return With \code{details = FALSE} (default), a character vector of
#'   answers (a single string when one question is asked, exactly as in
#'   previous versions). With \code{details = TRUE}, a tibble with
#'   columns \code{question}, \code{answer}, \code{score}, \code{start},
#'   \code{end}, and \code{context}.
#' @export
#' @examplesIf conflibert_available()
#' conflibert_qa(
#'   context  = "The ceasefire was signed in Geneva on March 15th.",
#'   question = "Where was the ceasefire signed?"
#' )
#'
#' # Several questions against one passage, with scores:
#' conflibert_qa(
#'   context  = "The ceasefire was signed in Geneva on March 15th.",
#'   question = c("Where was the ceasefire signed?",
#'                "When was the ceasefire signed?"),
#'   details  = TRUE
#' )
conflibert_qa <- function(context, question, details = FALSE) {
  stopifnot(
    is.character(context), length(context) >= 1,
    is.character(question), length(question) >= 1
  )
  n <- max(length(context), length(question))
  if (!length(context) %in% c(1L, n) || !length(question) %in% c(1L, n)) {
    stop(
      "`context` and `question` must have the same length, ",
      "or one of them must be a single string.",
      call. = FALSE
    )
  }
  context <- rep_len(context, n)
  question <- rep_len(question, n)

  py <- .get_py()
  .cb_announce_load("qa", "question answering")

  rows <- lapply(seq_len(n), function(i) py$qa(context[i], question[i]))
  answers <- vapply(rows, function(r) as.character(r$answer), character(1))

  if (!details) {
    return(answers)
  }
  tibble::tibble(
    question = question,
    answer   = answers,
    score    = vapply(rows, function(r) as.numeric(r$score), numeric(1)),
    start    = vapply(rows, function(r) as.integer(r$start), integer(1)),
    end      = vapply(rows, function(r) as.integer(r$end), integer(1)),
    context  = context
  )
}
