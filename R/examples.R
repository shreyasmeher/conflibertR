#' Load Example Dataset
#'
#' Load one of the bundled example datasets for testing fine-tuning
#' and model comparison. Returns a list with train, dev, and test
#' data frames ready to pass to \code{\link{conflibert_finetune}} or
#' \code{\link{conflibert_compare}}.
#'
#' @param name One of:
#'   \describe{
#'     \item{\code{"binary"}}{conflict vs non-conflict, 2 classes. Returns
#'       \code{$train}, \code{$dev}, \code{$test}.}
#'     \item{\code{"multiclass"}}{4 event types: Diplomacy, Armed Conflict,
#'       Protest, Humanitarian. Returns \code{$train}, \code{$dev},
#'       \code{$test}.}
#'     \item{\code{"active"}}{small labeled seed and an unlabeled pool for
#'       demoing \code{\link{conflibert_active_start}}. Returns
#'       \code{$seed}, \code{$pool}, \code{$dev}, and
#'       \code{$pool_labels}, a named integer vector mapping each
#'       pool text to its true label (for simulation / automated
#'       testing only; do not use these in a real workflow).}
#'   }
#' @return A named list of data frames (and a character vector for the
#'   active-learning pool). See \code{name} for the shape per dataset.
#' @export
#' @examples
#' # Loading the bundled data is pure R and needs no Python backend:
#' data <- conflibert_example("binary")
#' data$train
#'
#' @examplesIf conflibert_available()
#' # Fine-tune with example data (needs the Python backend)
#' data <- conflibert_example("binary")
#' result <- conflibert_finetune(
#'   train = data$train, dev = data$dev, test = data$test,
#'   model = "ConfliBERT", task = "binary", epochs = 1
#' )
#'
#' # Compare models with example data
#' comparison <- conflibert_compare(
#'   train = data$train, dev = data$dev, test = data$test,
#'   models = c("ConfliBERT", "BERT Base Uncased"),
#'   task = "binary", epochs = 1
#' )
conflibert_example <- function(name = c("binary", "multiclass", "active")) {
  name <- match.arg(name)
  dir <- system.file("examples", name, package = "conflibertR")
  if (dir == "") stop("Example dataset not found. Is conflibertR installed correctly?")

  read_tsv <- function(file) {
    df <- utils::read.delim(
      file.path(dir, file),
      header = FALSE, sep = "\t",
      col.names = c("text", "label"),
      stringsAsFactors = FALSE,
      quote = ""
    )
    df$label <- as.integer(df$label)
    df
  }

  if (name == "active") {
    pool_lines <- readLines(file.path(dir, "pool.txt"), warn = FALSE)
    pool_lines <- trimws(pool_lines)
    pool_lines <- pool_lines[nzchar(pool_lines)]
    oracle <- NULL
    oracle_path <- file.path(dir, "pool_with_labels.tsv")
    if (file.exists(oracle_path)) {
      oracle_df <- read_tsv("pool_with_labels.tsv")
      oracle <- stats::setNames(oracle_df$label, oracle_df$text)
    }
    return(list(
      seed         = read_tsv("seed.tsv"),
      pool         = pool_lines,
      dev          = read_tsv("dev.tsv"),
      pool_labels  = oracle
    ))
  }

  list(
    train = read_tsv("train.tsv"),
    dev   = read_tsv("dev.tsv"),
    test  = read_tsv("test.tsv")
  )
}
