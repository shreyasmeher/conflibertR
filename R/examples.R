#' Load Example Dataset
#'
#' Load one of the bundled example datasets for testing fine-tuning
#' and model comparison. Returns a list with train, dev, and test
#' data frames ready to pass to \code{\link{conflibert_finetune}} or
#' \code{\link{conflibert_compare}}.
#'
#' @param name \code{"binary"} (conflict vs non-conflict, 2 classes) or
#'   \code{"multiclass"} (4 event types: Diplomacy, Armed Conflict,
#'   Protest, Humanitarian).
#' @return A list with three data frames: \code{$train}, \code{$dev},
#'   \code{$test}. Each has columns \code{text} and \code{label}.
#' @export
#' @examples
#' \dontrun{
#' data <- conflibert_example("binary")
#' data$train
#'
#' # Fine-tune with example data
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
#' }
conflibert_example <- function(name = c("binary", "multiclass")) {
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

  list(
    train = read_tsv("train.tsv"),
    dev   = read_tsv("dev.tsv"),
    test  = read_tsv("test.tsv")
  )
}
