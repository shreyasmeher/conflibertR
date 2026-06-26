#' Fine-tune a Classification Model
#'
#' Train a binary or multiclass text classifier on your own data using
#' any of the supported base models (ConfliBERT, BERT, RoBERTa,
#' ModernBERT, DeBERTa, DistilBERT).
#'
#' @param train A data.frame with \code{text} and \code{label} columns (training set).
#' @param dev A data.frame with \code{text} and \code{label} columns (validation set).
#' @param test A data.frame with \code{text} and \code{label} columns (test set).
#' @param model Base model name. One of: \code{"ConfliBERT"},
#'   \code{"BERT Base Uncased"}, \code{"BERT Base Cased"},
#'   \code{"RoBERTa Base"}, \code{"ModernBERT Base"},
#'   \code{"DeBERTa v3 Base"}, \code{"DistilBERT Base"}.
#'   Default: \code{"ConfliBERT"}.
#' @param task \code{"binary"} or \code{"multiclass"}. Default: \code{"binary"}.
#' @param epochs Number of training epochs. Default: 3.
#' @param batch_size Training batch size. Default: 8.
#' @param lr Learning rate. Default: 2e-5.
#' @param save_dir Optional directory to save the trained model. If provided,
#'   the model and tokenizer are saved there and can be loaded later with
#'   \code{\link{conflibert_load}}.
#' @param use_lora If \code{TRUE}, fine-tune with a LoRA adapter
#'   (parameter-efficient; cuts GPU memory roughly 5x). The adapter is merged
#'   into the base model before saving, so loading is the same as for
#'   full fine-tuning. Default: \code{FALSE}.
#' @param lora_rank LoRA rank. Default: 8.
#' @param lora_alpha LoRA alpha. Default: 16.
#' @param seed Random seed for reproducibility. Seeds the classifier-head
#'   initialization, data shuffling, and dropout so that two runs with the
#'   same seed on the same hardware and package versions give identical
#'   results. Change it to study run-to-run variability. Default: 42.
#' @return An object of class \code{"conflibert_finetune"} (a list, so all
#'   existing \code{$} access keeps working) with:
#'   \describe{
#'     \item{metrics}{Tibble of test-set metrics.}
#'     \item{runtime}{Training time in seconds.}
#'     \item{predictions}{Integer vector of predicted class labels.}
#'     \item{probabilities}{Matrix of class probabilities (rows = samples,
#'       columns = classes).}
#'     \item{true_labels}{Integer vector of true labels.}
#'     \item{model_dir}{Path where the model checkpoint was saved.}
#'     \item{model, task}{The base model name and task type.}
#'   }
#'   It has a themed \code{print()} method and a \code{plot()} method
#'   showing the test-set confusion matrix.
#' @export
#' @examples
#' \dontrun{
#' train <- data.frame(
#'   text  = c("Troops advanced.", "Nice weather today."),
#'   label = c(1L, 0L)
#' )
#' result <- conflibert_finetune(train, dev = train, test = train, epochs = 1)
#' result$metrics
#' }
conflibert_finetune <- function(
    train, dev, test,
    model = "ConfliBERT", task = "binary",
    epochs = 3, batch_size = 8, lr = 2e-5,
    save_dir = NULL,
    use_lora = FALSE, lora_rank = 8, lora_alpha = 16,
    seed = 42
) {
  stopifnot(
    is.data.frame(train), "text" %in% names(train), "label" %in% names(train),
    is.data.frame(dev),   "text" %in% names(dev),   "label" %in% names(dev),
    is.data.frame(test),  "text" %in% names(test),  "label" %in% names(test)
  )
  py <- .get_py()

  r <- py$finetune(
    train_texts  = as.character(train$text),
    train_labels = as.integer(train$label),
    dev_texts    = as.character(dev$text),
    dev_labels   = as.integer(dev$label),
    test_texts   = as.character(test$text),
    test_labels  = as.integer(test$label),
    model_name   = model,
    task         = task,
    epochs       = as.integer(epochs),
    batch_size   = as.integer(batch_size),
    lr           = as.numeric(lr),
    save_dir     = save_dir,
    use_lora     = isTRUE(use_lora),
    lora_rank    = as.integer(lora_rank),
    lora_alpha   = as.integer(lora_alpha),
    seed         = as.integer(seed)
  )

  metrics_df <- tibble::as_tibble(as.data.frame(r$metrics))
  probs <- r$probabilities
  probs_mat <- if (is.matrix(probs)) probs else do.call(rbind, probs)

  out <- list(
    metrics      = metrics_df,
    runtime      = r$runtime,
    predictions  = unlist(r$predictions),
    probabilities = probs_mat,
    true_labels  = unlist(r$true_labels),
    model_dir    = r$model_dir,
    model        = model,
    task         = task
  )
  class(out) <- "conflibert_finetune"
  out
}


#' Compare Multiple Models
#'
#' Fine-tune several base models on the same dataset and return a
#' comparison table of test-set metrics. Useful for selecting the best
#' architecture for your data.
#'
#' @inheritParams conflibert_finetune
#' @param models Character vector of model names to compare.
#'   See \code{\link{conflibert_models}} for available names.
#' @return A tibble with one row per model and columns for each metric
#'   plus \code{runtime}. It has a themed \code{print()} method that
#'   ranks models, and a \code{plot()} method comparing metrics.
#' @export
#' @examples
#' \dontrun{
#' comparison <- conflibert_compare(
#'   train  = train_df,
#'   dev    = dev_df,
#'   test   = test_df,
#'   models = c("ConfliBERT", "BERT Base Uncased", "RoBERTa Base"),
#'   task   = "binary",
#'   epochs = 3
#' )
#' comparison
#' }
conflibert_compare <- function(
    train, dev, test,
    models = c("ConfliBERT", "BERT Base Uncased"),
    task = "binary",
    epochs = 3, batch_size = 8, lr = 2e-5,
    use_lora = FALSE, lora_rank = 8, lora_alpha = 16,
    seed = 42
) {
  stopifnot(
    is.data.frame(train), is.data.frame(dev), is.data.frame(test),
    length(models) >= 2
  )
  py <- .get_py()

  results <- py$compare(
    train_texts  = as.character(train$text),
    train_labels = as.integer(train$label),
    dev_texts    = as.character(dev$text),
    dev_labels   = as.integer(dev$label),
    test_texts   = as.character(test$text),
    test_labels  = as.integer(test$label),
    model_names  = as.list(models),
    task         = task,
    epochs       = as.integer(epochs),
    batch_size   = as.integer(batch_size),
    lr           = as.numeric(lr),
    use_lora     = isTRUE(use_lora),
    lora_rank    = as.integer(lora_rank),
    lora_alpha   = as.integer(lora_alpha),
    seed         = as.integer(seed)
  )

  rows <- lapply(results, function(r) tibble::as_tibble(as.data.frame(r)))
  # a failed model returns only (model, error); align columns before binding
  all_cols <- unique(unlist(lapply(rows, names)))
  rows <- lapply(rows, function(df) {
    for (mc in setdiff(all_cols, names(df))) df[[mc]] <- NA
    df[all_cols]
  })
  out <- do.call(rbind, rows)
  .cb_result(out, "conflibert_comparison")
}


#' List Available Models
#'
#' Returns the names of base models that can be used for fine-tuning
#' and comparison.
#'
#' @return A character vector of model names.
#' @export
#' @examples
#' conflibert_models()
conflibert_models <- function() {
  c(
    "ConfliBERT",
    "BERT Base Uncased",
    "BERT Base Cased",
    "RoBERTa Base",
    "ModernBERT Base",
    "DeBERTa v3 Base",
    "DistilBERT Base"
  )
}
