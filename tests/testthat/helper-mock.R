# A fake Python backend so the R-side logic (validation, tibble shaping,
# classes, print and plot methods) can be tested without Python, torch,
# or any model downloads.

fake_py <- function() {
  list(
    is_loaded = function(task) TRUE,
    classify_batch = function(texts) {
      n <- length(texts)
      list(
        label = rep(c("Positive", "Negative"), length.out = n),
        class = rep(c(1L, 0L), length.out = n),
        confidence = rep(0.9, n),
        prob_negative = rep(0.1, n),
        prob_positive = rep(0.9, n)
      )
    },
    multilabel_batch = function(texts) {
      n <- length(texts)
      list(
        categories = c("Armed Assault", "Bombing or Explosion",
                       "Kidnapping", "Other"),
        probabilities = replicate(n, c(0.1, 0.2, 0.8, 0.05),
                                  simplify = FALSE)
      )
    },
    ner_batch = function(texts) {
      texts <- unlist(texts)
      n <- length(texts)
      list(
        doc_id = as.list(seq_len(n)),
        entity = as.list(substr(texts, 1, 4)),
        label  = as.list(rep("Location", n)),
        score  = as.list(rep(0.99, n)),
        start  = as.list(rep(1L, n)),
        end    = as.list(rep(4L, n))
      )
    },
    qa = function(context, question) {
      list(answer = "Geneva", score = 0.87, start = 29L, end = 34L)
    }
  )
}

empty_ner_py <- function() {
  py <- fake_py()
  py$ner_batch <- function(texts) {
    list(doc_id = list(), entity = list(), label = list(),
         score = list(), start = list(), end = list())
  }
  py
}

with_fake_py <- function(code, py = fake_py()) {
  cb <- get(".cb", envir = asNamespace("conflibertR"))
  old <- cb$py
  cb$py <- py
  on.exit(cb$py <- old, add = TRUE)
  force(code)
}

# A fabricated fine-tune result for print/plot tests.
fake_finetune <- function() {
  out <- list(
    metrics = tibble::tibble(accuracy = 0.9, precision = 0.88,
                             recall = 0.92, f1 = 0.9, loss = 0.31),
    runtime = 12.3,
    predictions = c(1L, 0L, 1L, 1L, 0L),
    probabilities = matrix(runif(10), ncol = 2),
    true_labels = c(1L, 0L, 1L, 0L, 0L),
    model_dir = tempdir(),
    model = "ConfliBERT",
    task = "binary"
  )
  class(out) <- "conflibert_finetune"
  out
}

# A fabricated comparison result.
fake_comparison <- function() {
  df <- tibble::tibble(
    model = c("ConfliBERT", "BERT Base Uncased"),
    runtime = c(60, 55),
    accuracy = c(0.91, 0.87),
    f1 = c(0.9, 0.85)
  )
  conflibertR:::.cb_result(df, "conflibert_comparison")
}

# A fabricated active-learning session (print/autoplot only).
fake_session <- function() {
  s <- list(
    round = 2L,
    query = tibble::tibble(
      text = c("Sample one.", "Sample two."),
      uncertainty = c(0.69, 0.65)
    ),
    metrics = tibble::tibble(
      round = c(0L, 1L), train_size = c(20L, 30L),
      accuracy = c(0.7, 0.8), f1 = c(0.65, 0.78),
      uncertainty_mean = c(0.6, 0.5), uncertainty_max = c(0.7, 0.62)
    ),
    labeled_n = 30L, pool_n = 40L, done = FALSE,
    .state = list(
      num_labels = 2L,
      params = list(model = "ConfliBERT", task = "binary",
                    strategy = "entropy", query_size = 10L)
    )
  )
  class(s) <- "conflibert_al_session"
  s
}
