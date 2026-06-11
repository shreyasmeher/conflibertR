# ============================================================
# conflibertR 0.5.0 showcase
# Run top to bottom from the package root.
#
# Part 1 needs only the package itself (instant, no Python).
# Part 2 runs real inference (downloads models on first use).
# Part 3 runs a small real fine-tune + active learning round
#         (a few minutes on CPU) -- flip RUN_TRAINING to TRUE.
# ============================================================

devtools::load_all(".")   # or: library(conflibertR) after installing

RUN_TRAINING <- FALSE

# ------------------------------------------------------------
# Part 0: check your setup
# ------------------------------------------------------------
conflibert_status()

# ------------------------------------------------------------
# Part 1: the new look, no Python needed
# (fabricated objects, just to show printing and plotting)
# ------------------------------------------------------------

# A fine-tune result, as conflibert_finetune() now returns it:
demo_ft <- structure(list(
  metrics = tibble::tibble(accuracy = 0.92, precision = 0.91,
                           recall = 0.93, f1 = 0.92, loss = 0.21),
  runtime = 184.2,
  predictions = sample(0:1, 60, replace = TRUE),
  true_labels = sample(0:1, 60, replace = TRUE),
  probabilities = matrix(runif(120), ncol = 2),
  model_dir = "~/models/demo", model = "ConfliBERT", task = "binary"
), class = "conflibert_finetune")

demo_ft                        # themed print with metric bars
plot(demo_ft)                  # confusion matrix, base graphics
print(ggplot2::autoplot(demo_ft))   # confusion matrix, ggplot2

# A model comparison, as conflibert_compare() now returns it:
demo_cmp <- conflibertR:::.cb_result(tibble::tibble(
  model    = c("ConfliBERT", "BERT Base Uncased", "RoBERTa Base",
               "ModernBERT Base"),
  runtime  = c(45.2, 42.1, 48.3, 38.7),
  accuracy = c(0.92, 0.88, 0.90, 0.93),
  f1       = c(0.92, 0.87, 0.90, 0.93)
), "conflibert_comparison")

demo_cmp                       # ranked leaderboard with a star on the winner
plot(demo_cmp)                 # dot chart
print(ggplot2::autoplot(demo_cmp))

# An active-learning session (print + the two-panel progress plot):
demo_al <- structure(list(
  round = 4L,
  query = tibble::tibble(
    text = c("Militants ambushed a convoy near the checkpoint.",
             "The council approved the new budget."),
    uncertainty = c(0.691, 0.642)
  ),
  metrics = tibble::tibble(
    round = 0:3, train_size = c(20L, 30L, 40L, 50L),
    accuracy = c(0.68, 0.75, 0.82, 0.86), f1 = c(0.61, 0.72, 0.80, 0.85),
    uncertainty_mean = c(0.64, 0.58, 0.49, 0.41),
    uncertainty_max  = c(0.69, 0.66, 0.58, 0.50)
  ),
  labeled_n = 50L, pool_n = 21L, done = FALSE,
  .state = list(num_labels = 2L, params = list(
    model = "ConfliBERT", task = "binary", strategy = "entropy",
    query_size = 10L))
), class = "conflibert_al_session")

demo_al                        # themed session summary
plot(demo_al)                  # base graphics, two panels
print(ggplot2::autoplot(demo_al))   # ggplot2 version

# The exported theme for your own figures:
print(
  ggplot2::ggplot(mtcars, ggplot2::aes(wt, mpg)) +
    ggplot2::geom_point(colour = "#0ea5e9", size = 3) +
    ggplot2::labs(title = "theme_conflibert()",
                  subtitle = "the package look, on your own plots") +
    theme_conflibert()
)

# ------------------------------------------------------------
# Part 2: real inference (Python env required;
# first call per task downloads the model from HuggingFace)
# ------------------------------------------------------------

# Binary classification -- colored labels + confidence bars,
# plus an aggregated plot of the results
cl <- conflibert_classify(c(
  "Government troops clashed with rebels near the border.",
  "The local football team won the championship.",
  "A car bomb exploded outside the ministry.",
  "Parliament passed the new education budget.",
  "Gunmen stormed a checkpoint overnight."
))
cl
plot(cl)                       # texts per class, avg confidence
print(ggplot2::autoplot(cl))

# NER -- entities highlighted inside the source sentence,
# with confidence scores and character offsets
en <- conflibert_ner(c(
  "NATO forces were deployed near Kabul in September.",
  "The UN Security Council met in New York to discuss Sudan.",
  "Rebels armed with rifles attacked a convoy near Mosul."
))
en
plot(en)                       # entity counts by type
print(ggplot2::autoplot(en))
print(ggplot2::autoplot(en, type = "entities"))   # most frequent entities

# Multilabel -- check marks + probability bars in the console.
# Plots aggregate across texts (single texts get probability bars).
ml <- conflibert_multilabel(c(
  "Insurgents kidnapped two aid workers near the border.",
  "A roadside bomb destroyed a military convoy.",
  "Gunmen opened fire on a patrol in the north.",
  "Militants abducted a local official on Tuesday.",
  "An explosion damaged the power station.",
  "Armed men assaulted villagers near the river."
))
ml
plot(ml)                       # share of texts flagged per category
print(ggplot2::autoplot(ml))
one <- conflibert_multilabel("Insurgents kidnapped two aid workers.")
print(ggplot2::autoplot(one))  # single text: category probabilities

# QA -- now vectorized, with scores and answer spans
conflibert_qa(
  context  = "The ceasefire was signed in Geneva on March 15th by both parties.",
  question = c("Where was the ceasefire signed?",
               "When was the ceasefire signed?"),
  details  = TRUE
)

# ------------------------------------------------------------
# Part 3: real training demos (slow on CPU; set RUN_TRAINING)
# ------------------------------------------------------------
if (RUN_TRAINING) {
  data <- conflibert_example("binary")

  # Fine-tune -> themed result + confusion matrix
  ft <- conflibert_finetune(
    train = data$train, dev = data$dev, test = data$test,
    model = "ConfliBERT", task = "binary", epochs = 1
  )
  ft
  plot(ft)

  # One active-learning round with simulated labels
  al <- conflibert_example("active")
  session <- conflibert_active_start(
    seed = al$seed, pool = al$pool, dev = al$dev,
    query_size = 5, epochs = 1
  )
  session
  labels  <- unname(al$pool_labels[session$query$text])
  session <- conflibert_active_next(session, labels)
  plot(session)
  print(ggplot2::autoplot(session))
}
