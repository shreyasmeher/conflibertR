# conflibertR

R interface to [ConfliBERT](https://github.com/eventdata/ConfliBERT), a pretrained language model for conflict and political violence text analysis.

**Inference:** Named Entity Recognition, Binary Classification, Multilabel Classification, Question Answering.

**Training:** Fine-tune custom classifiers on your own data with any of 7 base models, and compare their performance side by side.

## Installation

```r
# install.packages("devtools")
devtools::install_github("shreyasmeher/conflibertR")
```

## One-time setup

Install the Python dependencies (torch, transformers, tensorflow). Conda is recommended as it handles torch compatibility better across platforms:

```r
library(conflibertR)

# Recommended (requires conda/miniconda):
conflibert_install(method = "conda")

# Alternative (if you don't have conda):
conflibert_install(method = "virtualenv")

# Restart R after this completes
```

## Inference

```r
library(conflibertR)
```

### Named Entity Recognition

```r
conflibert_ner("NATO forces were deployed near Kabul in September.")
#> # A tibble: 3 x 2
#>   entity    label
#>   <chr>     <chr>
#> 1 NATO      Organisation
#> 2 Kabul     Location
#> 3 September Temporal

# Vectorized -- multiple texts at once
conflibert_ner(c(
  "The UN Security Council met in New York.",
  "Soldiers from the 4th Brigade advanced."
))
#> # A tibble with doc_id, entity, label columns
```

### Binary Classification

```r
conflibert_classify("A bomb exploded in the crowded market.")
#> # A tibble: 1 x 6
#>   text                                   label    class confidence prob_negative prob_positive
#>   <chr>                                  <chr>    <int>      <dbl>         <dbl>         <dbl>
#> 1 A bomb exploded in the crowded market. Positive     1      0.98          0.02          0.98

# Vectorized
conflibert_classify(c(
  "Government troops clashed with rebels.",
  "The weather was sunny and warm."
))
```

### Multilabel Classification

```r
conflibert_multilabel("Insurgents kidnapped two aid workers near the border.")
#> # A tibble: 4 x 4
#>   text                      label                probability predicted
#>   <chr>                     <chr>                      <dbl> <lgl>
#> 1 Insurgents kidnapped ...  Armed Assault              0.12  FALSE
#> 2 Insurgents kidnapped ...  Bombing or Explosion       0.05  FALSE
#> 3 Insurgents kidnapped ...  Kidnapping                 0.91  TRUE
#> 4 Insurgents kidnapped ...  Other                      0.08  FALSE
```

### Question Answering

```r
conflibert_qa(
  context  = "The ceasefire was signed in Geneva on March 15th by both parties.",
  question = "Where was the ceasefire signed?"
)
#> [1] "Geneva"
```

## Benchmarking

Evaluate the pretrained binary classifier against your own labeled data:

```r
conflibert_benchmark(
  texts  = my_data$text,
  labels = my_data$label
)
#> # A tibble: 1 x 5
#>   accuracy precision recall    f1     n
#>      <dbl>     <dbl>  <dbl> <dbl> <int>
#> 1    0.85      0.83   0.87  0.85   200
```

## Example Datasets

The package bundles small synthetic datasets for quick testing:

```r
# Binary: conflict vs non-conflict (80 train / 20 dev / 20 test)
data <- conflibert_example("binary")

# Multiclass: 4 event types (80 train / 20 dev / 20 test)
data <- conflibert_example("multiclass")

data$train
#> # A data.frame: 80 x 2
#>   text                                                                    label
#>   <chr>                                                                   <int>
#> 1 Government forces launched an offensive against rebel positions ...          1
#> 2 The national football team secured a convincing victory ...                  0
#> ...
```

## Fine-tuning

Train a custom classifier on your own data (or use the built-in examples):

```r
data <- conflibert_example("binary")

result <- conflibert_finetune(
  train = data$train,
  dev   = data$dev,
  test  = data$test,
  model = "ConfliBERT",
  task  = "binary",
  epochs = 3
)

result$metrics
#> # A tibble: 1 x 4
#>   accuracy precision recall    f1
#>      <dbl>     <dbl>  <dbl> <dbl>
#> 1    0.92      0.91   0.93  0.92

# Save the trained model for later
result <- conflibert_finetune(
  train = data$train, dev = data$dev, test = data$test,
  model = "RoBERTa Base",
  save_dir = "./my_model"
)
```

## Comparing Models

Compare multiple base architectures on the same dataset:

```r
data <- conflibert_example("binary")

comparison <- conflibert_compare(
  train  = data$train,
  dev    = data$dev,
  test   = data$test,
  models = c("ConfliBERT", "BERT Base Uncased", "RoBERTa Base", "ModernBERT Base"),
  task   = "binary",
  epochs = 3
)

comparison
#> # A tibble: 4 x 6
#>   model            accuracy precision recall    f1 runtime
#>   <chr>               <dbl>     <dbl>  <dbl> <dbl>   <dbl>
#> 1 ConfliBERT          0.92      0.91   0.93  0.92    45.2
#> 2 BERT Base Uncased   0.88      0.86   0.89  0.87    42.1
#> 3 RoBERTa Base        0.90      0.89   0.91  0.90    48.3
#> 4 ModernBERT Base     0.93      0.92   0.94  0.93    38.7
```

### Available models

```r
conflibert_models()
#> [1] "ConfliBERT"        "BERT Base Uncased" "BERT Base Cased"
#> [4] "RoBERTa Base"      "ModernBERT Base"   "DeBERTa v3 Base"
#> [7] "DistilBERT Base"
```

## How it works

The package uses [reticulate](https://rstudio.github.io/reticulate/) to call HuggingFace Transformers models via Python. Models are downloaded from HuggingFace Hub on first use and cached locally. No GPU is required (CPU works fine), but CUDA and Apple Silicon MPS are used automatically when available.

## Citation

Brandt, P.T., Alsarra, S., D'Orazio, V., Heintze, D., Khan, L., Meher, S., Osorio, J. and Sianan, M., 2025. Extractive versus Generative Language Models for Political Conflict Text Classification. *Political Analysis*, pp.1-29.
