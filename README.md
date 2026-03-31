# conflibertR

R interface to [ConfliBERT](https://github.com/eventdata/ConfliBERT), a pretrained language model for conflict and political violence text analysis.

All four inference tasks are supported: Named Entity Recognition, Binary Classification, Multilabel Classification, and Question Answering. Functions accept character vectors and return tibbles.

## Installation

```r
# From the local package directory
install.packages("path/to/conflibertR", repos = NULL, type = "source")

# Or with devtools
devtools::install_local("path/to/conflibertR")
```

## One-time setup

Install the Python dependencies (torch, transformers, tensorflow):

```r
library(conflibertR)
conflibert_install()
# Restart R after this completes
```

## Usage

```r
library(conflibertR)
```

### Named Entity Recognition

```r
conflibert_ner("NATO forces were deployed near Kabul in September.")
#> # A tibble: 3 x 2
#>   entity label
#>   <chr>  <chr>
#> 1 NATO   Organisation
#> 2 Kabul  Location
#> 3 September Temporal

# Multiple texts
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
#>   text                                                  label                probability predicted
#>   <chr>                                                 <chr>                      <dbl> <lgl>
#> 1 Insurgents kidnapped two aid workers near the border. Armed Assault              0.12  FALSE
#> 2 Insurgents kidnapped two aid workers near the border. Bombing or Explosion       0.05  FALSE
#> 3 Insurgents kidnapped two aid workers near the border. Kidnapping                 0.91  TRUE
#> 4 Insurgents kidnapped two aid workers near the border. Other                      0.08  FALSE
```

### Question Answering

```r
conflibert_qa(
  context  = "The ceasefire was signed in Geneva on March 15th by both parties.",
  question = "Where was the ceasefire signed?"
)
#> [1] "Geneva"
```

## How it works

The package uses [reticulate](https://rstudio.github.io/reticulate/) to call HuggingFace Transformers models via Python. Models are downloaded from HuggingFace Hub on first use and cached locally. No GPU is required (CPU works fine), but CUDA and Apple Silicon MPS are used automatically when available.

## Citation

Brandt, P.T., Alsarra, S., D'Orazio, V., Heintze, D., Khan, L., Meher, S., Osorio, J. and Sianan, M., 2025. Extractive versus Generative Language Models for Political Conflict Text Classification. *Political Analysis*, pp.1-29.
