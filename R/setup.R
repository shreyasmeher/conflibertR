#' Install Python dependencies for ConfliBERT
#'
#' Creates a Python environment and installs the packages needed
#' to run ConfliBERT models (torch, transformers, and friends).
#' Only needs to be run once.
#'
#' As of conflibertR 0.5.0 the backend is PyTorch-only: TensorFlow is no
#' longer required, which makes installation considerably smaller and
#' more reliable. If you have issues with the default virtualenv method,
#' try \code{method = "conda"} instead.
#'
#' @param envname Name of the environment. Default: \code{"conflibert"}.
#' @param method \code{"auto"} (default; uses conda when available),
#'   \code{"conda"}, or \code{"virtualenv"}.
#' @param qa If \code{TRUE}, also install TensorFlow. The published
#'   ConfliBERT QA checkpoint only ships TensorFlow weights;
#'   \code{\link{conflibert_qa}} converts them to PyTorch on first use
#'   (a one-time step that needs TensorFlow) and caches the result, after
#'   which TensorFlow is never used again. All other functions are pure
#'   PyTorch. Default: \code{FALSE}.
#'
#' @details Models downloaded from 'Hugging Face' are cached under
#'   \code{$HF_HOME} (default \code{~/.cache/huggingface}) and the converted
#'   QA weights under \code{~/.cache/conflibertR} (or \code{$XDG_CACHE_HOME}).
#'   Set those environment variables to relocate the caches.
#' @return Invisible \code{NULL}. Called for its side effect.
#' @export
#' @examples
#' \dontrun{
#' conflibert_install()
#'
#' # Include TensorFlow for the one-time QA weight conversion:
#' conflibert_install(qa = TRUE)
#' }
conflibert_install <- function(envname = "conflibert", method = "auto",
                               qa = FALSE) {
  packages <- if (isTRUE(qa)) {
    # the one-time TF -> PyTorch QA conversion needs tensorflow and a
    # transformers version that still bundles TF support
    c("torch", "transformers>=4.40,<4.50", "accelerate", "peft",
      "scikit-learn", "tensorflow", "tf-keras")
  } else {
    c("torch", "transformers>=4.40", "accelerate", "peft",
      "scikit-learn")
  }

  # Auto-detect: use conda if available, otherwise virtualenv
  if (method == "auto") {
    has_conda <- nzchar(Sys.which("conda"))
    if (!has_conda) {
      has_conda <- tryCatch({
        bin <- reticulate::conda_binary()
        file.exists(bin)
      }, error = function(e) FALSE, warning = function(w) FALSE)
    }
    method <- if (has_conda) "conda" else "virtualenv"
    cli::cli_alert_info("Detected method: {method}")
  }

  if (method == "conda") {
    cli::cli_progress_step("Installing Python dependencies with conda")
    reticulate::conda_create(envname)
    reticulate::conda_install(envname, packages = packages, pip = TRUE)
  } else {
    cli::cli_progress_step("Installing Python dependencies with virtualenv")
    reticulate::virtualenv_create(envname)
    reticulate::virtualenv_install(envname, packages = packages)
  }

  cli::cli_alert_success(
    "Python dependencies installed in the {.val {envname}} environment."
  )
  cli::cli_alert_info("Restart R, then run {.code library(conflibertR)}.")
  invisible(NULL)
}
