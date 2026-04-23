#' Install Python dependencies for ConfliBERT
#'
#' Creates a Python environment and installs the packages needed
#' to run ConfliBERT models (torch, transformers, tensorflow, tf-keras).
#' Only needs to be run once.
#'
#' If you have issues with the default virtualenv method (especially
#' torch compatibility), try \code{method = "conda"} instead.
#'
#' @param envname Name of the environment. Default: \code{"conflibert"}.
#' @param method \code{"conda"} (recommended) or \code{"virtualenv"}.
#' @return Invisible \code{NULL}. Called for its side effect.
#' @export
#' @examples
#' \dontrun{
#' # Recommended:
#' conflibert_install(method = "conda")
#'
#' # Alternative:
#' conflibert_install(method = "virtualenv")
#' }
conflibert_install <- function(envname = "conflibert", method = "auto") {
  packages <- c("torch", "transformers>=4.40,<5", "accelerate", "peft",
                 "scikit-learn", "tensorflow", "tf-keras")

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
    message("Detected method: ", method)
  }

  if (method == "conda") {
    message("Installing with conda...")
    reticulate::conda_create(envname)
    reticulate::conda_install(envname, packages = packages, pip = TRUE)
  } else {
    message("Installing with virtualenv...")
    reticulate::virtualenv_create(envname)
    reticulate::virtualenv_install(envname, packages = packages)
  }

  message("Python dependencies installed in '", envname, "' environment.")
  message("Restart R, then: library(conflibertR)")
  invisible(NULL)
}
