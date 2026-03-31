#' Install Python dependencies for ConfliBERT
#'
#' Creates a Python virtual environment and installs the packages needed
#' to run ConfliBERT models (torch, transformers, tensorflow, tf-keras).
#' Only needs to be run once.
#'
#' @param envname Name of the virtual environment. Default: \code{"conflibert"}.
#' @param method \code{"virtualenv"} (default) or \code{"conda"}.
#' @return Invisible \code{NULL}. Called for its side effect.
#' @export
#' @examples
#' \dontrun{
#' conflibert_install()
#' }
conflibert_install <- function(envname = "conflibert", method = "virtualenv") {
  packages <- c("torch", "transformers>=4.40,<5", "tensorflow", "tf-keras")

  if (method == "virtualenv") {
    reticulate::virtualenv_create(envname)
    reticulate::virtualenv_install(envname, packages = packages)
  } else {
    reticulate::conda_create(envname)
    reticulate::conda_install(envname, packages = packages, pip = TRUE  )
  }

  message("Python dependencies installed in '", envname, "' environment.")
  message("Restart R, then: library(conflibertR)")
  invisible(NULL)
}
