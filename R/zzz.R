# Internal environment for caching the Python module
.cb <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  .cb$py <- NULL
  # Point reticulate to the conflibert virtualenv if it exists
  venv_path <- file.path("~", ".virtualenvs", "conflibert")
  if (dir.exists(venv_path)) {
    reticulate::use_virtualenv("conflibert", required = FALSE)
  }
}

#' Get the Python inference module (lazy-loaded and cached)
#' @noRd
.get_py <- function() {
  if (is.null(.cb$py)) {
    py_path <- system.file("python", package = "conflibertR")
    if (py_path == "") {
      stop(
        "Cannot find the Python inference module. ",
        "Is conflibertR installed correctly?",
        call. = FALSE
      )
    }
    .cb$py <- reticulate::import_from_path("inference", path = py_path)
  }
  .cb$py
}
