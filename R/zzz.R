# Internal environment for caching the Python module
.cb <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  .cb$py <- NULL
  .cb$env_found <- TRUE
  # Try to find the conflibert environment (conda first, then virtualenv)
  conda_envs <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  if (!is.null(conda_envs) && "conflibert" %in% conda_envs$name) {
    reticulate::use_condaenv("conflibert", required = FALSE)
  } else {
    venv_path <- file.path("~", ".virtualenvs", "conflibert")
    if (dir.exists(venv_path)) {
      reticulate::use_virtualenv("conflibert", required = FALSE)
    } else {
      .cb$env_found <- FALSE
    }
  }
}

.onAttach <- function(libname, pkgname) {
  if (!isTRUE(.cb$env_found)) {
    packageStartupMessage(
      "conflibertR: no 'conflibert' Python environment found.\n",
      "  Run conflibert_install() once, then restart R.\n",
      "  Use conflibert_status() to check your setup."
    )
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
    .cb$py <- tryCatch(
      reticulate::import_from_path("inference", path = py_path),
      error = function(e) {
        cli::cli_abort(
          c(
            "Could not load the conflibertR Python backend.",
            "x" = "{conditionMessage(e)}",
            "i" = "Run {.code conflibert_status()} to diagnose your setup.",
            "i" = paste(
              "If you have not installed the Python dependencies yet,",
              "run {.code conflibert_install()} and restart R."
            )
          ),
          call = NULL
        )
      }
    )
  }
  .cb$py
}
