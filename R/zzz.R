# Internal environment for caching the Python module
.cb <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  .cb$py <- NULL
  .cb$available <- NULL
  .cb$env_found <- TRUE
  # Find the conflibert environment (conda first, then virtualenv) with
  # filesystem checks only: running `conda env list` here can take over
  # 10 seconds on some systems, and loading the package must stay fast.
  if (!is.null(.cb_conda_env_dir())) {
    reticulate::use_condaenv("conflibert", required = FALSE)
  } else if (.cb_virtualenv_found()) {
    reticulate::use_virtualenv("conflibert", required = FALSE)
  } else {
    .cb$env_found <- FALSE
  }
}

# Locate the "conflibert" conda env directory without running the conda
# binary (`conda env list` can take >10s to cold-start; on CRAN's
# Windows pretest machine that showed up as elapsed time in every
# example guarded by conflibert_available()). Checks the env directory
# adjacent to the conda installation plus the user-level default, which
# covers standard conda/miniconda/miniforge layouts.
.cb_conda_env_dir <- function() {
  root <- tryCatch(dirname(dirname(reticulate::conda_binary())),
                   error = function(e) NULL)
  candidates <- c(
    if (!is.null(root)) file.path(root, "envs", "conflibert"),
    path.expand("~/.conda/envs/conflibert"),
    file.path(Sys.getenv("USERPROFILE", ""), ".conda", "envs",
              "conflibert")
  )
  candidates <- candidates[nzchar(candidates) & dir.exists(candidates)]
  if (length(candidates)) candidates[[1L]] else NULL
}

.cb_virtualenv_found <- function() {
  tryCatch(reticulate::virtualenv_exists("conflibert"),
           error = function(e) FALSE)
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
