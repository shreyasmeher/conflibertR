#' Check the conflibertR Setup
#'
#' Run a quick diagnostic of the Python backend: whether the
#' \code{"conflibert"} environment exists, which Python is active, which
#' required packages are importable, and what compute device will be
#' used. Prints a checklist and gives specific advice when something is
#' missing.
#'
#' @return Invisibly, a list with \code{env_found}, \code{python},
#'   \code{packages} (named logical vector), \code{device}, and
#'   \code{ok}.
#' @export
#' @examples
#' \dontrun{
#' conflibert_status()
#' }
conflibert_status <- function() {
  cli::cli_rule(left = cli::style_bold("conflibertR status"))

  # environment discovery
  conda_envs <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  conda_found <- !is.null(conda_envs) && "conflibert" %in% conda_envs$name
  venv_found <- tryCatch(
    reticulate::virtualenv_exists("conflibert"),
    error = function(e) FALSE
  )
  env_found <- conda_found || venv_found

  if (conda_found) {
    cli::cli_alert_success("Found conda environment {.val conflibert}")
  } else if (venv_found) {
    cli::cli_alert_success("Found virtualenv {.val conflibert}")
  } else {
    cli::cli_alert_danger("No {.val conflibert} Python environment found")
    cli::cli_text(cli::col_grey(
      "  Run {.code conflibert_install()} and restart R."
    ))
  }

  # active python
  python <- tryCatch({
    cfg <- reticulate::py_config()
    cfg$python
  }, error = function(e) NULL)
  if (!is.null(python)) {
    cli::cli_alert_success("Python: {.path {python}}")
  } else {
    cli::cli_alert_danger("Python could not be initialized")
  }

  # required packages
  pkgs <- c(
    torch = "torch", transformers = "transformers",
    peft = "peft", `scikit-learn` = "sklearn",
    accelerate = "accelerate"
  )
  packages <- vapply(pkgs, function(mod) {
    tryCatch(reticulate::py_module_available(mod),
             error = function(e) FALSE)
  }, logical(1))
  for (i in seq_along(packages)) {
    if (packages[i]) {
      cli::cli_alert_success("Python package {.pkg {names(packages)[i]}}")
    } else {
      cli::cli_alert_danger(
        "Python package {.pkg {names(packages)[i]}} is missing"
      )
    }
  }

  # compute device
  device <- NULL
  if (isTRUE(packages[["torch"]])) {
    device <- tryCatch({
      torch <- reticulate::import("torch")
      if (isTRUE(torch$cuda$is_available())) "cuda" else "cpu"
    }, error = function(e) NULL)
  }
  if (!is.null(device)) {
    cli::cli_alert_info("Compute device: {.val {device}}")
  }

  # backend import
  backend_ok <- tryCatch({
    .get_py()
    TRUE
  }, error = function(e) FALSE)
  if (backend_ok) {
    cli::cli_alert_success("conflibertR Python backend loads correctly")
  } else {
    cli::cli_alert_danger("conflibertR Python backend failed to load")
  }

  ok <- env_found && all(packages) && backend_ok
  if (ok) {
    cli::cli_alert_success(cli::style_bold("Everything looks good."))
  } else {
    cli::cli_text(cli::col_grey(
      "# fix the items above, then run conflibert_status() again"
    ))
  }

  invisible(list(
    env_found = env_found, python = python,
    packages = packages, device = device, ok = ok
  ))
}
