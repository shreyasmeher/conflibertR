## Resubmission

This is a resubmission of conflibertR. 0.5.1 was reviewed by Konstanze
Lauseker on 2026-07-02 (four points, all addressed below); the fixed
0.5.2 was then flagged by the incoming pretest on r-devel-windows with
one additional NOTE, also fixed:

* Pretest NOTE: "Examples with CPU (user + system) or elapsed time >
  10s" for conflibert_load and conflibert_multilabel (near-zero CPU,
  ~10s elapsed each). The time was spent in the conflibert_available()
  guard around the examples, which ran `conda env list` on a machine
  where conda is installed but the 'conflibert' environment is not
  (conda's cold start took ~10s per call; the same cost was visible in
  the [10s] "package can be loaded" steps of that log). Backend
  detection now uses filesystem checks only, never runs the conda
  binary, and caches its result for the session, so on machines
  without the environment every example returns in well under a
  second.

The four points from the 0.5.1 review:

* "please omit the redundant 'R' from the start of the description"
  Done. The Description now begins "An interface to 'ConfliBERT', ...".

* "\dontrun{} should only be used if the example really cannot be
  executed [...] Please replace \dontrun with \donttest if possible. /
  Please unwrap the examples if they are executable in < 5 sec"
  All examples were rewritten to be self-contained and are no longer
  wrapped in \dontrun{}. Because every backend function needs a
  user-provisioned Python environment ('torch', 'transformers') plus
  model downloads from 'Hugging Face', unconditional \donttest{} would
  error under --run-donttest on machines without that environment. The
  examples are therefore guarded with @examplesIf
  conflibert_available(), a new exported helper that returns TRUE only
  when the Python backend is usable: the examples execute as real code
  on any machine with the documented setup and are skipped cleanly
  elsewhere (including the CRAN check farm). Pure-R examples
  (conflibert_example(), conflibert_models(), conflibert_available())
  run unconditionally in < 5 sec. The single remaining \dontrun{} is
  conflibert_install(), which installs software (it creates a Python
  environment and downloads several GB), so it must never run
  unattended; a comment in the example says so.

* "Please do not install packages in your functions, examples or
  vignette."
  Nothing is installed by any function, example, test, or vignette.
  The only installation code in the package is the dedicated setup
  function conflibert_install(), which users must call explicitly (the
  same pattern as keras::install_keras()); its example is \dontrun{}
  and both vignettes are knitted with eval = FALSE, so no check or
  build ever executes it.

* "Please do not modify the .GlobalEnv."
  Fixed. The only such code saved and restored .Random.seed around a
  seeded stats::kmeans() call. That k-means is now started from
  deterministic farthest-first centers, so the package no longer calls
  set.seed() and no longer reads or writes .Random.seed or anything
  else in the global environment.

## Submission

conflibertR is an interface to the 'ConfliBERT' family of language
models for conflict and political-violence text. The package calls a
'Python' backend (via 'reticulate') for inference and fine-tuning; the
required Python interpreter and modules are declared in
SystemRequirements and installed by the user with conflibert_install().
The test suite runs against a mocked backend (no Python or network),
and both vignettes are non-evaluated, so R CMD check never downloads a
model or starts Python.

## Test environments

- local: macOS (aarch64), R 4.4.1
- GitHub Actions: macOS-latest (release), windows-latest (release),
  ubuntu-latest (R-devel, release, oldrel-1)
- win-builder: R-devel and R-release

## R CMD check results

0 errors | 0 warnings | 1 note

The single NOTE is "CRAN incoming feasibility" (new submission;
possibly misspelled words in DESCRIPTION: "Hu", "et", "al",
"multilabel", "pretrained"). These are spelled correctly: "Hu" is an
author surname, "et al" is the standard Latin citation form, and
"multilabel"/"pretrained" are technical terms. They are listed in
inst/WORDLIST.

(Two additional NOTEs appear only on the local macOS machine and not
on CRAN's setup: "unable to verify current time" from an
unsynchronized clock, and HTML validation warnings from the outdated
HTML Tidy that ships with macOS.)

## Downstream dependencies

There are no reverse dependencies (new submission).
