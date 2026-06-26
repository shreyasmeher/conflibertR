## Submission

This is a new submission of conflibertR, an R interface to the 'ConfliBERT'
family of language models for conflict and political-violence text. The
package calls a 'Python' backend (via 'reticulate') for inference and
fine-tuning; the required Python interpreter and modules are declared in
SystemRequirements and installed by the user with conflibert_install().

Because the models require a user-provisioned Python environment and
network access to 'Hugging Face', the examples that touch the backend are
wrapped in \dontrun{} rather than \donttest{}: they cannot run on the CRAN
check farm (which has no 'conflibert' Python environment), so \donttest{}
would fail under --run-donttest. Examples that are pure R (e.g.
conflibert_example(), theme_conflibert()) are runnable or wrapped in
\donttest{} and do execute during the check. The test suite runs against a
mocked backend (no Python or network), and both vignettes are
non-evaluated, so R CMD check never downloads a model or starts Python.

## Test environments

- local: macOS (aarch64), R 4.4.1
- GitHub Actions: macOS-latest (release), windows-latest (release),
  ubuntu-latest (R-devel, release, oldrel-1) -- all passing
- win-builder: R-devel (2026-06-25 r90191) and R-release (4.6.1) --
  0 errors, 0 warnings, 1 note on both

## R CMD check results

0 errors | 0 warnings | 1 note

On win-builder the single NOTE is:

* checking CRAN incoming feasibility ... NOTE
  Possibly misspelled words in DESCRIPTION:
    Hu (7:59) al (7:65) et (7:62) multilabel (9:36) pretrained (6:48)

These are spelled correctly: "Hu" is an author surname, "et al" is the
standard Latin citation form, and "multilabel"/"pretrained" are technical
terms. They are listed in inst/WORDLIST.

On submission CRAN will additionally show the standard "New submission"
NOTE. (A local-only "unable to verify current time" NOTE can appear on a
machine with an unsynchronized clock and does not occur on CRAN.)

## Downstream dependencies

There are no reverse dependencies (new submission).
