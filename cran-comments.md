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

- local: macOS, R 4.4.1
- win-builder: R-devel and R-release
- (add R-hub results here if used)

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Shreyas Meher <shreyasmeher@gmail.com>'
  New submission

The only NOTE is the expected "New submission". A local-only
"unable to verify current time" NOTE can appear on machines with an
unsynchronized clock and does not occur on CRAN's infrastructure.

## Downstream dependencies

There are no reverse dependencies (new submission).
