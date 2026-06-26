test_that("conflibert_load rejects a missing directory", {
  expect_error(conflibert_load("/no/such/dir"), "does not exist")
})

test_that("finetune validates its data frames before touching Python", {
  bad <- data.frame(x = 1)
  good <- data.frame(text = "a", label = 0L)
  expect_error(conflibert_finetune(bad, good, good))
  expect_error(conflibert_finetune(good, bad, good))
})

test_that("active learning input checks fire", {
  expect_error(
    conflibertR:::.al_check_labeled(data.frame(text = "a"), "seed"),
    "label"
  )
  expect_error(conflibertR:::.al_check_pool(character(0)), "empty")
  expect_equal(
    conflibertR:::.al_check_pool(data.frame(text = c("a", "b"))),
    c("a", "b")
  )
})

test_that("training entry points expose a seed argument (default 42)", {
  # conflibert_active_start already uses `seed` for the labeled starter
  # set, so its reproducibility knob is named `random_seed`.
  knobs <- list(
    conflibert_finetune     = "seed",
    conflibert_compare      = "seed",
    conflibert_active_start = "random_seed"
  )
  for (nm in names(knobs)) {
    fmls <- formals(get(nm))
    arg <- knobs[[nm]]
    expect_true(arg %in% names(fmls), info = nm)
    expect_equal(eval(fmls[[arg]]), 42, info = nm)
  }
})

test_that(".al_with_seed is reproducible and restores the RNG state", {
  with_seed <- conflibertR:::.al_with_seed

  # same seed -> same draw
  a <- with_seed(123, runif(3))
  b <- with_seed(123, runif(3))
  expect_equal(a, b)

  # NULL seed leaves expr untouched (just returns its value)
  expect_equal(with_seed(NULL, 42L), 42L)

  # the caller's global RNG stream is not disturbed
  set.seed(999)
  before <- .Random.seed
  invisible(with_seed(7, runif(5)))
  expect_identical(.Random.seed, before)
  # and the next draw matches what it would have been with no seeded call
  set.seed(999)
  expected_next <- runif(2)
  set.seed(999)
  invisible(with_seed(7, runif(5)))
  expect_equal(runif(2), expected_next)
})

test_that("conflibert_models lists the supported architectures", {
  m <- conflibert_models()
  expect_true("ConfliBERT" %in% m)
  expect_length(m, 7L)
})

test_that("example datasets load with the documented shape", {
  d <- conflibert_example("binary")
  expect_named(d, c("train", "dev", "test"))
  expect_true(all(c("text", "label") %in% names(d$train)))

  a <- conflibert_example("active")
  expect_named(a, c("seed", "pool", "dev", "pool_labels"))
  expect_type(a$pool, "character")
})
