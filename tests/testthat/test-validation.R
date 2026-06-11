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
