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

test_that(".al_kmeans_centers is deterministic and never touches the RNG", {
  centers_of <- conflibertR:::.al_kmeans_centers
  emb <- matrix(c(
    0,     0,
    0.1,   0,
    10,   10,
    10.1, 10,
    -10,   5,
    -10.2, 5.1
  ), ncol = 2, byrow = TRUE)

  # deterministic: same input -> same centers, no seeding needed
  expect_identical(centers_of(emb, 3L), centers_of(emb, 3L))

  # anchored at the top-uncertainty candidate (row 1), k rows returned,
  # and the centers are distinct points from the input
  cc <- centers_of(emb, 3L)
  expect_equal(nrow(cc), 3L)
  expect_equal(cc[1, ], emb[1, ])
  expect_equal(nrow(unique(cc)), 3L)

  # the caller's RNG stream is not consumed or altered
  set.seed(999)
  before <- .Random.seed
  invisible(centers_of(emb, 3L))
  expect_identical(.Random.seed, before)

  # fewer distinct rows than k: returns only the distinct centers
  flat <- matrix(1, nrow = 4, ncol = 2)
  expect_equal(nrow(centers_of(flat, 3L)), 1L)
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
