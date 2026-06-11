ctx <- "The ceasefire was signed in Geneva on March 15th."

test_that("conflibert_qa keeps the legacy scalar-in, string-out contract", {
  with_fake_py({
    r <- conflibert_qa(ctx, "Where was the ceasefire signed?")
    expect_identical(r, "Geneva")
  })
})

test_that("conflibert_qa is vectorized with recycling", {
  with_fake_py({
    r <- conflibert_qa(ctx, c("Where?", "When?"))
    expect_identical(r, c("Geneva", "Geneva"))
  })
})

test_that("conflibert_qa details returns a tibble with span info", {
  with_fake_py({
    r <- conflibert_qa(ctx, c("Where?", "When?"), details = TRUE)
    expect_s3_class(r, "tbl_df")
    expect_named(r, c("question", "answer", "score", "start", "end",
                      "context"))
    expect_equal(nrow(r), 2L)
    expect_equal(substr(r$context[1], r$start[1], r$end[1]), r$answer[1])
  })
})

test_that("conflibert_qa rejects incompatible lengths", {
  expect_error(
    conflibert_qa(c("a", "b", "c"), c("q1", "q2")),
    "same length"
  )
})
