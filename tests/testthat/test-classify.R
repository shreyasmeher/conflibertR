test_that("conflibert_classify returns a classed tibble with stable columns", {
  with_fake_py({
    r <- conflibert_classify(c("A bomb exploded.", "Sunny day."))
    expect_s3_class(r, "conflibert_classify")
    expect_s3_class(r, "tbl_df")
    expect_equal(nrow(r), 2L)
    expect_named(r, c("text", "label", "class", "confidence",
                      "prob_negative", "prob_positive"))
    expect_type(r$class, "integer")
    expect_type(r$confidence, "double")
  })
})

test_that("conflibert_classify validates input", {
  expect_error(conflibert_classify(1:3))
  expect_error(conflibert_classify(character(0)))
})

test_that("conflibert_multilabel single text drops doc_id", {
  with_fake_py({
    r <- conflibert_multilabel("Insurgents kidnapped two aid workers.")
    expect_s3_class(r, "conflibert_multilabel")
    expect_false("doc_id" %in% names(r))
    expect_equal(nrow(r), 4L)
    expect_equal(r$predicted, r$probability >= 0.5)
  })
})

test_that("conflibert_multilabel keeps doc_id for multiple texts", {
  with_fake_py({
    r <- conflibert_multilabel(c("Text one.", "Text two."))
    expect_equal(unique(r$doc_id), c(1L, 2L))
    expect_equal(nrow(r), 8L)
    # rows stay aligned text-to-doc
    expect_equal(unique(r$text[r$doc_id == 2L]), "Text two.")
  })
})
