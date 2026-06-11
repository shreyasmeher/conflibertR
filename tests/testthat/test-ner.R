test_that("conflibert_ner returns entities with scores and offsets", {
  with_fake_py({
    r <- conflibert_ner("Kabul was quiet.")
    expect_s3_class(r, "conflibert_ner")
    expect_false("doc_id" %in% names(r))
    expect_named(r, c("entity", "label", "score", "start", "end"))
    expect_equal(r$entity, "Kabu")
    expect_equal(substr("Kabul was quiet.", r$start, r$end), r$entity)
  })
})

test_that("conflibert_ner keeps doc_id for multiple texts", {
  with_fake_py({
    r <- conflibert_ner(c("Kabul was quiet.", "Mosul too."))
    expect_equal(r$doc_id, c(1L, 2L))
    expect_equal(attr(r, "texts"), c("Kabul was quiet.", "Mosul too."))
  })
})

test_that("conflibert_ner handles texts with no entities", {
  with_fake_py(py = empty_ner_py(), {
    r <- conflibert_ner(c("nothing here", "or here"))
    expect_s3_class(r, "conflibert_ner")
    expect_equal(nrow(r), 0L)
    expect_true(all(c("entity", "label", "score", "start", "end")
                    %in% names(r)))
  })
})

test_that("conflibert_ner validates input", {
  expect_error(conflibert_ner(NULL))
  expect_error(conflibert_ner(list("a")))
})
