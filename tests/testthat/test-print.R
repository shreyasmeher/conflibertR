test_that("classify results print without error", {
  with_fake_py({
    r <- conflibert_classify(c("A bomb exploded.", "Sunny day."))
    expect_output(print(r), "ConfliBERT classification")
  })
})

test_that("ner results print with entity labels", {
  with_fake_py({
    r <- conflibert_ner(c("Kabul was quiet.", "Mosul too."))
    expect_output(print(r), "ConfliBERT entities")
  })
  with_fake_py(py = empty_ner_py(), {
    r <- conflibert_ner("nothing")
    expect_output(print(r), "No entities found")
  })
})

test_that("multilabel results print without error", {
  with_fake_py({
    r <- conflibert_multilabel("Insurgents kidnapped two aid workers.")
    expect_output(print(r), "ConfliBERT event types")
  })
})

test_that("comparison results print ranked", {
  expect_output(print(fake_comparison()), "Model comparison")
})

test_that("finetune results print metrics", {
  expect_output(print(fake_finetune()), "Fine-tuned model")
})

test_that("active learning session prints", {
  expect_output(print(fake_session()), "Active learning session")
})
