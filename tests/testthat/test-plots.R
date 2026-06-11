skip_if_not_installed("ggplot2")

test_that("autoplot methods return ggplot objects", {
  with_fake_py({
    cl <- conflibert_classify(c("Bomb exploded.", "Nice day.", "Riots."))
    expect_s3_class(ggplot2::autoplot(cl), "ggplot")

    en <- conflibert_ner(c("Kabul was quiet.", "Mosul too."))
    expect_s3_class(ggplot2::autoplot(en), "ggplot")
    expect_s3_class(ggplot2::autoplot(en, type = "entities"), "ggplot")

    ml <- conflibert_multilabel(c("Text one.", "Text two."))
    expect_s3_class(ggplot2::autoplot(ml), "ggplot")
  })
  expect_s3_class(ggplot2::autoplot(fake_comparison()), "ggplot")
  expect_s3_class(ggplot2::autoplot(fake_finetune()), "ggplot")
  expect_s3_class(ggplot2::autoplot(fake_session()), "ggplot")
  expect_s3_class(
    ggplot2::autoplot(fake_session(), which = "uncertainty"), "ggplot"
  )
})

test_that("multilabel autoplot aggregates for many texts", {
  with_fake_py({
    ml <- conflibert_multilabel(sprintf("Text %d.", 1:8))
    p <- ggplot2::autoplot(ml)
    expect_s3_class(p, "ggplot")
    expect_match(p$labels$title, "across texts")

    one <- conflibert_multilabel("Just one text.")
    p1 <- ggplot2::autoplot(one)
    expect_match(p1$labels$title, "probabilities")
  })
})

test_that("base plot methods draw without error", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_invisible(plot(fake_finetune()))
  expect_invisible(plot(fake_comparison()))
  with_fake_py({
    cl <- conflibert_classify(c("Bomb exploded.", "Nice day."))
    expect_invisible(plot(cl))
    en <- conflibert_ner(c("Kabul was quiet.", "Mosul too."))
    expect_invisible(plot(en))
    ml <- conflibert_multilabel(c("Text one.", "Text two."))
    expect_invisible(plot(ml))
    expect_invisible(plot(conflibert_multilabel("Just one text.")))
  })
  expect_invisible(plot(fake_session()))
})

test_that("theme_conflibert is a ggplot2 theme with grid options", {
  expect_s3_class(theme_conflibert(), "theme")
  expect_s3_class(theme_conflibert(grid = "x"), "theme")
  expect_s3_class(theme_conflibert(grid = "none"), "theme")
})
