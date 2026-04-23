#!/usr/bin/env Rscript
# Regenerate the README screenshots for the active-learning workflow.
#
#   Rscript tools/make_screenshots.R
#
# Produces:
#   man/figures/active-learning-plot.png    # plot(session) — two panels
#   man/figures/active-learning-gadget.png  # conflibert_active_label() UI
#
# The plot is built from synthetic metrics so this script runs in
# seconds without needing the Python backend. The gadget screenshot is
# taken by launching the gadget UI as a standalone Shiny app in a
# background R process and capturing it with webshot2 (headless Chrome).

suppressPackageStartupMessages({
  stopifnot(requireNamespace("tibble", quietly = TRUE))
  stopifnot(requireNamespace("shiny", quietly = TRUE))
  stopifnot(requireNamespace("miniUI", quietly = TRUE))
  stopifnot(requireNamespace("webshot2", quietly = TRUE))
  stopifnot(requireNamespace("callr", quietly = TRUE))
  devtools::load_all(".", quiet = TRUE)
})

fig_dir <- file.path("man", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------
# 1. Learning-curve plot from a fake but realistic session
# ---------------------------------------------------------------------

fake <- list(
  round = 5L,
  query = tibble::tibble(text = character(0), uncertainty = numeric(0)),
  metrics = tibble::tibble(
    round            = 0:4,
    train_size       = c(20L, 30L, 40L, 50L, 60L),
    accuracy         = c(0.55, 0.70, 0.82, 0.88, 0.91),
    precision        = c(0.52, 0.68, 0.81, 0.87, 0.90),
    recall           = c(0.45, 0.66, 0.79, 0.86, 0.92),
    f1               = c(0.48, 0.67, 0.80, 0.86, 0.91),
    uncertainty_mean = c(0.69, 0.61, 0.52, 0.43, 0.34),
    uncertainty_max  = c(0.69, 0.66, 0.58, 0.50, 0.41)
  ),
  labeled_n = 60L, pool_n = 1L, done = FALSE,
  .state = list(params = list(
    model = "ConfliBERT", task = "binary",
    strategy = "entropy", query_size = 10L
  ))
)
class(fake) <- "conflibert_al_session"

plot_path <- file.path(fig_dir, "active-learning-plot.png")
grDevices::png(plot_path, width = 1600, height = 1200, res = 200)
plot(fake)
grDevices::dev.off()
message("wrote ", plot_path)

# ---------------------------------------------------------------------
# 2. Gadget screenshot via headless Chrome
# ---------------------------------------------------------------------
# We rebuild the gadget UI as a standalone Shiny app (no gadget title
# bar, since runGadget supplies that via RStudio's modal chrome). We
# render the same rows / radio buttons the real gadget does, launch
# the app in a background R process, then screenshot the rendered page.

sample_query <- tibble::tibble(
  text = c(
    "A car bomb exploded near a military checkpoint killing at least twelve soldiers",
    "The oceanographic institute published research on coral reef restoration",
    "Mortar shells struck a residential district, destroying several buildings",
    "Annual tourism numbers reached an all-time high at the coastal resorts",
    "Armed clashes between government forces and insurgents left dozens dead",
    "The symphony orchestra performed a new composition to critical acclaim",
    "A suicide bomber detonated explosives at a crowded marketplace",
    "Researchers announced a breakthrough in renewable energy storage",
    "Sniper fire killed two civilians in the besieged neighborhood",
    "The city council approved funding for a new public library"
  ),
  uncertainty = c(0.693, 0.691, 0.688, 0.685, 0.681,
                  0.676, 0.674, 0.670, 0.667, 0.663)
)

app_dir <- tempfile("al_gadget_app_")
dir.create(app_dir)
saveRDS(sample_query, file.path(app_dir, "query.rds"))

writeLines(c(
  'library(shiny)',
  'query <- readRDS("query.rds")',
  'n <- nrow(query)',
  'ui <- fluidPage(',
  '  tags$head(tags$style(HTML("',
  '    body { font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif;',
  '           background: #fff; margin: 0; }',
  '    .al-header { background: #111827; color: #fff; padding: 12px 18px;',
  '                 font-size: 15px; font-weight: 600;',
  '                 display: flex; justify-content: space-between; align-items: center; }',
  '    .al-submit { background: #2563eb; color: #fff; border: none;',
  '                 padding: 6px 14px; border-radius: 4px; font-weight: 600; cursor: pointer; }',
  '    .al-progress { padding: 10px 18px; background: #f3f4f6;',
  '                   border-bottom: 1px solid #e5e7eb; font-size: 13px; color: #374151; }',
  '    .al-row { padding: 12px 18px; border-bottom: 1px solid #e5e7eb; }',
  '    .al-row:hover { background: #f9fafb; }',
  '    .al-idx { color: #9ca3af; font-weight: 600; min-width: 28px; display: inline-block; }',
  '    .al-text { font-size: 14px; line-height: 1.45; color: #111827; }',
  '    .al-meta { color: #9ca3af; font-size: 11px; margin-top: 4px; }',
  '    .shiny-options-group { margin-top: 2px; }',
  '    .shiny-options-group label { font-weight: 500; margin-right: 16px; }',
  '  "))),',
  '  div(class = "al-header",',
  '      span("Label 10 samples (round 2)"),',
  '      tags$button(class = "al-submit", "Submit")),',
  '  div(class = "al-progress",',
  '      "Labeled 3 of 10 — Click Submit when every row has a choice."),',
  '  uiOutput("rows")',
  ')',
  'server <- function(input, output, session) {',
  '  preselect <- c("1", "0", "1", "", "", "", "", "", "", "")',
  '  output$rows <- renderUI({',
  '    rows <- lapply(seq_len(n), function(i) {',
  '      div(class = "al-row",',
  '        fluidRow(',
  '          column(8,',
  '            div(style = "display:flex; gap:10px;",',
  '              span(class = "al-idx", paste0(i, ".")),',
  '              div(',
  '                div(class = "al-text", query$text[i]),',
  '                div(class = "al-meta",',
  '                    sprintf("uncertainty: %.3f", query$uncertainty[i]))',
  '              )',
  '            )',
  '          ),',
  '          column(4,',
  '            radioButtons(paste0("lbl_", i), NULL,',
  '              choiceNames = c("0", "1"), choiceValues = c("0", "1"),',
  '              selected = if (nzchar(preselect[i])) preselect[i] else character(0),',
  '              inline = TRUE)',
  '          )',
  '        )',
  '      )',
  '    })',
  '    do.call(tagList, rows)',
  '  })',
  '}',
  'shinyApp(ui, server)'
), con = file.path(app_dir, "app.R"))

port <- httpuv::randomPort()
bg <- callr::r_bg(
  function(app_dir, port) {
    shiny::runApp(app_dir, port = port, launch.browser = FALSE,
                  host = "127.0.0.1", quiet = TRUE)
  },
  args = list(app_dir = app_dir, port = port)
)

# give the app a moment to start
Sys.sleep(3)

gadget_path <- file.path(fig_dir, "active-learning-gadget.png")
ok <- FALSE
tryCatch({
  webshot2::webshot(
    url = sprintf("http://127.0.0.1:%d", port),
    file = gadget_path,
    vwidth = 900, vheight = 800,
    delay = 2
  )
  ok <- file.exists(gadget_path)
}, error = function(e) {
  message("webshot2 failed: ", conditionMessage(e))
})

bg$kill()

if (ok) {
  message("wrote ", gadget_path)
} else {
  message("!! gadget screenshot NOT written; see error above")
}
