# batch_ne_wrapper.R
# -------------------------------------------------------------
# Batch parser/summarizer for LDNe results from NeEstimator
# to accompany RGSM_LDNe_model_consistent_sim.py outputs.
#
# Workflow (simple & robust):
# 1) Run the Python simulator to create GENEPOP files under
#    ./synthetic_genotypes/rep_XXX/RGSM_Y{year}_rep{rep}.gen
# 2) Run NeEstimator (LD method; parametric CI) on EACH .gen file.
#    In NeEstimator, export the LD results text report for each run
#    into ./ne_results/ using the SAME basename plus .txt, e.g.:
#    ne_results/RGSM_Y2010_rep003.txt
# 3) Run this script; it parses the reports, compiles a table of
#    Ne and 95% CI by (year, replicate), then summarizes per year.
# -------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(stringr)
})

# --- Paths ---
manifest_csv <- "synthetic_genotypes/manifest.csv"
results_dir  <- "ne_results"          # where you saved the NeEstimator text outputs
out_dir      <- "summaries"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Helper: robust numeric extractor ---
num_extract <- function(x) {
  as.numeric(str_extract(x, "-?\\d+(?:[.,]\\d+)?"))
}

# --- Parser for a single NeEstimator LD results .txt file ---
# Tries to find lines containing Ne and 95% CI; returns tibble with one row.
parse_ne_file <- function(path) {
  lines <- read_lines(path, skip_empty_rows = FALSE, progress = FALSE)
  # Normalize decimal points
  lines <- str_replace_all(lines, ",", ".")

  # Find candidate lines
  ne_idx <- which(str_detect(lines, regex("(^|\\b)Ne\\b\"?\"?\s*[:=]", ignore_case = TRUE)))
  if (length(ne_idx) == 0) ne_idx <- which(str_detect(lines, regex("Effective\\s+population", ignore_case = TRUE)))
  ci_idx <- which(str_detect(lines, regex("95%.*CI|CI.*95%|Confidence\\s*Interval", ignore_case = TRUE)))

  Ne <- NA_real_
  ci_lo <- NA_real_
  ci_hi <- NA_real_

  if (length(ne_idx) > 0) {
    # take the first occurrence
    ne_line <- lines[ne_idx[1]]
    # get the first number on that line
    Ne <- suppressWarnings(num_extract(ne_line))
  }
  if (length(ci_idx) > 0) {
    ci_line <- lines[ci_idx[1]]
    nums <- str_extract_all(ci_line, "-?\\d+(?:[.,]\\d+)?")[[1]]
    nums <- as.numeric(str_replace_all(nums, ",", "."))
    if (length(nums) >= 2) {
      ci_lo <- nums[1]
      ci_hi <- nums[2]
    }
  }

  tibble(Ne = Ne, ci_lo = ci_lo, ci_hi = ci_hi, report_path = path)
}

# --- Discover all result files and map to (year, rep) ---
# Expects filenames like RGSM_Y2010_rep003.txt (order not important)
list_result_files <- function(dir) {
  fs <- list.files(dir, pattern = "\\\.txt$", full.names = TRUE, recursive = TRUE)
  if (length(fs) == 0) return(tibble())
  tibble(path = fs) %>%
    mutate(
      file = basename(path),
      year = as.integer(str_extract(file, "(?<=Y)\\d{4}")),
      rep  = as.integer(str_extract(file, "(?<=rep)\\d{3}"))
    ) %>%
    filter(!is.na(year), !is.na(rep))
}

# --- Load manifest to cross-check sample sizes and ensure coverage ---
manifest <- read_csv(manifest_csv, show_col_types = FALSE) %>%
  mutate(
    year = as.integer(str_extract(genepop_path, "(?<=Y)\\d{4}")),
    rep  = as.integer(str_extract(genepop_path, "(?<=rep)\\d{3}"))
  )

message(sprintf("Manifest rows: %d", nrow(manifest)))

# --- Gather and parse all reports ---
res_index <- list_result_files(results_dir)
if (nrow(res_index) == 0) {
  stop("No NeEstimator .txt results found in ne_results/. Make sure you've exported them with names like RGSM_Y2010_rep003.txt.")
}

parsed <- res_index %>%
  mutate(parsed = map(path, parse_ne_file)) %>%
  unnest(parsed)

# Join on manifest to add n_diploids (if desired) and paths
parsed2 <- parsed %>%
  left_join(manifest %>% select(year, rep, n_diploids, genepop_path), by = c("year", "rep")) %>%
  arrange(year, rep)

# Save the raw-by-replicate table
write_csv(parsed2, file.path(out_dir, "synthetic_LDNe_by_rep.csv"))
message("Wrote per-replicate LDNe table: summaries/synthetic_LDNe_by_rep.csv")

# --- Summarize across replicates for each year ---
sum_by_year <- parsed2 %>%
  group_by(year) %>%
  summarise(
    n_rep = sum(!is.na(Ne)),
    Ne_median = median(Ne, na.rm = TRUE),
    Ne_q025   = quantile(Ne, 0.025, na.rm = TRUE),
    Ne_q975   = quantile(Ne, 0.975, na.rm = TRUE),
    # Optional: combine CIs across reps (not strictly correct, but useful)
    CI_lo_med = median(ci_lo, na.rm = TRUE),
    CI_hi_med = median(ci_hi, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(year)

write_csv(sum_by_year, file.path(out_dir, "synthetic_LDNe_by_year_summary.csv"))
message("Wrote year summary: summaries/synthetic_LDNe_by_year_summary.csv")

# --- Quick plot (optional) ---
try({
  p <- ggplot(parsed2, aes(x = factor(year), y = Ne)) +
    geom_boxplot(outlier.shape = NA) +
    geom_point(position = position_jitter(width = 0.15), alpha = 0.4, size = 1) +
    geom_point(data = sum_by_year, aes(y = Ne_median), color = "red", size = 2) +
    labs(x = "Year", y = "Synthetic LDNe (across replicates)",
         title = "Synthetic LDNe by year (replicate distribution and median)") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(out_dir, "synthetic_LDNe_by_year.png"), p, width = 10, height = 5, dpi = 150)
  message("Saved plot: summaries/synthetic_LDNe_by_year.png")
}, silent = TRUE)

# --- OPTIONAL: template to run a CLI if available ---
# If you have a command-line NeEstimator, you can adapt this stub.
# It will iterate over the manifest rows, run the CLI, and save outputs
# to results_dir with the expected filenames.
#
# run_cli <- FALSE
# ne_cli <- "C:/Program Files/NeEstimatorV2/Ne2Console.exe"  # <-- adjust if you have it
#
# if (run_cli) {
#   dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
#   man <- manifest
#   for (i in seq_len(nrow(man))) {
#     gen <- man$genepop_path[i]
#     yr  <- man$year[i]
#     rp  <- man$rep[i]
#     out <- file.path(results_dir, sprintf("RGSM_Y%04d_rep%03d.txt", yr, rp))
#     # TODO: replace the following system2 call with the actual CLI args
#     # The idea is to specify: input file, method=LD, CI=parametric, MAF/MAC filters
#     # system2(ne_cli, args = c("--input", gen, "--method", "LD", "--ci", "parametric", "--out", out))
#   }
# }
