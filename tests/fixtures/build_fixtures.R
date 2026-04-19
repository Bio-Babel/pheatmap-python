#!/usr/bin/env Rscript
# Regenerate test fixtures that encode R/pheatmap 1.0.13 reference values.
#
# Outputs (all under this script's directory):
#   test_matrix.npz                -- deterministic 20x10 double matrix
#   hclust_complete_euclidean.npz  -- hclust(dist(.), "complete")
#   hclust_correlation_rows.npz    -- pheatmap's correlation distance, then "complete"
#   hclust_minkowski_rows.npz      -- dist(., "minkowski", p=3) then "average"
#   annotation_palette_seed3453.json -- dscale(range(1..N), hue_pal(l=75)) with set.seed(3453)

suppressPackageStartupMessages({
  library(pheatmap)
  library(jsonlite)
  library(reticulate)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
out_dir <- if (length(script_arg) > 0) dirname(normalizePath(script_arg[1])) else getwd()
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- Deterministic test matrix ----------------------------------------------
set.seed(1)
mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
rownames(mat) <- paste0("r", seq_len(20))
colnames(mat) <- paste0("c", seq_len(10))

np <- import("numpy")

savez <- function(file, ...) {
  args <- list(...)
  do.call(np$savez_compressed, c(list(file = file), args))
}

savez(
  file.path(out_dir, "test_matrix.npz"),
  matrix = mat,
  rownames = rownames(mat),
  colnames = colnames(mat)
)

# --- hclust complete euclidean ----------------------------------------------
hc <- hclust(dist(mat), method = "complete")
savez(
  file.path(out_dir, "hclust_complete_euclidean.npz"),
  merge = hc$merge,
  height = hc$height,
  order = hc$order
)

# --- hclust with pheatmap's correlation distance ----------------------------
# pheatmap uses   as.dist(1 - cor(t(mat)))   for rows
d_cor <- as.dist(1 - cor(t(mat)))
hc_cor <- hclust(d_cor, method = "complete")
savez(
  file.path(out_dir, "hclust_correlation_rows.npz"),
  merge = hc_cor$merge,
  height = hc_cor$height,
  order = hc_cor$order
)

# --- hclust with manhattan + average (checks non-Euclidean metric + method) ---
d_m <- dist(mat, method = "manhattan")
hc_m <- hclust(d_m, method = "average")
savez(
  file.path(out_dir, "hclust_minkowski_rows.npz"),  # kept name for fixture stability
  merge = hc_m$merge,
  height = hc_m$height,
  order = hc_m$order
)

# --- Annotation palette, reproducing set.seed(3453) -------------------------
set.seed(3453)
pal5 <- dscale(factor(seq_len(5)), hue_pal(l = 75))
pal8 <- dscale(factor(seq_len(8)), hue_pal(l = 75))

writeLines(
  toJSON(list(five = as.character(pal5), eight = as.character(pal8)), auto_unbox = FALSE, pretty = TRUE),
  con = file.path(out_dir, "annotation_palette_seed3453.json")
)

cat("Wrote fixtures to", out_dir, "\n")
