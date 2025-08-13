# Install if needed
options(repos = c(CRAN = "https://cloud.r-project.org"))

install.packages(c("haven", "dplyr", "tidyr"))

# Load libraries
library(haven)   # for reading .dta
library(dplyr)   # for data manipulation
library(tidyr)   # for completing cartesian products


# Read Stata file
export_data <- read_dta("applications/firms_export/clean_data/datasets/MEX.dta")
# export_data <- read_dta("applications/firms_export/clean_data/datasets/BGR.dta")

export_data <- export_data %>% select(-q, -c)
export_data <- export_data %>% filter(y == 2006, d != "OTH") %>% select(-y)
dim(export_data)

# Keep only top 20 destinations by count (number of firms exporting to them)
top_d <- export_data %>%
  count(d, sort = TRUE) %>%
  slice_head(n = 100) %>%
  pull(d)

export_data <- export_data %>% filter(d %in% top_d)


# Sum revenue by firm and destination
grouped_df <- export_data %>%
  group_by(f, d) %>%
  summarise(r = sum(v, na.rm = TRUE), .groups = "drop")

all_firms <- unique(grouped_df$f)
all_destinations <- unique(grouped_df$d)

cat("Number of firms:", length(all_firms), "\n")
cat("Number of destinations:", length(all_destinations), "\n")

# Get revenue for each firm-destination pair, filling in missing pairs with 0
full_grid <- expand.grid(f = all_firms, d = all_destinations)

df <- full_grid %>%
  left_join(grouped_df, by = c("f", "d")) %>%
  mutate(r = ifelse(is.na(r), 0, r))

df$f <- as.factor(df$f)
df$d <- as.factor(df$d)

# Poisson regression model
library(fixest)
model <- feglm(r ~ 1 | f + d, family = "poisson", data = df)
cat("\n===== MODEL SUMMARY BELOW =====\n")
print(summary(model))
df$predicted_revenue <- predict(model)

write.csv(df, "applications/firms_export/clean_data/poisson_results_MEX.csv")
# write.csv(df, "applications/firms_export/clean_data/poisson_results_BGR.csv")
