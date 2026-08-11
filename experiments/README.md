These CSVs are empty placeholders (headers only). They are populated by:

- learning_rate_results.csv  <-  python -m src.train --mode lr_search
- epoch_results.csv          <-  python -m src.train --mode epoch_search
- model_comparison.csv       <-  python -m src.evaluate --compare_all

No results are fabricated -- run the commands above to generate real numbers.
