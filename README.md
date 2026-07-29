Amazon Multi-Domain Review Sentiment Analysis — Portfolio Project
1. `Amazon_Reviews_Sentiment.ipynb` — the full, executed notebook: business context →
   EDA → text preprocessing (stopword removal + stemming) → TF-IDF vectorization →
   modeling (Logistic Regression + Gradient Boosted Trees) → evaluation → per-category
   business insight → conclusion & limitations.

2. `bigquery_ml_scripts.sql` — the equivalent BigQuery ML SQL (CREATE MODEL,
   ML.EVALUATE, ML.CONFUSION_MATRIX, ML.WEIGHTS)

4. `Amazon_Reviews_Sentiment_Summary.pptx` — 7-slide executive summary (business
   context → data & method → EDA finding → model results → business insight → 
   conclusion & limitations), built from the notebook's actual output numbers.

Key results 
- Logistic Regression: 67.6% accuracy · Gradient Boosted Trees: 63.6% accuracy
- The "neutral" class is consistently hardest to classify for both models (F1 ≈ 0.18–0.28)
- All three categories (book/electronics/beauty) share an identical sentiment split
  (29.8% / 21.1% / 49.1%) by construction of the dataset — so the useful category-level
  differentiator is model accuracy (book easiest at 69.9%, beauty hardest at 65.3%) and
  the actual negative-review keywords, not raw negative rate.
