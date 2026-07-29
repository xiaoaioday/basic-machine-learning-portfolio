CREATE OR REPLACE TABLE `DA04.amazon_reviews_tfidf_split` AS
WITH numbered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY label ORDER BY RAND()) AS rn,
    COUNT(*) OVER (PARTITION BY label) AS label_total
  FROM `DA04.amazon_reviews_tfidf`
)
SELECT
  * EXCEPT(rn, label_total),
  CASE WHEN rn <= 0.8 * label_total THEN 'training' ELSE 'evaluation' END AS split_field
FROM numbered;


-- ---------------------------------------------------------------------
-- STEP 1: Model 1 — Logistic Regression (multiclass)
-- ---------------------------------------------------------------------
CREATE OR REPLACE MODEL `DA04.model_logreg_sentiment`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['label'],
  auto_class_weights = TRUE   -- labels are imbalanced (29.8% / 21.1% / 49.1%)
) AS
SELECT * EXCEPT(review_id, product_category, split_field)
FROM `DA04.amazon_reviews_tfidf_split`
WHERE split_field = 'training';


-- ---------------------------------------------------------------------
-- STEP 2: Model 2 — Boosted Tree Classifier (comparison model)
-- ---------------------------------------------------------------------
CREATE OR REPLACE MODEL `DA04.model_boostedtree_sentiment`
OPTIONS(
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['label'],
  max_iterations = 50
) AS
SELECT * EXCEPT(review_id, product_category, split_field)
FROM `DA04.amazon_reviews_tfidf_split`
WHERE split_field = 'training';


-- ---------------------------------------------------------------------
-- STEP 3: Evaluate both models on the held-out evaluation split
-- ---------------------------------------------------------------------
SELECT * FROM ML.EVALUATE(
  MODEL `DA04.model_logreg_sentiment`,
  (SELECT * EXCEPT(review_id, product_category, split_field)
   FROM `DA04.amazon_reviews_tfidf_split`
   WHERE split_field = 'evaluation')
);

SELECT * FROM ML.EVALUATE(
  MODEL `DA04.model_boostedtree_sentiment`,
  (SELECT * EXCEPT(review_id, product_category, split_field)
   FROM `DA04.amazon_reviews_tfidf_split`
   WHERE split_field = 'evaluation')
);


-- ---------------------------------------------------------------------
-- STEP 4: Confusion matrix — check specifically how often the 'neutral'
-- class (label = 1) gets confused with negative/positive
-- ---------------------------------------------------------------------
SELECT * FROM ML.CONFUSION_MATRIX(
  MODEL `DA04.model_logreg_sentiment`,
  (SELECT * EXCEPT(review_id, product_category, split_field)
   FROM `DA04.amazon_reviews_tfidf_split`
   WHERE split_field = 'evaluation')
);


-- ---------------------------------------------------------------------
-- STEP 5: Model interpretability — top terms per class
-- (equivalent to inspecting Logistic Regression .coef_ in scikit-learn)
-- ---------------------------------------------------------------------
SELECT
  processed_input AS term,
  class_label,
  weight
FROM ML.WEIGHTS(MODEL `DA04.model_logreg_sentiment`)
CROSS JOIN UNNEST(category_weights)
ORDER BY class_label, weight DESC;


-- ---------------------------------------------------------------------
-- STEP 6: Per-category accuracy — join predictions back to product_category
-- (ML.EVALUATE alone drops the metadata columns, so predict separately here)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE `DA04.eval_predictions_by_category` AS
SELECT
  e.product_category,
  e.label AS actual_label,
  p.predicted_label
FROM (
  SELECT *
  FROM `DA04.amazon_reviews_tfidf_split`
  WHERE split_field = 'evaluation'
) e,
UNNEST([STRUCT(
  (SELECT predicted_label
   FROM ML.PREDICT(MODEL `DA04.model_logreg_sentiment`, (SELECT e.* EXCEPT(review_id, product_category, split_field)))
  ) AS predicted_label
)]) p;

SELECT
  product_category,
  COUNTIF(actual_label = predicted_label) / COUNT(*) AS accuracy,
  COUNT(*) AS n
FROM `DA04.eval_predictions_by_category`
GROUP BY product_category
ORDER BY accuracy DESC;

