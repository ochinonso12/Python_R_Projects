# ============================================================
# CMP9781M Big Data Analytics & Modelling
# Assessment 2 — Pima Indians Diabetes Classification
# ============================================================

# Pipeline overview:
#   1. Start Spark and load data from HDFS
#   2. Preprocess — fix zero values, assemble features
#   3. Train/test split
#   4. Train 4 models with hyperparameter tuning
#   5. Evaluate all models with multiple metrics
#   6. Structured Streaming — retrain best model on new data - streaming.py
# ============================================================


# ------------------------------------------------------------
# SECTION 1: Start Spark session and connect to cluster
# ------------------------------------------------------------
# As shown in lectures, we connect PySpark to the Spark master
# running on m1 in the Oracle NAT VM cluster environment.

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("spark://m1:7077") \
    .appName("DiabetesClassification") \
    .getOrCreate()

# Confirm the session is running
print("Spark version:", spark.version)


# ------------------------------------------------------------
# SECTION 1.2: Load data from HDFS
# ------------------------------------------------------------
# As shown in the MLlib lecture, we load CSV data stored on
# HDFS using spark.read.csv with header and schema inference.

df = spark.read.csv(
    'hdfs://m1:9000/medical_data/diabetes.csv',
    header=True,
    inferSchema=True
)

# Show the first few rows to confirm loading worked
print("Dataset loaded. Row count:", df.count())
df.show(5)
df.printSchema()


# ------------------------------------------------------------
# SECTION 2: Preprocessing
# ------------------------------------------------------------
# The dataset contains zero values in columns where zero is
# biologically impossible (e.g. Glucose=0, BMI=0).
# These represent missing values and must be replaced before
# training. We replace them with the column mean.

from pyspark.sql.functions import mean, when, col

# Columns where zero means missing data
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zero values with column mean for each affected column
for c in zero_cols:
    col_mean = df.filter(col(c) != 0).select(mean(col(c))).first()[0]
    df = df.withColumn(c, when(col(c) == 0, col_mean).otherwise(col(c)))

print("Zero values replaced with column means.")
df.show(5)


# ------------------------------------------------------------
# SECTION 2.1: Feature construction with VectorAssembler
# ------------------------------------------------------------
# As shown in the MLlib lecture, VectorAssembler combines all
# input columns into a single 'features' vector required by
# Spark MLlib models.

from pyspark.ml.feature import VectorAssembler

# All 8 predictor columns
feature_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)

# Transform the dataframe to add the features column
transformed_df = assembler.transform(df)

# Select only the columns needed for training
final_df = transformed_df.select('features', 'Outcome')
final_df.show(5)


# ------------------------------------------------------------
# SECTION 3: Train / test split
# ------------------------------------------------------------
# As shown in the MLlib lecture, we split into 80% training
# and 20% testing. seed=42 ensures reproducibility.

(train_data, test_data) = final_df.randomSplit([0.8, 0.2], seed=42)

print("Training rows:", train_data.count())
print("Testing rows:", test_data.count())


# ------------------------------------------------------------
# SECTION 3.1: Helper function — evaluate a trained model
# ------------------------------------------------------------
# This function runs predictions and computes all 5 metrics.
# We call it once per model to keep the code clean.

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

def evaluate_model(model, test_data, model_name):
    """
    Runs predictions and prints accuracy, AUC-ROC,
    F1-score, precision and recall for a trained model.
    """
    predictions = model.transform(test_data)

    # Accuracy
    acc_eval = MulticlassClassificationEvaluator(
        labelCol='Outcome',
        predictionCol='prediction',
        metricName='accuracy'
    )
    accuracy = acc_eval.evaluate(predictions)

    # F1-score
    f1_eval = MulticlassClassificationEvaluator(
        labelCol='Outcome',
        predictionCol='prediction',
        metricName='f1'
    )
    f1 = f1_eval.evaluate(predictions)

    # Precision
    prec_eval = MulticlassClassificationEvaluator(
        labelCol='Outcome',
        predictionCol='prediction',
        metricName='weightedPrecision'
    )
    precision = prec_eval.evaluate(predictions)

    # Recall
    rec_eval = MulticlassClassificationEvaluator(
        labelCol='Outcome',
        predictionCol='prediction',
        metricName='weightedRecall'
    )
    recall = rec_eval.evaluate(predictions)

    # AUC-ROC (binary evaluator)
    auc_eval = BinaryClassificationEvaluator(
        labelCol='Outcome',
        rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )
    auc = auc_eval.evaluate(predictions)

    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# ------------------------------------------------------------
# SECTION 4: Model 1 — Logistic Regression
# ------------------------------------------------------------
# As shown in the MLlib lecture, Logistic Regression uses the
# sigmoid function to predict binary class probabilities.
# CrossValidator performs 5-fold cross-validation to find
# the best hyperparameter combination.

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression

print("\n=== Training Model 1: Logistic Regression ===")

# Initialise the model
lr = LogisticRegression(
    labelCol='Outcome',
    featuresCol='features'
)

# Define the hyperparameter grid to search
lr_param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100]) \
    .build()

# Evaluator used during cross-validation
lr_evaluator = BinaryClassificationEvaluator(
    labelCol='Outcome',
    metricName='areaUnderROC'
)

# CrossValidator tries all parameter combinations
# using 5-fold cross-validation on the training set
lr_cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=lr_param_grid,
    evaluator=lr_evaluator,
    numFolds=5
)

# Fit the best model
lr_model = lr_cv.fit(train_data)

# Evaluate on test set
lr_results = evaluate_model(lr_model, test_data, "Logistic Regression")

# Alternate evaluation code since spark-submit wasn't working on the virtual machine
#  — this is the same code as in evaluate_model() but written out here to show the process step-by-step.
# predictions = lr_model.transform(test_data)

# accuracy = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='accuracy'
# ).evaluate(predictions)

# f1 = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='f1'
# ).evaluate(predictions)

# precision = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedPrecision'
# ).evaluate(predictions)

# recall = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedRecall'
# ).evaluate(predictions)

# auc = BinaryClassificationEvaluator(
#     labelCol='Outcome',
#     rawPredictionCol='rawPrediction',
#     metricName='areaUnderROC'
# ).evaluate(predictions)

# print("--- Logistic Regression Results ---")
# print("Accuracy  : " + str(round(accuracy, 4)))
# print("AUC-ROC   : " + str(round(auc, 4)))
# print("F1-score  : " + str(round(f1, 4)))
# print("Precision : " + str(round(precision, 4)))
# print("Recall    : " + str(round(recall, 4)))

# ------------------------------------------------------------
# SECTION 4.1: Model 2 — Decision Tree Classifier
# ------------------------------------------------------------
# Decision Tree recursively splits the feature space to
# minimise node impurity (Gini or entropy).
# maxDepth controls overfitting — deeper trees overfit.

from pyspark.ml.classification import DecisionTreeClassifier

print("\n=== Training Model 2: Decision Tree ===")

dt = DecisionTreeClassifier(
    labelCol='Outcome',
    featuresCol='features'
)

dt_param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [3, 5, 7, 10]) \
    .addGrid(dt.minInstancesPerNode, [1, 5]) \
    .addGrid(dt.impurity, ['gini', 'entropy']) \
    .build()

dt_evaluator = BinaryClassificationEvaluator(
    labelCol='Outcome',
    metricName='areaUnderROC'
)

dt_cv = CrossValidator(
    estimator=dt,
    estimatorParamMaps=dt_param_grid,
    evaluator=dt_evaluator,
    numFolds=5
)

dt_model = dt_cv.fit(train_data)

dt_results = evaluate_model(dt_model, test_data, "Decision Tree")

# Alternate evaluation code since spark-submit wasn't working on the virtual machine
#  — this is the same code as in evaluate_model() but written out here to show the process step-by-step.
# predictions = dt_model.transform(test_data)

# accuracy = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='accuracy'
# ).evaluate(predictions)

# f1 = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='f1'
# ).evaluate(predictions)

# precision = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedPrecision'
# ).evaluate(predictions)

# recall = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedRecall'
# ).evaluate(predictions)

# auc = BinaryClassificationEvaluator(
#     labelCol='Outcome',
#     rawPredictionCol='rawPrediction',
#     metricName='areaUnderROC'
# ).evaluate(predictions)

# print("--- Decision Tree Results ---")
# print("Accuracy  : " + str(round(accuracy, 4)))
# print("AUC-ROC   : " + str(round(auc, 4)))
# print("F1-score  : " + str(round(f1, 4)))
# print("Precision : " + str(round(precision, 4)))
# print("Recall    : " + str(round(recall, 4)))

# ------------------------------------------------------------
# SECTION 4.2: Model 3 — Random Forest Classifier
# ------------------------------------------------------------
# Random Forest builds many decision trees on random subsets
# of the data and combines their votes (bagging).
# This reduces variance compared to a single decision tree.

from pyspark.ml.classification import RandomForestClassifier

print("\n=== Training Model 3: Random Forest ===")

rf = RandomForestClassifier(
    labelCol='Outcome',
    featuresCol='features'
)

rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 2]) \
    .build()

rf_evaluator = BinaryClassificationEvaluator(
    labelCol='Outcome',
    metricName='areaUnderROC'
)

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=rf_evaluator,
    numFolds=5
)

rf_model = rf_cv.fit(train_data)

rf_results = evaluate_model(rf_model, test_data, "Random Forest")

# Alternate evaluation code since spark-submit wasn't working on the virtual machine
#  — this is the same code as in evaluate_model() but written out here to show the process step-by-step.
# predictions = rf_model.transform(test_data)

# accuracy = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='accuracy'
# ).evaluate(predictions)

# f1 = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='f1'
# ).evaluate(predictions)

# precision = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedPrecision'
# ).evaluate(predictions)

# recall = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedRecall'
# ).evaluate(predictions)

# auc = BinaryClassificationEvaluator(
#     labelCol='Outcome',
#     rawPredictionCol='rawPrediction',
#     metricName='areaUnderROC'
# ).evaluate(predictions)

# print("--- Random Forest Results ---")
# print("Accuracy  : " + str(round(accuracy, 4)))
# print("AUC-ROC   : " + str(round(auc, 4)))
# print("F1-score  : " + str(round(f1, 4)))
# print("Precision : " + str(round(precision, 4)))
# print("Recall    : " + str(round(recall, 4)))


# ------------------------------------------------------------
# SECTION 4.3: Model 4 — Gradient Boosted Tree Classifier
# ------------------------------------------------------------
# GBT builds trees sequentially — each new tree corrects
# the errors of the previous ensemble.
# stepSize (learning rate) controls each tree's contribution.

from pyspark.ml.classification import GBTClassifier

print("\n=== Training Model 4: Gradient Boosted Trees ===")

gbt = GBTClassifier(
    labelCol='Outcome',
    featuresCol='features'
)

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [10, 20, 50]) \
    .addGrid(gbt.maxDepth, [2, 3, 5]) \
    .addGrid(gbt.stepSize, [0.01, 0.05, 0.1]) \
    .build()

gbt_evaluator = BinaryClassificationEvaluator(
    labelCol='Outcome',
    metricName='areaUnderROC'
)

gbt_cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=gbt_param_grid,
    evaluator=gbt_evaluator,
    numFolds=5
)

gbt_model = gbt_cv.fit(train_data)

gbt_results = evaluate_model(gbt_model, test_data, "Gradient Boosted Trees")

# Alternate evaluation code since spark-submit wasn't working on the virtual machine
#  — this is the same code as in evaluate_model() but written out here to show the process step-by-step.
# predictions = gbt_model.transform(test_data)

# accuracy = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='accuracy'
# ).evaluate(predictions)

# f1 = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='f1'
# ).evaluate(predictions)

# precision = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedPrecision'
# ).evaluate(predictions)

# recall = MulticlassClassificationEvaluator(
#     labelCol='Outcome',
#     predictionCol='prediction',
#     metricName='weightedRecall'
# ).evaluate(predictions)

# auc = BinaryClassificationEvaluator(
#     labelCol='Outcome',
#     rawPredictionCol='rawPrediction',
#     metricName='areaUnderROC'
# ).evaluate(predictions)

# print("--- Gradient Boosted Trees Results ---")
# print("Accuracy  : " + str(round(accuracy, 4)))
# print("AUC-ROC   : " + str(round(auc, 4)))
# print("F1-score  : " + str(round(f1, 4)))
# print("Precision : " + str(round(precision, 4)))
# print("Recall    : " + str(round(recall, 4)))


# ------------------------------------------------------------
# SECTION 5: Comparison summary
# ------------------------------------------------------------
# Print all results together so they are easy to compare
# and include in the report.

print("\n=== FULL RESULTS COMPARISON ===")
print(f"{'Model':<30} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 75)

all_results = [lr_results, dt_results, rf_results, gbt_results]

for r in all_results:
    print(
        f"{r['model']:<30} "
        f"{r['accuracy']:>10.4f} "
        f"{r['auc']:>10.4f} "
        f"{r['f1']:>10.4f} "
        f"{r['precision']:>10.4f} "
        f"{r['recall']:>10.4f}"
    )

# Identify best model by AUC-ROC
best = max(all_results, key=lambda x: x['auc'])
print(f"\nBest model by AUC-ROC: {best['model']} ({best['auc']:.4f})")
