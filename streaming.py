# ------------------------------------------------------------
# SECTION 6: Structured Streaming pipeline
# ------------------------------------------------------------
# We use foreachBatch to retrain the best model on each new batch
# of patient data arriving as CSV files in a directory.
#
# Each new CSV file dropped into 'streaming_data/' becomes
# one micro-batch — the model retrains and saves automatically.

from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType
)

print("\n=== Starting Structured Streaming Pipeline ===")

# Define the schema explicitly as taught in the lecture
# (streaming requires a schema — inferSchema not available)
schema = StructType([
    StructField("Pregnancies", IntegerType(), True),
    StructField("Glucose", DoubleType(), True),
    StructField("BloodPressure", DoubleType(), True),
    StructField("SkinThickness", DoubleType(), True),
    StructField("Insulin", DoubleType(), True),
    StructField("BMI", DoubleType(), True),
    StructField("DiabetesPedigreeFunction", DoubleType(), True),
    StructField("Age", DoubleType(), True),
    StructField("Outcome", IntegerType(), True)
])

# Create the streaming DataFrame from the monitored directory
# Each new CSV file dropped here becomes one micro-batch
stream_df = spark.readStream \
    .schema(schema) \
    .option("header", True) \
    .csv("hdfs://m1:9000/streaming_data/")

# Path to save the updated model after each batch
model_save_path = "hdfs://m1:9000/models/diabetes_best_model"

# Define the function applied to each micro-batch

def train_on_batch(batch_df, batch_id):
    """
    Called automatically for each new micro-batch.
    Preprocesses the batch, trains the best model pipeline,
    evaluates, and saves the updated model.
    """
    print(f"\nProcessing batch {batch_id} — row count: {batch_df.count()}")

    # Skip empty batches
    if batch_df.count() == 0:
        print(f"Batch {batch_id} is empty, skipping.")
        return

    # Replace zero values with column means (same as batch pipeline)
    for c in zero_cols:
        col_mean = batch_df.filter(col(c) != 0) \
                           .select(mean(col(c))).first()[0]
        if col_mean is not None:
            batch_df = batch_df.withColumn(
                c, when(col(c) == 0, col_mean).otherwise(col(c))
            )

    # Assemble features
    batch_assembled = assembler.transform(batch_df)
    batch_final = batch_assembled.select('features', 'Outcome')

    # Split into train and test within this batch
    (batch_train, batch_test) = batch_final.randomSplit([0.8, 0.2], seed=42)

    # Use a Pipeline combining assembler and GBT
    # (Random Forest was identified as best model in batch evaluation)
    rf_stream = RandomForestClassifier(
        labelCol='Outcome',
        featuresCol='features',
        numTrees=100,
        maxDepth=5,
        stepSize=0.1
    )

    # Fit model on this batch
    batch_model = rf_stream.fit(batch_train)

    # Evaluate on the batch test split
    batch_results = evaluate_model(
        batch_model, batch_test,
        f"Random Forest Streaming Batch {batch_id}"
    )

    # Save the updated model — overwrite keeps only latest version
    batch_model.write().overwrite().save(model_save_path)
    print(f"Batch {batch_id} model saved to {model_save_path}")


# Start the streaming query as taught in the lecture
# foreachBatch calls train_on_batch for every micro-batch
query = stream_df.writeStream \
    .foreachBatch(train_on_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "hdfs://m1:9000/checkpoints/diabetes/") \
    .start()

print("Streaming pipeline started. Waiting for new data...")

# Keep the query running until manually stopped
query.awaitTermination()
