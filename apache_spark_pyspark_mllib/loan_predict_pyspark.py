



import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


# Resolve base directory of this script
base_dir = os.path.abspath(os.path.dirname(__file__))

# Construct full input and output paths
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("loandata_RandomForest").getOrCreate()

# Step 2: Load dataset
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Step 3: Clean column names
df = df.toDF(*[c.strip().lower() for c in df.columns])

# Step 4: Inspect columns
print("Available columns:", df.columns)
df.show(5)

# Step 5: Define feature columns
categorical_columns = ['gender', 'married', 'education']
numerical_columns = ['applicantincome', 'coapplicantincome', 'loanamount']

# Step 6: Index categorical and label columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed") for col in categorical_columns]
label_indexer = StringIndexer(inputCol="loan_status", outputCol="label")

# Step 7: Assemble features
feature_cols = [col + "_indexed" for col in categorical_columns] + numerical_columns
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Step 8: Common preprocessing steps
preprocessing_stages = indexers + [label_indexer, assembler]

# Step 9: Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 10: Define models
models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)
}

# Step 11: Evaluate each model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Output file to save accuracy scores
output_path = "output.txt"
with open(output_path, "w") as f:
    for name, classifier in models.items():
        print(f"\nTraining: {name}")
        f.write(f"\n{name}:\n")
        
        # Build pipeline
        pipeline = Pipeline(stages=preprocessing_stages + [classifier])
        
        # Train model
        model = pipeline.fit(train_data)
        
        # Predict
        predictions = model.transform(test_data)
        
        # Evaluate
        accuracy = evaluator.evaluate(predictions)
        print(f"{name} Accuracy: {accuracy:.4f}")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        
        # Optional: Save predictions
        predictions.select("label", "prediction").write.csv(f"{name.replace(' ', '_')}_predictions.csv", header=True, mode="overwrite")

# Step 12: Stop Spark session
spark.stop()


