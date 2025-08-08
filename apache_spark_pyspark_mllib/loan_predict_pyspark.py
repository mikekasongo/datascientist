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
spark = SparkSession.builder.appName("LoanClassificationModels").getOrCreate()

# Step 2: Load dataset
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Step 3: Drop unnecessary columns
df = df.drop("Education", "Self_Employed", "Loan_ID")

# Step 4: Drop rows with null values
df = df.dropna()

# Step 5: Encode categorical variables
gender_indexer = StringIndexer(inputCol="Gender", outputCol="Gender_Indexed")
married_indexer = StringIndexer(inputCol="Married", outputCol="Married_Indexed")
property_indexer = StringIndexer(inputCol="Property_Area", outputCol="Property_Area_Indexed")
label_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label")

# Step 6: Assemble features
assembler = VectorAssembler(
    inputCols=[
        "Gender_Indexed", "Married_Indexed", "Dependents", "ApplicantIncome",
        "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area_Indexed"
    ],
    outputCol="features"
)

# Step 7: Combine preprocessing steps
preprocessing_stages = [
    gender_indexer,
    married_indexer,
    property_indexer,
    label_indexer,
    assembler
]

# Step 8: Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 9: Define classifiers
models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)
}

# Step 10: Evaluation setup
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Step 11: Train, predict, and evaluate
with open(output_path, "w") as f:
    for name, classifier in models.items():
        print(f"\nTraining: {name}")
        f.write(f"\n{name}:\n")

        pipeline = Pipeline(stages=preprocessing_stages + [classifier])
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)

        accuracy = evaluator.evaluate(predictions)
        print(f"{name} Accuracy: {accuracy:.4f}")
        f.write(f"Accuracy: {accuracy:.4f}\n")

        # Optional: Save predictions to CSV
        predictions.select("label", "prediction").write.csv(
            os.path.join(base_dir, f"{name.replace(' ', '_')}_predictions.csv"),
            header=True,
            mode="overwrite"
        )

# Step 12: Stop Spark session
spark.stop()

