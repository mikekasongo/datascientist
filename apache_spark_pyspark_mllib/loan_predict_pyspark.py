import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Base directory of this script
base_dir = os.path.abspath(os.path.dirname(__file__))

# Input and output paths
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("LoanClassificationModels").getOrCreate()

# Step 2: Loading dataset
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Step 3: Dropping unnecessary columns
df = df.drop("Education", "Self_Employed", "Loan_ID")

# Step 4: Drop rows with null values
df = df.dropna()

# Step 5: Encoding categorical variables
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

# Step 7: Preprocessing steps
preprocessing_stages = [
    gender_indexer,
    married_indexer,
    property_indexer,
    label_indexer,
    assembler
]

# Step 8: Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 9: Classifiers Models
models = {
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100),
    "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="label"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)
}

# Step 10: Evaluation
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

# Step 12: Stop Spark session
spark.stop()

