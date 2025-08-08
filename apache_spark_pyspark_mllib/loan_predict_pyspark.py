import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
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

# Step 3: Drop unnecessary columns
df = df.drop("Education", "Self_Employed")

# Step 4: Drop rows with null values
df = df.dropna()

# Step 5: Encode categorical variables
gender_indexer = StringIndexer(inputCol="Gender", outputCol="gender_Indexed")
mariage_indexer = StringIndexer(inputCol="Married", outputCol="Married_Indexed")
Property_Area_indexer = StringIndexer(inputCol="Property_Area_indexed", outputCol="Property_Area_Indexed")
Loan_Status_indexer = StringIndexer(inputCol="Loan_Status", outputCol="Loan_Status_Indexed")

# Step 6: Assemble features
assembler = VectorAssembler(
    inputCols=["Loan_ID", "gender_Indexed", "Married_Indexed", "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area_indexed"],
    outputCol="features"
)

# Step 7: Create Random Forest model
rf = RandomForestClassifier(labelCol="Loan_Status_Indexed", featuresCol="features", numTrees=100)

# Step 8: Build pipeline
pipeline = Pipeline(stages=[gender_indexer, mariage_indexer, Property_Area_indexer, assembler, rf])

# Step 9: Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 10: Train model
model = pipeline.fit(train_data)

# Step 11: Predict
predictions = model.transform(test_data)

# Step 12: Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="Loan_Status_Indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Write accuracy to output file
with open(output_path, "w") as f:
    f.write(f"Random Forest Classification Accuracy: {accuracy:.4f}\n")

# Stop Spark
spark.stop()

