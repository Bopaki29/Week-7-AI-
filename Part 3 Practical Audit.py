
# Import necessary libraries
import pandas as pd
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the COMPAS dataset
compas_data = CompasDataset()

# Define privileged and unprivileged groups for bias analysis
privileged_groups = [{'race': 1}]   # Typically Caucasian
unprivileged_groups = [{'race': 0}] # Typically African-American

# Split dataset into training and testing sets
train, test = compas_data.split([0.7], shuffle=True)

# Check initial bias metric - Disparate Impact on training data
metric_train = BinaryLabelDatasetMetric(train,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)
print("Initial Disparate Impact (train):", metric_train.disparate_impact())

# Apply Reweighing algorithm to mitigate bias in training data
rw = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
rw.fit(train)
train_transformed = rw.transform(train)

# Prepare data for model training
X_train = train_transformed.features
y_train = train_transformed.labels.ravel()
X_test = test.features
y_test = test.labels.ravel()

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Evaluate fairness metrics on test data predictions
predicted_dataset = test.copy()
predicted_dataset.labels = y_pred

metric_test = ClassificationMetric(test, predicted_dataset,
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups)

print("Disparate Impact after mitigation:", metric_test.disparate_impact())
print("Equal Opportunity Difference:", metric_test.equal_opportunity_difference())
