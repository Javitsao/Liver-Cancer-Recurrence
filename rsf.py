import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

# Load your dataset from the CSV file
data = pd.read_csv("output1.csv")

# Define your features and target
X = data.drop(columns=['duration', 'recur.status_1', 'recur.status_0'])  # Features except 'duration' and 'recur.status_1'
y = Surv.from_dataframe('recur.status_1', 'duration', data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize and fit the Random Survival Forest model
rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
rsf.fit(X_train, y_train)

# Evaluate the model performance
c_index = rsf.score(X_test, y_test)

print("Concordance index on test set: {:.3f}".format(c_index))

# Predict the survival function on the test set
survival_functions = rsf.predict_survival_function(X_test)

# Extract the predicted recurrence status and duration
predicted_status = [fn.y[0] < 0.5 for fn in survival_functions]  # Recurrence if survival probability at time 0 is less than 0.5
predicted_duration = [fn.x[fn.y < 0.5][0] if len(fn.x[fn.y < 0.5]) > 0 else np.inf for fn in survival_functions]  # Time when survival probability drops below 0.5, or infinity if never

# Print some of the prediction results
print("Predicted Recurrence Status:")
print(predicted_status[:10])  # Print first 10 predictions
print("\nPredicted Duration:")
print(predicted_duration[:100])  # Print first 10 predictions