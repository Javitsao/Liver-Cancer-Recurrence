import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
import numpy as np
import matplotlib.pyplot as plt

# Assuming data is already loaded and prepared as X, y
data = pd.read_csv('output1.csv')
X = data.drop(columns=['duration', 'recur.status_0', 'recur.status_1'])
y = np.array([(bool(row['recur.status_1']), row['duration']) for index, row in data.iterrows()],
             dtype=[('Status', '?'), ('duration', '<f8')])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Train the Random Survival Forest model
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train, y_train)

# Predict the risk scores for the test set
risk_scores = rsf.predict(X_test)

# Calculate the C-index
event_indicator = y_test['Status']
event_time = y_test['duration']
c_index = concordance_index_censored(event_indicator, event_time, risk_scores)
print(f"C-index: {c_index[0]}")

# # Predicting survival functions for the test set
# survival_functions = rsf.predict_survival_function(X_test, return_array=False)

# # Extracting time points and survival probabilities for the first individual
# # Each survival function object in the list has 'x' for times and 'y' for survival probabilities
# time_points = survival_functions[0].x
# survival_probabilities = survival_functions[0].y

# # Plotting the survival function for the first individual
# plt.step(time_points, survival_probabilities, where="post", label="Survival function")
# plt.ylabel("Survival probability")
# plt.xlabel("Time in days")
# plt.title("Predicted Survival Function for the First Test Sample")
# plt.legend()
# plt.show()

# Predicting survival functions for the test set
survival_functions = rsf.predict_survival_function(X_test, return_array=False)

# Plotting the survival functions for all test samples
for sf in survival_functions:
    plt.step(sf.x, sf.y, where="post", alpha=0.5)  # Set alpha for better visibility when lines overlap

plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.title("Predicted Survival Functions for Test Samples")
plt.show()