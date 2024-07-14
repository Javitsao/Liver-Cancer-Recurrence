import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

# Load the dataset
data = pd.read_csv("output.csv")

# Select features - excluding duration, survival.month, recur.status_*, and death.status_*
X = data.drop(columns=['duration', 'survival.month', 'recur.status_0', 'recur.status_1', 'death.status_0', 'death.status_1'])

# Prepare the target variable, using 'duration' as time and 'recur.status_1' as the event
y = Surv.from_dataframe(event='recur.status_1', time='duration', data=data)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the RSF model
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42)

# Fit the model on the training data
rsf.fit(X_train, y_train)

# Calculate the c-index for the training data
c_index_train = rsf.score(X_train, y_train)
print(f"Training C-index: {c_index_train}")

# Calculate the c-index for the test data
c_index_test = rsf.score(X_test, y_test)
print(f"Test C-index: {c_index_test}")

# Predicting the survival function for the first 5 patients
survival_functions = rsf.predict_survival_function(X.iloc[:10])
#survival_functions = rsf.predict_survival_function(X_test)

# Visualize the survival functions
plt.figure(figsize=(10, 6))
for i, surv_func in enumerate(survival_functions):
    plt.step(surv_func.x, 1-surv_func.y, where="post", label=f"Patient {i+1}")

plt.ylabel("Recurrence probability")
plt.xlabel("Time in months")
plt.legend()
plt.title("Predicted Survival Functions")
plt.show()