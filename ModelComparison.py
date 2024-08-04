import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Define all models
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Naive Bayes', GaussianNB()),
    ('K-nearest neighbour', KNeighborsClassifier(n_neighbors=5))
]

# Load data
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Define the number of folds for cross-validation
num_folds = 5

# Split the data into training and test sets
def train_test_split(X, y, test_size=0.2):
    # Randomize index to split into training and test sets
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]

    # Split data in training and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Implement K-fold cross-validation
def k_fold_cross_validation(X, y, num_folds):
    fold_size = len(X) // num_folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for fold in range(num_folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train_fold, X_val_fold = X[train_indices], X[val_indices]
        y_train_fold, y_val_fold = y[train_indices], y[val_indices]

        yield X_train_fold, y_train_fold, X_val_fold, y_val_fold

# Initialize variables to keep track of the best model and its accuracy
best_model = None
best_accuracy = 0

# Cross-validation
for model_name, model in models:
    total_accuracy = 0

    for X_train_fold, y_train_fold, X_val_fold, y_val_fold in k_fold_cross_validation(X_train, y_train, num_folds):
        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the validation fold
        fold_accuracy = np.mean(model.predict(X_val_fold) == y_val_fold)
        total_accuracy += fold_accuracy

    # Calculate the average accuracy across folds
    average_accuracy = total_accuracy / num_folds

    print(f'{model_name}: Average Accuracy = {average_accuracy:.4f}')

    # Update the best model if the current model is better
    if average_accuracy > best_accuracy:
        best_model = model_name
        best_accuracy = average_accuracy

# Train the best model on the full training set
best_model = [model for model_name, model in models if model_name == best_model][0]
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
test_accuracy = np.mean(best_model.predict(X_test) == y_test)
print(f'Best Model: {best_model}\nTest Accuracy = {test_accuracy:.4f}')
