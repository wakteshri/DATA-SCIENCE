    # ==========================================================
# Binary Classification using ANN (Titanic Dataset)
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical columns
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print("\nProcessed Dataset")
print(df.head())

# ----------------------------------------------------------
# Feature and Target
# ----------------------------------------------------------

X = df.drop('Survived', axis=1)
y = df['Survived']

# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# Build ANN Model
# ----------------------------------------------------------

model = Sequential()

model.add(Dense(32,
                activation='relu',
                input_shape=(X_train.shape[1],)))

model.add(Dense(16,
                activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(8,
                activation='relu'))

# Output Layer
model.add(Dense(1,
                activation='sigmoid'))

# ----------------------------------------------------------
# Compile Model
# ----------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary")
model.summary()

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = model.predict(X_test)

# Convert probability to class
y_pred = (y_pred > 0.5).astype(int)

# ----------------------------------------------------------
# Performance Metrics
# ----------------------------------------------------------

print("\nAccuracy Score")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# Predict First Passenger
# ----------------------------------------------------------

sample = X.iloc[0:1]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPrediction Probability :", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction : Survived")
else:
    print("Prediction : Did Not Survive")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

model.save("titanic_ann_model.h5")

print("\nModel Saved Successfully")