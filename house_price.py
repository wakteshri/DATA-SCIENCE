# ==========================================================
# House Price Prediction using ANN (TensorFlow/Keras)
# Dataset: Housing.csv
# ==========================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("Housing.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 3. Data Cleaning
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# 4. Encode Categorical Columns
# -----------------------------
label = LabelEncoder()

binary_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_columns:
    df[col] = label.fit_transform(df[col])

# One-Hot Encoding
df = pd.get_dummies(df,
                    columns=['furnishingstatus'],
                    drop_first=True)

print("\nEncoded Dataset:")
print(df.head())

# -----------------------------
# 5. Feature and Target
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 7. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 8. Build ANN Model
# -----------------------------
model = Sequential()

# Input Layer
model.add(Dense(64,
                activation='relu',
                input_shape=(X_train.shape[1],)))

# Hidden Layer
model.add(Dense(32,
                activation='relu'))

# Dropout Layer
model.add(Dropout(0.2))

# Hidden Layer
model.add(Dense(16,
                activation='relu'))

# Output Layer
model.add(Dense(1,
                activation='linear'))

# -----------------------------
# 9. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# -----------------------------
# 10. Model Summary
# -----------------------------
print("\nModel Summary:")
model.summary()

# -----------------------------
# 11. Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 12. Evaluate Model
# -----------------------------
loss, mae = model.evaluate(X_test, y_test)

print("\n==============================")
print("Model Evaluation")
print("==============================")
print("Test Loss (MSE):", loss)
print("Test MAE:", mae)

# -----------------------------
# 13. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 14. Performance Metrics
# -----------------------------
mae_score = mean_absolute_error(y_test, y_pred)
rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nPerformance Metrics")
print("--------------------------")
print("Mean Absolute Error :", mae_score)
print("Root Mean Squared Error :", rmse_score)

# -----------------------------
# 15. Actual vs Predicted
# -----------------------------
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.flatten()
})

print("\nActual vs Predicted Prices:")
print(result.head(10))

# -----------------------------
# 16. Predict New House Price
# -----------------------------
sample = X.iloc[0:1]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPredicted Price for First House:")
print(prediction[0][0])

# -----------------------------
# 17. Save Model (Optional)
# -----------------------------
model.save("house_price_ann_model.h5")

print("\nModel saved successfully as house_price_ann_model.h5")