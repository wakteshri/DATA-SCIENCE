import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 1. Data Ingestion & Cleaning
df = pd.read_csv('Housing.csv')
df = df.dropna()

# 2. Categorical Encoding
# Convert binary text columns (yes/no) to 1/0
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
]
for col in binary_cols:
  df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode non-binary categorical variables ('furnishingstatus')
df = pd.get_dummies(
    df, columns=['furnishingstatus'], drop_first=True, dtype=int
)

# 3. Feature/Target Split
X = df.drop('price', axis=1)
y = df['price']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature & Target Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 6. Build ANN Regression Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1),  # Linear activation for continuous price output
])

# 7. Compile Model for Regression
model.compile(
    optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mae']
)

# 8. Train Model with Early Stopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
)

# 9. Model Evaluation (Rescaling back to actual house prices)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'\n--- Model Evaluation ---')
print(f'Mean Absolute Error (MAE) : ${mae:,.2f}')
print(f'Root Mean Squared Error (RMSE): ${rmse:,.2f}')
print(f'R² Score                  : {r2:.4f}')