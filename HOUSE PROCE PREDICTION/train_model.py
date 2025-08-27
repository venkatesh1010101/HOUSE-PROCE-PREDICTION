import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('Housing.csv')  # Make sure Housing.csv is in the same folder

# Preprocess categorical features (same as before)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Scaling numerical features
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
if 'furnishingstatus_semi-furnished' in df.columns:
    numerical_cols.append('furnishingstatus_semi-furnished')
if 'furnishingstatus_unfurnished' in df.columns:
    numerical_cols.append('furnishingstatus_unfurnished')

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Features & target
X = df.drop('price', axis=1)
y = df['price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler together as dictionary
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'columns': X.columns.tolist()}, f)

print("Model and scaler saved to 'model.pkl'")
