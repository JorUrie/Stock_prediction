from sklearn import svm
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf

# Descargar datos directamente de Yahoo Finance (ejemplo con 1 año de historial para AAPL)
data = yf.download('AAPL', period='1y')

if data.empty:
    print("Error: No se pudieron descargar datos de Yahoo Finance.")
    exit()

# En yfinance, la fecha es el índice (index). La convertimos a timestamp numérico.
X_raw = (data.index.astype('int64') // 10**9).values
X = X_raw.reshape(-1, 1)
y = data["Close"].values.ravel() # .ravel() evita la advertencia de DataConversion

# Initialize and train model with CatBoost
#model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=10, verbose=10)
#model.fit(X, y)

# Initialize and train model with sklearn
clf = svm.SVR() # Usar SVR para regresión (precios), no SVC (clasificación)
clf = clf.fit(X, y)

fecha_str = "19/03/2026 16:00:00"
fecha_obj = datetime.strptime(fecha_str, '%d/%m/%Y %H:%M:%S')
X_pred = np.array([[fecha_obj.timestamp()]]) # Convertir a timestamp y matriz 2D

#-----------------------------------------------------------------------------------------------------------------------
# DecisionTree Regression
# 2. Split data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify categorical features (CatBoost handles them directly if specified)
# cat_features = ['your_categorical_column_name']

# Initialize the CatBoost Regressor
# Common loss functions for regression are 'RMSE' (default) or 'MAE'
model = CatBoostRegressor(loss_function='RMSE', iterations=500, learning_rate=0.1, depth=6, verbose=0) # verbose=0 mutes training output

# Train the model
model.fit(X1_train, y1_train)

# Make predictions
y1_pred = model.predict(X1_test)

# Evaluate the model
print("CatBoost Prediction on test set:", y1_pred[:5])
#-----------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# Predict with DecisionTreeClassifier
prediccion = clf.predict(X_pred)
#-----------------------------------------------------------------------------------------------------------------------
# Predict with KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k_value = 5
knn = KNeighborsRegressor(n_neighbors=k_value)
knn.fit(X_train, y_train)
prediction = knn.predict(scaler.transform(X_pred))
#-----------------------------------------------------------------------------------------------------------------------

# Predictions: 5 values
#print("Valor predicho con CatBoost:", model.predict(X))
print("Valor predicho con SVR:", prediccion)
print("Valor predicho con KNN:", prediction)

# Calculando la media normalizada entre prediccion y prediction


# Calculando porcentaje de error con KNN cuando el valor real es: 45,577.47
error = (abs((prediction - 45577.47)/45577.47))*100
print("El error es: ", error)
