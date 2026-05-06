import streamlit as st
from sklearn import svm
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import date, time # Import date and time for st.date_input, st.time_input

Note = '''
La información de las bolsas se consiguen de la siguiente manera:
1. Ingresa a Google Sheets
2. Crea una nueva hoja de cálculo
3. En la celda A1, escribe la siguiente fórmula para obtener datos históricos del Dow Jones:
   `=GOOGLEFINANCE("INDEXDJX:.DJI", "close", DATE(2020,1,1), DATE(2024,12,31), "DAILY")`
4. Presiona Enter y espera a que se carguen los datos.
5. Una vez que los datos estén cargados, selecciona las celdas con los datos (incluyendo los encabezados) y cópialos (Ctrl+C).
6. Abre un editor de texto (como Notepad) y pega los datos copiados (Ctrl+V).
7. Guarda el archivo con el nombre "DJ_data.csv" y asegúrate de seleccionar "All Files" en el tipo de archivo para que se guarde como CSV. Asegúrate de que el archivo se guarde con la extensión .csv y no como un archivo de texto
'''

st.set_page_config(layout="wide")
st.title("Análisis y Predicción de Precios de Acciones")

with st.expander("Instrucciones para obtener datos históricos"):
    st.info(Note)

# Cargar datos y cachearlos para evitar recargas en cada interacción
@st.cache_data
def load_data():
    data = pd.read_csv("DJ_data.csv", header = 0)
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S')
    return data

data = load_data()

# Convertir la columna 'Date' a datetime y luego a numérico (timestamp)
X = (data["Date"].astype('int64') // 10**9).values.reshape(-1, 1)  # Timestamp Unix en segundos (2D)
y = data["Close"].values.ravel() # .ravel() para evitar DataConversionWarning

# Initialize and train model with CatBoost
#model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=10, verbose=10)
#model.fit(X, y)

# Initialize and train model with sklearn
clf = svm.SVR() # Usar SVR para regresión (precios), no SVC (clasificación)

# Cachear el entrenamiento del modelo SVR
@st.cache_resource
def train_svr_model(X_data, y_data):
    clf = svm.SVR()
    clf.fit(X_data, y_data)
    return clf

clf = train_svr_model(X, y)

# --- Entrada para la Predicción ---
st.subheader("Selecciona la fecha para la predicción")
selected_date = st.date_input("Fecha", datetime.now().date())
selected_time = st.time_input("Hora", time(16, 0, 0))

#-----------------------------------------------------------------------------------------------------------------------
# DecisionTree Regression
# 2. Split data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify categorical features (CatBoost handles them directly if specified)
# cat_features = ['your_categorical_column_name']

# Initialize the CatBoost Regressor
# Common loss functions for regression are 'RMSE' (default) or 'MAE'
# Cachear el entrenamiento del modelo CatBoost
@st.cache_resource
def train_catboost_model(X_train_data, y_train_data):
    model = CatBoostRegressor(loss_function='RMSE', iterations=500, learning_rate=0.1, depth=6, verbose=0) # verbose=0 mutes training output
    model.fit(X_train_data, y_train_data)
    return model

# Train the model
catboost_model = train_catboost_model(X1_train, y1_train)

# Make predictions
y1_pred = catboost_model.predict(X1_test)

# Evaluate the model
#print("Tree Regression RMSE:", y1_pred)
#-----------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# Predict with DecisionTreeClassifier

# Combinar fecha y hora seleccionadas por el usuario
prediction_datetime_str = f"{selected_date.strftime('%d/%m/%Y')} {selected_time.strftime('%H:%M:%S')}"
fecha_obj_pred = datetime.strptime(prediction_datetime_str, '%d/%m/%Y %H:%M:%S')
X_pred_input = np.array([[fecha_obj_pred.timestamp()]])

prediccion_svr = clf.predict(X_pred_input)
#-----------------------------------------------------------------------------------------------------------------------
# Predict with KNN
# Cachear el entrenamiento del modelo KNN
@st.cache_resource
def train_knn_model(X_train_data, y_train_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_data)
    k_value = 5
    knn = KNeighborsRegressor(n_neighbors=k_value)
    knn.fit(X_train_scaled, y_train_data)
    return knn, scaler

knn_model, knn_scaler = train_knn_model(X_train, y_train)
prediction_knn = knn_model.predict(knn_scaler.transform(X_pred_input))
#-----------------------------------------------------------------------------------------------------------------------

# Predictions: 5 values
# Predecir con CatBoost para la fecha seleccionada
prediccion_catboost = catboost_model.predict(X_pred_input)

st.subheader("Resultados de la Predicción")
st.write(f"Valor predicho con SVR para {prediction_datetime_str}: **{prediccion_svr[0]:.2f}**")
st.write(f"Valor predicho con CatBoost para {prediction_datetime_str}: **{prediccion_catboost[0]:.2f}**")
st.write(f"Valor predicho con KNN para {prediction_datetime_str}: **{prediction_knn[0]:.2f}**")

# Calculando la media normalizada entre prediccion y prediction
media_normalizada = (prediccion_svr[0] + prediccion_catboost[0] + prediction_knn[0]) / 3
st.write(f"Valor predicho con media normalizada para {prediction_datetime_str}: **{media_normalizada:.2f}**")

st.subheader("Visualización de Datos Históricos y Predicción")
plot_data = data[['Date', 'Close']].copy()
plot_data['Date'] = pd.to_datetime(plot_data['Date'])
plot_data = plot_data.set_index('Date')

combined_plot_data = plot_data.copy()
combined_plot_data.loc[fecha_obj_pred, 'Predicted Close'] = media_normalizada
st.line_chart(combined_plot_data[['Close', 'Predicted Close']])
