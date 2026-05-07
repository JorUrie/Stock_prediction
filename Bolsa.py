import os
import streamlit as st
from sklearn import svm
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from datetime import date, time # Import date and time for st.date_input, st.time_input
import matplotlib.pyplot as plt

Note = '''
La información de las bolsas se consiguen de la siguiente manera:
1. Ingresa a Google Sheets
2. Crea una nueva hoja de cálculo
3. En la celda A1, escribe la siguiente fórmula para obtener datos históricos del Dow Jones:
   `=GOOGLEFINANCE("INDEXDJX:.DJI", "close", DATE(2020,1,1), DATE(2024,12,31), "DAILY")`
4. Presiona Enter y espera a que se carguen los datos.
5. Una vez que los datos estén cargados, selecciona las celdas con los datos (incluyendo los encabezados) y cópialos (Ctrl+C) o descárgalos.
6. Abre un editor de texto (como Notepad) y pega los datos copiados (Ctrl+V).
7. Guarda el archivo con el nombre "DJ_data.csv" y asegúrate de seleccionar "All Files" en el tipo de archivo para que se guarde como CSV. Asegúrate de que el archivo se guarde con la extensión .csv y no como un archivo de texto

Link de descargas: https://www.google.com/finance/
'''

st.set_page_config(layout="wide")
st.title("Análisis y Predicción de Precios de Acciones")

with st.expander("Instrucciones para obtener datos históricos"):
    st.info(Note)

# --- Interfaz para subir el archivo ---
st.sidebar.header("Carga de Datos")
uploaded_files = st.sidebar.file_uploader("Sube tus archivos CSV", type=["csv"], accept_multiple_files=True)

# Cargar datos y cachearlos para evitar recargas en cada interacción
@st.cache_data
def process_file_data(file_input):
    try:
        data = pd.read_csv(file_input, header=0)
        # Limpieza de columnas sin nombre (Unnamed)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        # Asegurar formato de fecha
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S')
        return data
    except Exception as e:
        st.error(f"Error al leer el archivo {getattr(file_input, 'name', 'local')}: {e}")
        return None

# Initialize and train model with CatBoost
#model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=10, verbose=10)
#model.fit(X, y)

# Initialize and train model with sklearn
clf = svm.SVR() # Usar SVR para regresión (precios), no SVC (clasificación)

# Cachear el entrenamiento del modelo SVR
@st.cache_resource
def train_svr_model(X_data, y_data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    # C=1000 y epsilon=0.1 son valores robustos para datos financieros escalados
    clf = svm.SVR(kernel='rbf', C=1000, epsilon=0.05, gamma='scale')
    clf.fit(X_scaled, y_data)
    return clf, scaler

# --- Entrada para la Predicción ---
st.subheader("Configuración de la Proyección")
selected_date = st.date_input("Fecha", datetime.now().date())
selected_time = st.time_input("Hora", time(16, 0, 0))

@st.cache_resource
def train_catboost_model(X_train_data, y_train_data):
    model = CatBoostRegressor(loss_function='RMSE', iterations=500, learning_rate=0.1, depth=6, verbose=0) # verbose=0 mutes training output
    model.fit(X_train_data, y_train_data)
    return model

@st.cache_resource
def train_knn_model(X_train_data, y_train_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_data)
    # weights='distance' da más importancia a los puntos más parecidos/cercanos
    knn = KNeighborsRegressor(n_neighbors=7, weights='distance', metric='euclidean')
    knn.fit(X_train_scaled, y_train_data)
    return knn, scaler

@st.cache_resource
def train_arima_model(y_data):
    try:
        # Order (5,1,0) es un punto de partida común para datos financieros diarios
        model = ARIMA(y_data, order=(5, 1, 0))
        return model.fit()
    except Exception:
        return None

# Determinar qué archivos procesar
files_to_process = []
if uploaded_files:
    files_to_process = uploaded_files
else:
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "DJ_data.csv")
    if os.path.exists(file_path):
        files_to_process = [file_path]

if not files_to_process:
    st.warning("⚠️ No hay datos disponibles. Por favor, sube archivos CSV en la barra lateral.")
    st.stop()

for file_input in files_to_process:
    file_name = getattr(file_input, 'name', 'DJ_data.csv')
    with st.expander(f"Análisis para: {file_name}", expanded=True):
        data = process_file_data(file_input)
        if data is None: continue

        X = (data["Date"].astype('int64') // 10**9).values.reshape(-1, 1)
        y = data["Close"].values.ravel()

        # Entrenar modelos
        clf, svr_scaler = train_svr_model(X, y)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.3, random_state=42)
        catboost_model = train_catboost_model(X1_train, y1_train)
        knn_model, knn_scaler = train_knn_model(X1_train, y1_train)
        arima_fit = train_arima_model(y)

        # --- Proyección Histórica (Walk-forward) ---
        if st.checkbox("Calcular Proyección Histórica (Paso a paso)", value=False, key=f"hist_corr_{file_name}"):
            st.info("Calculando proyecciones históricas... Esto puede tardar dependiendo del tamaño del archivo.")
            hist_svr, hist_cat, hist_knn, hist_arima = [], [], [], []
            hist_arima_lower, hist_arima_upper = [], []
            start_idx = 10  # Comenzar después de 10 elementos
            
            progress_bar = st.progress(0)
            for i in range(start_idx, len(data)):
                # Datos hasta el momento i
                curr_X, curr_y = X[:i], y[:i]
                next_X = X[i].reshape(1, -1)
                
                # SVR
                s_m, s_sc = train_svr_model(curr_X, curr_y)
                hist_svr.append(s_m.predict(s_sc.transform(next_X))[0])
                
                # CatBoost (simplificado para velocidad en el bucle)
                c_m = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=4, verbose=0).fit(curr_X, curr_y)
                hist_cat.append(c_m.predict(next_X)[0])
                
                # KNN
                k_m, k_sc = train_knn_model(curr_X, curr_y)
                hist_knn.append(k_m.predict(k_sc.transform(next_X))[0])
                
                # ARIMA
                forecast = arima_fit.get_forecast(steps=1)
                hist_arima.append(forecast.predicted_mean[0])
                ci = forecast.conf_int()
                hist_arima_lower.append(ci.iloc[0,0])
                hist_arima_upper.append(ci.iloc[0,1])
                
                progress_bar.progress((i - start_idx) / (len(data) - start_idx))
            
            # Guardar resultados históricos en el dataframe para graficar
            data.loc[data.index[start_idx:], 'SVR_Hist'] = hist_svr
            data.loc[data.index[start_idx:], 'Cat_Hist'] = hist_cat
            data.loc[data.index[start_idx:], 'KNN_Hist'] = hist_knn
            data.loc[data.index[start_idx:], 'ARIMA_Hist'] = hist_arima
            data.loc[data.index[start_idx:], 'ARIMA_Lower'] = hist_arima_lower
            data.loc[data.index[start_idx:], 'ARIMA_Upper'] = hist_arima_upper

        # Calcular proyección para un punto
        target_datetime = datetime.combine(selected_date, selected_time)
        ts = np.array([[target_datetime.timestamp()]])
        
        # Proyecciones de regresores
        p_svr = clf.predict(svr_scaler.transform(ts))[0]
        p_cat = catboost_model.predict(ts)[0]
        p_knn = knn_model.predict(knn_scaler.transform(ts))[0]

        # Proyección ARIMA (basada en pasos desde el último dato)
        last_date = data["Date"].max()
        future_days = (target_datetime - last_date).days
        
        if arima_fit and future_days > 0:
            forecast = arima_fit.get_forecast(steps=future_days)
            p_arima = forecast.predicted_mean[-1]
            ci = forecast.conf_int()
            arima_lower = ci[-1,0]
            arima_upper = ci[-1,1]
        else:
            # Si la fecha es pasada o el modelo falla, usamos el último valor conocido
            p_arima = y[-1]
            arima_lower = np.nan
            arima_upper = np.nan

        p_geo = (p_svr * p_cat * p_knn * p_arima) ** (1/4)
        
        res_data = {
            "Fecha": [target_datetime],
            "SVR": [p_svr],
            "CatBoost": [p_cat],
            "KNN": [p_knn],
            "ARIMA": [p_arima],
            "ARIMA Lower": [arima_lower],
            "ARIMA Upper": [arima_upper],
            "Media Geom": [p_geo]
        }

        df_res = pd.DataFrame(res_data)
        st.write(f"Resultado de la Proyección ({target_datetime}):")
        st.dataframe(df_res.style.format({
            "SVR": "{:.2f}", "CatBoost": "{:.2f}", "KNN": "{:.2f}", "ARIMA": "{:.2f}", "ARIMA Lower": "{:.2f}", "ARIMA Upper": "{:.2f}", "Media Geom": "{:.2f}"
        }))

        # Gráfica
        plot_data = data[['Date', 'Close']].copy().set_index('Date')
        combined_plot_data = plot_data.copy()

        # Añadir las series históricas si existen
        if 'SVR_Hist' in data.columns:
            combined_plot_data['SVR_H'] = data.set_index('Date')['SVR_Hist']
            combined_plot_data['Cat_H'] = data.set_index('Date')['Cat_Hist']
            combined_plot_data['KNN_H'] = data.set_index('Date')['KNN_Hist']
            combined_plot_data['ARIMA_H'] = data.set_index('Date')['ARIMA_Hist']
            combined_plot_data['ARIMA_L'] = data.set_index('Date')['ARIMA_Lower']
            combined_plot_data['ARIMA_U'] = data.set_index('Date')['ARIMA_Upper']
        
        combined_plot_data.loc[target_datetime, 'SVR'] = p_svr
        combined_plot_data.loc[target_datetime, 'CatBoost'] = p_cat
        combined_plot_data.loc[target_datetime, 'KNN'] = p_knn
        combined_plot_data.loc[target_datetime, 'ARIMA'] = p_arima
        combined_plot_data.loc[target_datetime, 'Media'] = p_geo

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(combined_plot_data.index, combined_plot_data['Close'], label='Close', linewidth=2)
        
        if 'SVR_H' in combined_plot_data.columns:
            ax.plot(combined_plot_data.index, combined_plot_data['SVR_H'], label='SVR Hist', linestyle='--')
            ax.plot(combined_plot_data.index, combined_plot_data['Cat_H'], label='CatBoost Hist', linestyle='--')
            ax.plot(combined_plot_data.index, combined_plot_data['KNN_H'], label='KNN Hist', linestyle='--')
            ax.plot(combined_plot_data.index, combined_plot_data['ARIMA_H'], label='ARIMA Hist', linestyle='--')
            ax.fill_between(combined_plot_data.index, combined_plot_data['ARIMA_L'], combined_plot_data['ARIMA_U'], alpha=0.3, label='ARIMA CI', color='blue')
        
        # Plot future points
        future_mask = combined_plot_data.index == target_datetime
        ax.scatter(combined_plot_data.index[future_mask], combined_plot_data.loc[future_mask, 'SVR'], label='SVR Future', marker='o', s=50)
        ax.scatter(combined_plot_data.index[future_mask], combined_plot_data.loc[future_mask, 'CatBoost'], label='CatBoost Future', marker='o', s=50)
        ax.scatter(combined_plot_data.index[future_mask], combined_plot_data.loc[future_mask, 'KNN'], label='KNN Future', marker='o', s=50)
        ax.scatter(combined_plot_data.index[future_mask], combined_plot_data.loc[future_mask, 'ARIMA'], label='ARIMA Future', marker='o', s=50)
        ax.scatter(combined_plot_data.index[future_mask], combined_plot_data.loc[future_mask, 'Media'], label='Media Future', marker='o', s=50)
        
        if not np.isnan(arima_lower):
            ax.errorbar(target_datetime, p_arima, yerr=[[p_arima - arima_lower], [arima_upper - p_arima]], fmt='o', color='red', label='ARIMA CI Future', capsize=5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
