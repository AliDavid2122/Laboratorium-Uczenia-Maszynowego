# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który implementuje algorytm sieci neuronowej RNN do analizy 
# sekwencji danych czasowych.
# Opis działań do realizacji:
# • Wczytaj dane czasowe, np. dane dotyczące temperatury, cen akcji itp.
# • Zbuduj model sieci neuronowej RNN, uwzględniając odpowiednią architekturę
#   i funkcję aktywacji.
# • Przekształć dane w formę odpowiednią do modelu RNN
# • Dokonaj treningu modelu na danych czasowych.
# • Dokonaj predykcji na nowych danych czasowych i ocen jakość predykcji, 
#   np. za pomocą błędu średniokwadratowego (RMSE).
# • Przedstaw wyniki wizualizacyjnie, np. poprzez wykresy porównujące
#   rzeczywiste dane z przewidywanymi.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RNN
from keras.layers import SimpleRNN
from keras.layers import GRU
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import cv2
from tqdm import tqdm
from datetime import datetime
from os import makedirs

# Definicja liczby kroków czasowych i cech
n_steps = 5
n_pred = 2
n_epochs = 20
batch_size = 256

# Właściwości wideo
n_frames = 300
frame_width = 1920
frame_height = 1080
fps = 10

# Nazwy plików wyjsciowych
fname_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
makedirs(fname_prefix)
fname_prefix += "/";
fname_prefix += str(n_steps) + "steps_";
fname_prefix += str(n_pred) + "pred_";
fname_prefix += str(n_epochs) + "epochs_";
fname_prefix += str(batch_size) + "batchSize_";

# Wczytaj dane z pliku CSV
df = pd.read_csv('EURUSD_M1.csv', sep=',', index_col=0, header=None, names=["Time","Open","High","Low","Close","Volume"])
df.index = mdates.date2num(pd.to_datetime(df.index))

# Przekształć dane do formy odpowiedniej dla modelu RNN
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def prepare_data(data, n_steps, n_pred):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix+n_pred > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:end_ix+n_pred]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = prepare_data(scaled_data, n_steps, n_pred)

# Podział danych na zbiór treningowy i testowy
split = int(0.75 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
timestamps_train = df.index[:split].values
timestamps_test = df.index[split:-n_steps-n_pred].values

model = Sequential()
model.add(LSTM(units=n_steps, return_sequences=True, input_shape=(n_steps, 1)))
model.add(GRU(units=n_steps, return_sequences=False))
model.add(Dense(units=n_pred))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
plot_model(model, to_file=fname_prefix+'model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Ocena modelu na danych testowych
mse = model.evaluate(X_test, y_test, verbose=0)
print("MSE:", mse)

model.save(fname_prefix + "model.h5")

# Ocena modelu na danych testowych
predictions = model.predict(X_test)
predicted_data = scaler.inverse_transform(predictions)
X_test_scaled_back = scaler.inverse_transform(X_test.reshape(-1, n_steps))
y_test_scaled_back = scaler.inverse_transform(y_test.reshape(-1, n_pred))

def visualize_predictions(timestamps, y_test, predicted_data, title, fname='fig.png'):
    difference = predicted_data - y_test
    plt.figure(figsize=(12, 10))
    spec = plt.GridSpec(2, 1, height_ratios=[7, 3])
    tick_positions = np.linspace(timestamps[0], timestamps[-1], 12)
    
    # Wykres 1: Przewidziane i rzeczywiste ceny zamknięcia
    plt.subplot(spec[0])
    plt.plot(timestamps, y_test, label='Faktyczna', color='violet')
    plt.plot(timestamps, predicted_data, label='Predykcja', color='gold')
    plt.fill_between(timestamps, y_test.ravel(), predicted_data.ravel(), where=(difference.ravel() >= 0), interpolate=True, color='gold', alpha=0.3)
    plt.fill_between(timestamps, y_test.ravel(), predicted_data.ravel(), where=(difference.ravel() < 0), interpolate=True, color='violet', alpha=0.3)
    plt.title(title)
    plt.xlabel('Czas')
    plt.ylabel('Cena')
    plt.legend()
    plt.gca().set_xlim(left=min(timestamps))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    plt.xticks(tick_positions)
    plt.gca().tick_params(axis='x', rotation=90)
    plt.grid(True)
    
    # Wykres 2: Różnica między przewidywanymi a rzeczywistymi cenami zamknięcia
    plt.subplot(spec[1])
    plt.plot(timestamps, difference, label='Róźnica', color='black')
    plt.fill_between(timestamps, difference.ravel(), 0, where=(difference.ravel() >= 0), interpolate=True, color='gold', alpha=0.3)
    plt.fill_between(timestamps, difference.ravel(), 0, where=(difference.ravel() < 0), interpolate=True, color='violet', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title('Różnica między predykcją a faktyczną ceną')
    plt.xlabel('Czas')
    plt.ylabel('Róźnica cen')
    plt.legend()
    plt.gca().set_xlim(left=min(timestamps))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f')) 
    plt.xticks(tick_positions)
    plt.gca().tick_params(axis='x', rotation=90)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

visualize_predictions(timestamps_test, y_test_scaled_back[:,0], predicted_data[:,0], title="Predykcja i faktyczne ceny EUR-USD", fname=fname_prefix+'output_plot_all.png')
visualize_predictions(timestamps_test[-1440:], y_test_scaled_back[-1440:,0], predicted_data[-1440:,0], title="Predykcja i faktyczne ceny EUR-USD (ostatni dzień)", fname=fname_prefix+'output_plot_lastday.png')
visualize_predictions(timestamps_test[-60:], y_test_scaled_back[-60:,0], predicted_data[-60:,0], title="Predykcja i faktyczne ceny EUR-USD (ostatnia godzina)", fname=fname_prefix+'output_plot_lasthour.png')

# Initialize video writer
min_y = min(X_test_scaled_back[-n_frames:].min(), y_test_scaled_back[-n_frames:].min(), predicted_data[-n_frames:].min())
max_y = max(X_test_scaled_back[-n_frames:].max(), y_test_scaled_back[-n_frames:].max(), predicted_data[-n_frames:].max())
out = cv2.VideoWriter(fname_prefix+'output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

fig, ax = plt.subplots(figsize=(19.2, 10.8))

for k in tqdm(range(n_frames,0 , -1), desc="Processing frames", unit="frame"):
    ax.clear()
    y_source = X_test_scaled_back[-k].ravel()
    source_timestamps = timestamps_test[-n_steps-n_pred-k:-n_pred-k]
    y_actual = y_test_scaled_back[-k].ravel()
    y_prediction = predicted_data[-k].ravel()
    prediction_timestamps = timestamps_test[-n_pred-k:-k]
    y_source = np.concatenate((y_source, y_actual[0]), None)
    source_timestamps =  np.concatenate((source_timestamps, prediction_timestamps[0]), None)

    difference = y_prediction - y_actual
    ax.plot(source_timestamps, y_source, label='Źródło', color='navy')
    ax.plot(prediction_timestamps, y_actual, label='Faktyczna', color='violet')
    ax.plot(prediction_timestamps, y_prediction, label='Predykcja', color='gold')
    ax.fill_between(prediction_timestamps, y_actual, y_prediction, where=(difference.ravel() >= 0), interpolate=True, color='gold', alpha=0.3)
    ax.fill_between(prediction_timestamps, y_actual, y_prediction, where=(difference.ravel() < 0), interpolate=True, color='violet', alpha=0.3)
    ax.axvline(x=timestamps_test[-n_pred-k], color='black', linestyle='--', linewidth=2)

    ax.legend(loc='lower left')
    ax.set_xlim(left=source_timestamps[0], right=prediction_timestamps[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)
    ax.set_ylim(min_y, max_y)
    ax.set_title("Przewidiwanie kolejnych " + str(n_pred) + " minut zmiany ceny EUR-USD na podstawie ostatnich " + str(n_steps) + " minut")
    ax.set_xlabel('Czas')
    ax.set_ylabel('Cena')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))  # Format y-axis ticks to display in desired format
 
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)

plt.close(fig)
out.release()
