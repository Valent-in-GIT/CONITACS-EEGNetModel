# Importar las librerías necesarias
import mne
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D, SpatialDropout2D, AveragePooling2D
from keras.constraints import max_norm

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import numpy
import os

# Función para leer y procesar los datos del archivo GDF
def read_data(file_path):
    # Leer los datos brutos desde el archivo GDF
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    # Filtrar las frecuencias entre 8 y 30 Hz
    raw.filter(8., 30., fir_design='firwin')
    
    # Extraer los eventos y anotaciones
    events, a = mne.events_from_annotations(raw)
    raw.load_data()
    
    # Marcar los canales EOG como malos
    raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    # Seleccionar los canales EEG relevantes
    raw.pick(["EEG-0","EEG-2","EEG-3","EEG-C3","EEG-6","EEG-Cz", "EEG-7","EEG-C4", "EEG-10","EEG-11","EEG-12","EEG-14"]).load_data()
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    # Definir el intervalo de tiempo para las épocas
    tmin, tmax = 2.0, 6.0

    # Definir los IDs de los eventos
    event_id = dict({
        "left": 7, 
        "right": 8, 
        "foot": 9, 
        "tongue": 10})

    # Crear épocas basadas en los eventos y el intervalo de tiempo
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    # Obtener los datos y etiquetas de las épocas
    data = epochs.get_data()
    labels = epochs.events[:,-1] - 7  # Ajustar las etiquetas
    X_tr = np.array(data)
    labels = np.array(labels)
    label = to_categorical(labels, num_classes=4)  # Convertir etiquetas a one-hot encoding
    return X_tr, label

# Función para preprocesar los datos
def preprocessing(X_tr, labels):
    # Escalar los datos entre 0 y 1
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X_tr, labels, test_size=0.35, random_state=120, stratify=labels)
    return x_train, x_test, y_train, y_test

# Definir la arquitectura del modelo EEGNet
def EEGNet(nb_classes, Chans=12, Samples=1001, regRate=0.4, dropoutRate=0.4, kernLength=64, numFilters=8, dropoutType='Dropout'):
    model = Sequential()
    
    F1 = numFilters
    D = 2
    F2 = numFilters * D

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

    # Primera capa: Conv2D
    model.add(Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 4)))
    model.add(dropoutType(dropoutRate))

    # Segunda capa: SeparableConv2D
    model.add(SeparableConv2D(F2, (1, 16), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 8)))
    model.add(dropoutType(dropoutRate))

    # Capa final: Dense
    model.add(Flatten())
    model.add(Dense(nb_classes, kernel_constraint=max_norm(regRate)))
    model.add(Activation('softmax'))

    return model

# Función para entrenar el modelo
def train_model(model, x_train, y_train, x_test, y_test):
    # Compilar el modelo con el optimizador Adam y la función de pérdida de entropía cruzada categórica
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Entrenar el modelo
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=120, shuffle=True)
    return history

# Función para realizar la división k-fold de los datos
def k_fold_split(X_tr, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X_tr):
        x_train, x_test = X_tr[train_index], X_tr[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        yield x_train, x_test, y_train, y_test

# Función para evaluar el modelo
def evaluate_model(model, x_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión')
    plt.show()

    # Mostrar el reporte de clasificación
    print(classification_report(y_test_labels, y_pred_labels))
    print(f"Accuracy: {np.mean(y_pred_labels == y_test_labels)}")

# Función principal
def main():
    # Definir la lista de archivos GDF
    file_paths = ['BCICIV_2a_gdf/A03T.gdf', 'BCICIV_2a_gdf/A01T.gdf', 'BCICIV_2a_gdf/A05T.gdf', 'BCICIV_2a_gdf/A02T.gdf', 'BCICIV_2a_gdf/A06T.gdf', 
                  'BCICIV_2a_gdf/A07T.gdf', 'BCICIV_2a_gdf/A08T.gdf', 'BCICIV_2a_gdf/A09T.gdf']
    
    # Listas para almacenar los datos de entrenamiento y prueba de todos los archivos
    all_x_train = []
    all_x_test = []
    all_y_train = []
    all_y_test = []

    # Leer y preprocesar los datos de cada archivo
    for file_path in file_paths:
        X_tr, labels = read_data(file_path)
        for x_train, x_test, y_train, y_test in k_fold_split(X_tr, labels, n_splits=7):
            x_train, x_test, y_train, y_test = preprocessing(X_tr, labels)
            all_x_train.append(x_train)
            all_x_test.append(x_test)
            all_y_train.append(y_train)
            all_y_test.append(y_test)

    # Combinar todos los datos de entrenamiento y prueba
    combined_x_train = np.concatenate(all_x_train)
    combined_x_test = np.concatenate(all_x_test)
    combined_y_train = np.concatenate(all_y_train)
    combined_y_test = np.concatenate(all_y_test)

    # Añadir una dimensión extra a los datos para que se ajusten al modelo
    combined_x_train = combined_x_train[..., np.newaxis]
    combined_x_test = combined_x_test[..., np.newaxis]

    # Crear y entrenar el modelo
    model = EEGNet(nb_classes=4)
    train_model(model, combined_x_train, combined_y_train, combined_x_test, combined_y_test)
    evaluate_model(model, combined_x_test, combined_y_test)

    # Guardar el modelo en formato JSON y los pesos en un archivo HDF5
    model_json = model.to_json()
    with open("EEGNET_four_classes.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("EEGNET_four_classes.h5")
    print("Saved model to disk")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
