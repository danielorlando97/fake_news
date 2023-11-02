import pandas as pd
import spacy
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, accuracy_score

df = pd.read_csv('images_embeddings.csv')
dff = pd.read_csv('dataset_curso.csv')

dff = pd.merge(df, dff, how='outer', on='id')

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def get_emb(text):
    doc = nlp(text)
    return doc.vector

dff['text_emb'] = dff['clean_title'].apply(get_emb)


def transform(x):
    if type(x) == float:
        return np.zeros(512)
    if type(x) == str:
        return np.array(ast.literal_eval(x))

    return x

dff['img_embedding'] = dff['embedding'].apply(transform)

def transform(row):
    result = [x for x in row['text_emb']]
    return np.array(result + [x for x in row['img_embedding']])

dff['total_emb'] = dff.apply(transform, axis=1)

matriz_numpy = np.vstack(dff['total_emb'].to_numpy())
x_shape = matriz_numpy.shape[1]
print(int(x_shape/10), int(x_shape * 3/2))

callback = tf.keras.callbacks.EarlyStopping(
      monitor='loss',
      patience=4,
      restore_best_weights=True,
      start_from_epoch=0
  )

def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(matriz_numpy, dff['2_way_label'], test_size=0.2, random_state=42)
    layers_num = trial.suggest_int('num_layers', 2, 6)
    input_layer = layers.Input(shape=(x_shape,))
    last_layer = input_layer
    for index in range(layers_num):
      units = trial.suggest_int(f'n_units_l{index}', int(x_shape/10), int(x_shape * 3/2))
      activation = trial.suggest_categorical(
          f'dense_{index}_activation',
           ['relu', 'linear', 'sigmoid', 'softmax']
      )
      last_layer = layers.Dense(units, activation=activation)(last_layer)
      dropout_flag = trial.suggest_categorical(f'dropout_{index}', [True, False])
      if dropout_flag:
        dropout_rate = trial.suggest_float(f'dropout_{index}_rate', 0.0, 0.5)
        last_layer = layers.Dropout(dropout_rate)(last_layer)
    output = layers.Dense(1, activation='softmax')(last_layer)
    model = keras.models.Model(
        inputs=input_layer,
        outputs=output
    )
    # Compilar el modelo


    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=trial.suggest_float(f'learning_rate', 0.00001, 0.01)),
        loss="mean_squared_error",
        metrics=['acc']
    )
    model.fit(X_train, y_train, epochs = 20, callbacks=[callback])
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

print('Start fitting')
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
