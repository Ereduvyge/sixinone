import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_floatx('float32')
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from typing import Union


class Predictor:
    models = {}
    model = None
    model_name = 'Not defined'
    best_accuracy = 0

    def __init__(self, dataset: pd.DataFrame, label: str, categorical_features: Union[list[str], None], numerical_features: Union[list[str], None]):
        self.data = dataset
        self.label = label
        self.categorical_features = categorical_features if categorical_features else []
        self.numerical_features = numerical_features if numerical_features else []


    def get_models(self):
        self.data = self.data[[self.label, *self.categorical_features, *self.numerical_features]]
        self.data = self.data.dropna(how='any')

        self.label_encoders = {_: LabelEncoder() for _ in [self.label, *self.categorical_features]}
        self.scaler = MinMaxScaler()

        # Кодирование целевой переменной (label)
        self.data[self.label] = self.label_encoders[self.label].fit_transform(self.data[self.label])

        # Кодирование категориальных признаков
        for col in self.categorical_features:
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        self.data[self.numerical_features] = self.scaler.fit_transform(self.data[self.numerical_features])

        X = self.data.drop(self.label, axis=1)
        y = self.data[self.label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.models = {}

        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)
        perceptron_pred = perceptron.predict(X_test)

        self.models['Perceptron'] = [perceptron, accuracy_score(y_test, perceptron_pred)]

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        log_reg_pred = log_reg.predict(X_test)

        self.models['LogisticRegression'] = [log_reg, accuracy_score(y_test, log_reg_pred)]

        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        self.models['RandomForestClassifier'] = [rf_model, accuracy_score(y_test, rf_pred)]

        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

        self.models['XGBClassifier'] = [xgb, accuracy_score(y_test, xgb_pred)]

        # Определение архитектуры нейронной сети
        early_stop = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(np.unique(y_train)), activation='softmax'))

        # Компиляция модели
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Обучение модели
        model.fit(X_train, y_train, epochs=32, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

        self.models['MLP'] = [model, model.evaluate(X_test, y_test)[1]]


        # Индексы категориальных признаков
        cat_idxs = [X.columns.get_loc(col) for col in self.categorical_features]
        cat_dims = [len(self.label_encoders[col].classes_) for col in self.categorical_features]

        t_X = X.values
        t_y = y.values

        # Разделение данных на обучающую и тестовую выборки
        t_X_train, t_X_test, t_y_train, t_y_test = train_test_split(t_X, t_y, test_size=0.2, random_state=42)

        # Создание и обучение модели TabNet
        clf = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            cat_idxs=cat_idxs, cat_dims=cat_dims,
            cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax'  # для TabNetClassifier используется 'sparsemax'
        )

        clf.fit(
            t_X_train, t_y_train,
            max_epochs=12,
            patience=10,
            batch_size=128,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        self.models['TabNet'] = [clf, accuracy_score(t_y_test, clf.predict(t_X_test))]

    
    def choose_option(self):
        best_option = max(self.models.items(), key=lambda x: x[1][1])
        self.model_name = best_option[0]
        self.model = best_option[1][0]
        self.best_accuracy = best_option[1][1]

    
    def predict(self, values: dict) -> np.array:
        predict_data = pd.DataFrame.from_dict(values)
        predict_data = predict_data[[*self.categorical_features, *self.numerical_features]]
        for col in self.categorical_features:
            predict_data[col] = self.label_encoders[col].fit_transform(predict_data[col])
        predict_data[self.numerical_features] = self.scaler.fit_transform(predict_data[self.numerical_features])

        if self.model_name == 'TabNet':
            predict_data = predict_data.values
            
        predictions = self.model.predict(predict_data)
        if predictions.ndim > 1:
            predictions = [predictions[i].argmax(axis=-1) for i in range(len(predictions))]
        
        results = self.label_encoders[self.label].inverse_transform(predictions)
        return results