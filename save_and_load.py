import joblib
import os
from keras.models import model_from_json
from predictor import Predictor
from pytorch_tabnet.tab_model import TabNetClassifier

SAVE_DIR = 'saved_models'
PREDICTORS_META_FILE = 'predictors_meta.pkl'
LABEL_ENCODERS_FILE = 'label_encoders.pkl'
SCALER_FILE = 'scaler.pkl'


predictors_meta = {}
predictors = {}


def save_model(model, model_name, model_filename):
    if model_name == 'MLP':
        model_json = model.to_json()
        with open(model_filename + '.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(model_filename + '.weights.h5')
    elif model_name == 'TabNet':
        model.save_model(model_filename)
    else:
        joblib.dump(model, model_filename + '.pkl')


def save_predictors_meta():
    with open(PREDICTORS_META_FILE, 'wb') as f:
        joblib.dump(predictors_meta, f)


def load_model_from_file(model_name, model_filename):
    if model_name == 'MLP':
        with open(model_filename + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_filename + '.weights.h5')
        return model
    elif model_name == 'TabNet':
        model = TabNetClassifier()
        model.load_model(model_filename + '.zip')
        return model
    else:
        return joblib.load(model_filename + '.pkl')


def save_label_encoders(encoders, filename):
    with open(filename, 'wb') as f:
        joblib.dump(encoders, f)


def load_label_encoders(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)
    

def save_scaler(encoders, filename):
    with open(filename, 'wb') as f:
        joblib.dump(encoders, f)


def load_scaler(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)


def load_predictors_meta():
    if os.path.exists(PREDICTORS_META_FILE):
        with open(PREDICTORS_META_FILE, 'rb') as f:
            predictors_meta = joblib.load(f)
    else:
        predictors_meta = {}
    for key, meta in predictors_meta.items():
        model_name = meta['model_name']
        model_filename = meta['model_filename']
        model = load_model_from_file(model_name, model_filename)
        best_accuracy = meta['best_accuracy']
        
        label_encoders_filename = f"{SAVE_DIR}/{key}_{LABEL_ENCODERS_FILE}"
        if os.path.exists(label_encoders_filename):
            label_encoders = load_label_encoders(label_encoders_filename)
        else:
            label_encoders = None
        
        scaler_filename = f"{SAVE_DIR}/{key}_{SCALER_FILE}"
        if os.path.exists(scaler_filename):
            scaler = load_scaler(scaler_filename)
        else:
            scaler = None

        predictor = Predictor(
            dataset=None,  
            label=meta['label'],
            categorical_features=meta['categorical_features'],
            numerical_features=meta['numerical_features']
        )
        predictor.model_name = model_name
        predictor.model = model
        predictor.label_encoders = label_encoders
        predictor.scaler = scaler
        predictor.best_accuracy = best_accuracy

        predictors[key] = predictor