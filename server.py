import os
import pandas as pd
import json
import requests
from io import StringIO
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash

from predictor import Predictor
from save_and_load import *

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Путь для сохранения загруженных файлов
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        field_mapping = request.form.to_dict()
        return redirect(url_for('view', key=field_mapping['key']))
    
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('Файл загружен. Переходите к выборке.')
            return redirect(url_for('map_fields', filename=filename))
        else:
            flash('Неверный тип файла. Только .csv файлы разрешены.')
            return redirect(url_for('upload_file'))
    
    return render_template('upload.html')

@app.route('/map_fields/<filename>', methods=['GET', 'POST'])
def map_fields(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    columns = df.columns.tolist()

    if request.method == 'POST':
        field_mapping = request.form.to_dict()
        label = [key for key, value in field_mapping.items() if value == 'label']
        if len(label) != 1:
            return redirect(url_for('map_fields', filename=filename))
        numerical = [key for key, value in field_mapping.items() if value == 'numerical']
        categorical = [key for key, value in field_mapping.items() if value == 'categorical']
        key = str(uuid.uuid4())

        url = 'http://127.0.0.1:5000/add_predictor'
        json_data = {
            'key': key,
            'label': label[0],
            'categorical_features': categorical,
            'numerical_features': numerical
        }
        files = {
            'dataset': open(filename, 'rb')
        }
        response = requests.post(url, files=files, data={'json': json.dumps(json_data)})
        if response.status_code == 200:
            try:
                os.remove(filepath)
            finally:
                return redirect(url_for('success', key=key))
        else:
            print(response.json())
            flash('Произошла ошибка. Попробуйте снова')
            try:
                os.remove(filepath)
            finally:
                return redirect(url_for('upload_file'))
        
    
    return render_template('mapping.html', columns=columns)

@app.route('/success/<key>')
def success(key):
    return render_template('success.html', key=key)


@app.route('/view/<key>')
def view(key):
    if key not in predictors:
        return f"404", 404
    predictor = predictors[key]
    return render_template('view.html', predictor=predictor, key=key)


@app.route('/add_predictor', methods=['POST'])
def add_predictor():
    try:
        json_data = json.loads(request.form.get('json'))
        
        predictor_key = json_data['key']
        label = json_data['label']
        categorical_features = json_data['categorical_features']
        numerical_features = json_data['numerical_features']

        uploaded_file = request.files.get('dataset')

        if uploaded_file:
            dataset_str = uploaded_file.read().decode('utf-8')
            dataset = pd.read_csv(StringIO(dataset_str))
        else:
            return jsonify({'error': 'No dataset file uploaded'}), 400
    except KeyError as e:
        return jsonify({"success": False, "error": f"Missing key in request data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    if predictor_key in predictors:
        return jsonify({"success": False, "error": f"Predictor with key '{predictor_key}' already exists"}), 400

    try:
        predictor = Predictor(
            dataset=dataset,
            label=label,
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )
        predictor.get_models()
        predictor.choose_option()

        model_name = predictor.model_name
        best_accuracy = predictor.best_accuracy
        model_filename = f"{SAVE_DIR}/{predictor_key}_{model_name}"

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        save_model(predictor.model, model_name, model_filename)
        
        label_encoders_filename = f"{SAVE_DIR}/{predictor_key}_{LABEL_ENCODERS_FILE}"
        save_label_encoders(predictor.label_encoders, label_encoders_filename)
        scaler_filename = f"{SAVE_DIR}/{predictor_key}_{SCALER_FILE}"
        save_scaler(predictor.scaler, scaler_filename)

        predictors_meta[predictor_key] = {
            'label': label,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'model_name': model_name,
            'model_filename': model_filename,
            'label_encoders_filename': label_encoders_filename,
            'scaler_filename': scaler_filename,
            'best_accuracy': best_accuracy,
        }
        save_predictors_meta()
        load_predictors_meta()

        return jsonify({"success": True, "message": f"Predictor '{predictor_key}' added successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        post_data = request.get_json()
        key = post_data['key']
        data = post_data['data']
    except KeyError:
        return jsonify({
            "success": False,
            "error": "Request must contain 'key' and 'data' fields"
        }), 400

    if key not in predictors:
        return jsonify({
            "success": False,
            "error": f"No predictor found for key '{key}'"
        }), 400

    predictor: Predictor = predictors[key]

    required_keys = [*predictor.categorical_features, *predictor.numerical_features]
    all_keys = set().union(*[d.keys() for d in data])

    for ind, d in enumerate(data):
        missing_keys = [key for key in required_keys if key not in d.keys()]
        if missing_keys:
            return jsonify({
                "success": False,
                "error": f"Required key(s) {', '.join(missing_keys)} missing in dictionary with index {ind}"
            }), 400

    dict_data = {}
    for key in required_keys + list(all_keys - set(required_keys)):
        if key in predictor.numerical_features:
            dict_data[key] = [float(d.get(key, None)) for d in data]
        else:
            dict_data[key] = [d.get(key, None) for d in data]

    try:
        predictions = predictor.predict(dict_data)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    return jsonify({
        "success": True,
        "via": f"{predictor.model_name} with approx. accuracy = {predictor.best_accuracy}",
        "labels": predictions.tolist()
    })


if __name__ == '__main__':
    try:
        load_predictors_meta()
    except FileNotFoundError:
        pass
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
