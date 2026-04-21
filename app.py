from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

knn_model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')
EXPECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang', 'oldpeak'
]

CATEGORICAL_FIELDS = {
    'sex': [
        {'value': 0, 'label': 'Female'},
        {'value': 1, 'label': 'Male'},
    ],
    'cp': [
        {'value': 0, 'label': 'Typical angina'},
        {'value': 1, 'label': 'Atypical angina'},
        {'value': 2, 'label': 'Non-anginal pain'},
        {'value': 3, 'label': 'Asymptomatic'},
    ],
    'fbs': [
        {'value': 0, 'label': 'No'},
        {'value': 1, 'label': 'Yes'},
    ],
    'restecg': [
        {'value': 0, 'label': 'Normal'},
        {'value': 1, 'label': 'ST-T abnormality'},
        {'value': 2, 'label': 'Left ventricular hypertrophy'},
    ],
    'exang': [
        {'value': 0, 'label': 'No'},
        {'value': 1, 'label': 'Yes'},
    ],
}


def run_prediction(payload):
    input_df = pd.DataFrame([payload])
    input_df = input_df[EXPECTED_COLUMNS]
    scaled_features = scaler.transform(input_df)

    prediction = knn_model.predict(scaled_features)
    prediction_proba = knn_model.predict_proba(scaled_features)
    heart_attack_probability = float(prediction_proba[0][1])

    if heart_attack_probability >= 0.8:
        risk_level = 'Tinggi'
        recommendation = 'Segera konsultasikan hasil ini ke dokter atau fasilitas kesehatan untuk evaluasi lebih lanjut.'
    elif heart_attack_probability >= 0.5:
        risk_level = 'Sedang'
        recommendation = 'Disarankan melakukan pemeriksaan medis dan mulai lebih ketat memantau tekanan darah, kolesterol, dan gaya hidup.'
    else:
        risk_level = 'Rendah'
        recommendation = 'Tetap jaga pola makan, olahraga teratur, tidur cukup, dan lakukan pemeriksaan kesehatan secara berkala.'

    return {
        'prediction': int(prediction[0]),
        'probability_no_heart_attack': float(prediction_proba[0][0]),
        'probability_heart_attack': heart_attack_probability,
        'risk_level': risk_level,
        'recommendation': recommendation
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', categorical_fields=CATEGORICAL_FIELDS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        return jsonify(run_prediction(data))

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        data = {
            column: float(request.form[column])
            for column in EXPECTED_COLUMNS
        }
        result = run_prediction(data)
        return render_template(
            'index.html',
            result=result,
            values=data,
            categorical_fields=CATEGORICAL_FIELDS
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=str(e),
            values=request.form,
            categorical_fields=CATEGORICAL_FIELDS
        ), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
