"""
REST API for AI-Based Intrusion Detection System
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import IDSPredictor
import json

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'preprocessor.pkl')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'models', 'results.json')

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = IDSPredictor(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH
        )
    return predictor

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'system': 'AI-Based Intrusion Detection System',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400
        p = get_predictor()
        result = p.predict_single(data)
        return jsonify({
            'success': True,
            'result': result,
            'alert_message': '⚠️ INTRUSION DETECTED!' if result['alert'] else '✅ Traffic is Normal'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)