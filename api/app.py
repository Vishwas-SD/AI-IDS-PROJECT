"""
REST API for AI-Based Intrusion Detection System
"""

import sys
import os

# Fix paths for Windows
PROJECT_ROOT = r'C:\Users\Chethan\Desktop\AI_IDS_PROJECT'
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from flask import Flask, request, jsonify
from predict import IDSPredictor
import json

app = Flask(__name__)

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
        'status': 'running',
        'endpoints': {
            'GET /': 'System info',
            'POST /predict': 'Predict single traffic record',
            'GET /health': 'Health check',
            'GET /model-info': 'Model performance info'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})


@app.route('/model-info', methods=['GET'])
def model_info():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No results found. Train the model first.'}), 404


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
    print("\n" + "="*60)
    print("  AI-BASED INTRUSION DETECTION SYSTEM - API SERVER")
    print("="*60)
    print("  Starting server on http://0.0.0.0:5000")
    print("  Press CTRL+C to stop")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)