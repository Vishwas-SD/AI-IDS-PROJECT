"""
REST API for AI-Based Intrusion Detection System
"""
import sys
import os
import json
import subprocess
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import IDSPredictor
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
# ── API KEY ──────────────────────────────────────────────
API_KEY = "oxvenom-ids-2026"

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized — Invalid API Key', 'code': 401}), 401
        return f(*args, **kwargs)
    return decorated

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'preprocessor.pkl')

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
@require_api_key
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
    
    train_logs = []
training_running = False

@app.route('/train/start', methods=['POST'])
@require_api_key
def train_start():
    global training_running, train_logs
    if training_running:
        return jsonify({'status': 'already running'})

    data = request.get_json() or {}
    options = data.get('options', {})

    training_running = True
    train_logs = []

    def run_training():
        global training_running, train_logs
        train_script = os.path.join(PROJECT_ROOT, 'src', 'train_model.py')
        python = os.path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe')

        # Pass options as environment variables
        env = os.environ.copy()
        env['DETAILED_ANALYSIS'] = '1' if options.get('detailed') else '0'
        env['SAVE_MODELS']       = '1' if options.get('save') else '0'
        env['CROSS_VALIDATION']  = '1' if options.get('crossval') else '0'

        process = subprocess.Popen(
            [python, train_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        for line in process.stdout:
            line = line.strip()
            if line:
                train_logs.append(line)
        process.wait()
        train_logs.append('[✓] Training complete!')
        training_running = False

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/train/logs', methods=['GET'])
@require_api_key
def get_train_logs():
    return jsonify({
        'logs': train_logs,
        'running': training_running
    })
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
@require_api_key
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({'success': True, 'filename': filename, 'path': filepath})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)