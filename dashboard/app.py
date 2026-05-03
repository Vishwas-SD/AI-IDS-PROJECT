from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  IDS DASHBOARD — http://127.0.0.1:8080")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False)