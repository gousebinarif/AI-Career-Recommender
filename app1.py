from flask import Flask, request, jsonify, render_template, g
import joblib, os, sqlite3, json, subprocess, threading, time
import numpy as np

DB_PATH = 'recommender.db'
PREPROCESSOR_FILE = 'preprocessor.joblib'
RF_FILE = 'rf_model.joblib'
NN_DIR = 'nn_model'  # tensorflow saved model dir
LABEL_MAP_FILE = 'label_map.joblib'

app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT UNIQUE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS user_profiles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      profile_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS recommendations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_profile_id INTEGER,
      result_json TEXT,
      model_version TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(user_profile_id) REFERENCES user_profiles(id)
    );
    """)
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def models_exist():
    return os.path.exists(PREPROCESSOR_FILE) and os.path.exists(RF_FILE) and os.path.exists(LABEL_MAP_FILE) and os.path.exists(NN_DIR)

# Load models if available
pre = rf = nn = label_map = None
if models_exist():
    try:
        pre = joblib.load(PREPROCESSOR_FILE)
        rf = joblib.load(RF_FILE)
        import tensorflow as tf
        nn = tf.keras.models.load_model(NN_DIR)
        label_map = joblib.load(LABEL_MAP_FILE)
        print('Models loaded.')
    except Exception as e:
        print('Error loading models:', e)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_input(payload):
    import pandas as pd
    df = pd.DataFrame([payload])
    X = pre.transform(df)
    return X

@app.route('/predict', methods=['POST'])
def predict():
    global pre, rf, nn, label_map
    if not models_exist():
        return jsonify({'error':'Models not found. POST to /train to train models first.'}), 400
    try:
        payload = request.get_json()
        X = preprocess_input(payload)
        rf_proba = rf.predict_proba(X)[0]
        nn_proba = nn.predict(X)[0]
        ensemble_proba = (rf_proba + nn_proba) / 2.0
        top_indices = ensemble_proba.argsort()[::-1][:5]
        results = [{'career': label_map[idx], 'score': float(ensemble_proba[idx])} for idx in top_indices]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_profile', methods=['POST'])
def save_profile():
    # payload expected: {user: {name,email}, profile: {...}, results: [...]}
    data = request.get_json()
    user = data.get('user', {})
    profile = data.get('profile', {})
    results = data.get('results', [])
    db = get_db()
    cursor = db.cursor()
    # upsert user by email if provided
    user_id = None
    if 'email' in user:
        cursor.execute('SELECT id FROM users WHERE email = ?', (user['email'],))
        row = cursor.fetchone()
        if row:
            user_id = row['id']
        else:
            cursor.execute('INSERT INTO users (name,email) VALUES (?,?)', (user.get('name'), user.get('email')))
            user_id = cursor.lastrowid
    else:
        cursor.execute('INSERT INTO users (name,email) VALUES (?,?)', (user.get('name'), None))
        user_id = cursor.lastrowid
    cursor.execute('INSERT INTO user_profiles (user_id, profile_json) VALUES (?,?)', (user_id, json.dumps(profile)))
    profile_id = cursor.lastrowid
    cursor.execute('INSERT INTO recommendations (user_profile_id, result_json, model_version) VALUES (?,?,?)', (profile_id, json.dumps(results), 'v1'))
    db.commit()
    return jsonify({'status':'saved','user_id':user_id,'profile_id':profile_id})

@app.route('/history/<int:user_id>', methods=['GET'])
def history(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT up.id as profile_id, up.profile_json, r.result_json, r.created_at FROM user_profiles up JOIN recommendations r ON r.user_profile_id = up.id WHERE up.user_id = ? ORDER BY r.created_at DESC', (user_id,))
    rows = cursor.fetchall()
    out = []
    for row in rows:
        out.append({'profile_id': row['profile_id'], 'profile': json.loads(row['profile_json']), 'results': json.loads(row['result_json']), 'created_at': row['created_at']})
    return jsonify(out)

@app.route('/train', methods=['POST'])
def train_endpoint():
    # Run train.py synchronously and return output
    if os.path.exists('train.py'):
        try:
            # run training as subprocess, capture output
            proc = subprocess.Popen([sys.executable, 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = []
            for line in proc.stdout:
                output.append(line)
            proc.wait()
            # reload models if trained
            global pre, rf, nn, label_map
            try:
                pre = joblib.load(PREPROCESSOR_FILE)
                rf = joblib.load(RF_FILE)
                import tensorflow as tf
                nn = tf.keras.models.load_model(NN_DIR)
                label_map = joblib.load(LABEL_MAP_FILE)
            except Exception as e:
                return jsonify({'status':'trained_but_reload_failed','error':str(e),'log':''.join(output)}), 500
            return jsonify({'status':'trained','log':''.join(output)})
        except Exception as e:
            return jsonify({'error':str(e)}), 500
    else:
        return jsonify({'error':'train.py not found on server'}), 500

if __name__ == '__main__':
    # initialize DB and start app
    with app.app_context():
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
