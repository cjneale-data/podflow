from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import requests
import tempfile
import os
import pickle
from pathlib import Path
from datetime import datetime
import warnings
import gc
import os
from dotenv import load_dotenv

load_dotenv() 
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Database Models ---
class Podcast(db.Model):
    __tablename__ = 'podcasts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    author = db.Column(db.String(255))
    feed_url = db.Column(db.String(500), unique=True)
    artwork_url = db.Column(db.String(500))
    genre = db.Column(db.String(100))
    topics = db.Column(db.Text)
    episodes = db.relationship('Episode', backref='podcast', lazy=True)

class Episode(db.Model):
    __tablename__ = 'episodes'
    id = db.Column(db.Integer, primary_key=True)
    podcast_id = db.Column(db.Integer, db.ForeignKey('podcasts.id'))
    title = db.Column(db.String(255))
    description = db.Column(db.Text)
    pub_date = db.Column(db.DateTime)
    audio_url = db.Column(db.String(500))
    guid = db.Column(db.String(500))
    duration = db.Column(db.Integer)
    chapters = db.relationship('Chapter', backref='episode', lazy=True)

class Chapter(db.Model):
    __tablename__ = 'chapters'
    id = db.Column(db.Integer, primary_key=True)
    episode_id = db.Column(db.Integer, db.ForeignKey('episodes.id'))
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)
    title = db.Column(db.String(255))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============================================================
# DEEP LEARNING MODEL
# ============================================================
class PodcastSegmenter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(PodcastSegmenter, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

# ============================================================
# UTILS
# ============================================================
model = None
scaler = None
device = torch.device('cpu') # Force CPU for stability on small droplets
CONFIG = {
    'sample_rate': 16000,
    'hop_length_sec': 2,
    'sequence_length': 128
}

def load_model_artifacts():
    global model, scaler
    try:
        if not Path('models/dl_scaler.pkl').exists(): return
        with open('models/dl_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        input_dim = scaler.mean_.shape[0]
        model = PodcastSegmenter(input_dim=input_dim).to(device)
        
        if not Path('models/best_deep_model.pth').exists(): return
        model.load_state_dict(torch.load('models/best_deep_model.pth', map_location=device))
        model.eval()
        print(f"✓ Model Loaded (Dim: {input_dim})")
    except Exception as e:
        print(f"✗ Load Error: {e}")

def download_audio(url, max_size_mb=500):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Stream download to avoid memory spike
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise Exception(f"DL Error: {e}")

# ============================================================
# LOW-MEMORY FEATURE EXTRACTION (CHUNKED)
# ============================================================
def extract_features_from_chunk(y, sr, hop_length):
    """Process a single 10-min chunk of audio"""
    frame_length = min(2048, len(y))
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    target_len = energy.shape[0]
    
    def fix(arr):
        if arr.ndim == 1:
            if arr.shape[0] > target_len: return arr[:target_len]
            elif arr.shape[0] < target_len: return np.pad(arr, (0, target_len - arr.shape[0]))
            return arr
        else: 
            if arr.shape[-1] > target_len: return arr[..., :target_len]
            elif arr.shape[-1] < target_len: return np.pad(arr, ((0,0), (0, target_len - arr.shape[-1])))
            return arr

    features = {}
    features['energy'] = energy
    
    S = np.abs(librosa.stft(y))
    features['zcr'] = fix(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0])
    features['centroid'] = fix(librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=hop_length)[0])
    features['flatness'] = fix(librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0])
    features['rolloff'] = fix(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0])
    features['spectral_contrast'] = fix(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr, hop_length=hop_length), axis=0))
    features['onset_strength'] = fix(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length))
    features['chroma_mean'] = fix(np.mean(librosa.feature.chroma_stft(S=S, sr=sr, hop_length=hop_length), axis=0))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_d1 = librosa.feature.delta(mfcc, order=1)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    
    mfcc = fix(mfcc)
    mfcc_d1 = fix(mfcc_d1)
    mfcc_d2 = fix(mfcc_d2)
    
    for i in range(13):
        features[f"mfcc_{i+1}"] = mfcc[i]
        features[f"mfcc_d1_{i+1}"] = mfcc_d1[i]
        features[f"mfcc_d2_{i+1}"] = mfcc_d2[i]
        
    return pd.DataFrame(features)

def extract_and_process_features(audio_path):
    # 1. Get Duration first (without loading file)
    # librosa.get_duration(path=...) works in newer versions, 
    # otherwise we load with sr=None just for header but that's risky. 
    # Use soundfile via librosa if available.
    try:
        total_duration = librosa.get_duration(path=audio_path)
    except:
        # Fallback for older librosa: load just 1 sec to get SR, estimate from file size? 
        # Safest: Load with soundfile directly if needed, but let's try standard load
        y_dummy, sr_dummy = librosa.load(audio_path, sr=None, duration=0.1)
        total_duration = librosa.get_duration(y=y_dummy, sr=sr_dummy) * (os.path.getsize(audio_path)/1000) # Rough guess
        # Actually, let's just assume we loop until data ends
        total_duration = 14400 # Max 4 hours default

    sr = CONFIG['sample_rate']
    hop_length = int(sr * CONFIG['hop_length_sec'])
    
    # 2. Chunked Loading & Feature Extraction
    CHUNK_SEC = 600 # Process 10 mins at a time
    df_chunks = []
    
    current_offset = 0
    while True:
        # Load only 10 mins
        y, _ = librosa.load(audio_path, sr=sr, offset=current_offset, duration=CHUNK_SEC)
        
        if len(y) == 0:
            break
            
        # Extract features for this small chunk
        df_chunk = extract_features_from_chunk(y, sr, hop_length)
        df_chunks.append(df_chunk)
        
        # Explicit cleanup
        del y
        gc.collect()
        
        current_offset += CHUNK_SEC
        # If we read less than we asked for, we are done
        if len(df_chunk) < (CHUNK_SEC / CONFIG['hop_length_sec']) * 0.9: 
            break

    # 3. Combine Chunks
    if not df_chunks: raise ValueError("No audio data loaded")
    df = pd.concat(df_chunks, ignore_index=True)
    
    # 4. Global Feature Engineering (Rolling Context)
    # Now that we have the full timeline in low-memory DF, we apply rolling windows
    HOP_SEC = 2
    SHORT = 12; MID = 30; LONG = 90
    
    df['delta_energy'] = df['energy'].diff().fillna(0)
    df['delta_onset'] = df['onset_strength'].diff().fillna(0)
    df['delta_flatness'] = df['flatness'].diff().fillna(0)
    
    for col in ['energy', 'zcr', 'spectral_contrast', 'flatness', 'chroma_mean']:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=SHORT).mean().fillna(0)
        df[f'{col}_deviation'] = df[col] - df[f'{col}_rolling_mean']
        
    df['relative_position'] = df.index / len(df)
    
    mfcc_cols = [f'mfcc_{i+1}' for i in range(13)]
    df['mfcc_magnitude_change'] = df[mfcc_cols].diff().pow(2).sum(axis=1).pow(0.5).fillna(0)
    df['prev_frame_silence'] = df['energy'].shift(1).apply(lambda x: 1 if x < 0.02 else 0).fillna(0)
    df['spectral_flux'] = df['onset_strength'].diff().abs().fillna(0)
    
    window_lag = 2
    for col in ['spectral_contrast', 'energy']:
        fut = df[col].shift(-window_lag)
        past = df[col].shift(window_lag)
        df[f'{col}_future_past_diff'] = (fut - past).abs().fillna(0)
        
    df['time_since_silence_sec'] = df.groupby((df['energy']<0.02).cumsum()).cumcount().apply(lambda x: x*HOP_SEC)
    df['time_since_mfcc_change_sec'] = df.groupby((df['mfcc_magnitude_change']<=0.05).cumsum()).cumcount().apply(lambda x: x*HOP_SEC)
    
    df['energy_60s_mean'] = df['energy'].rolling(MID).mean().fillna(0)
    df['contrast_60s_mean'] = df['spectral_contrast'].rolling(MID).mean().fillna(0)
    df['energy_180s_mean'] = df['energy'].rolling(LONG).mean().fillna(0)
    df['contrast_180s_mean'] = df['spectral_contrast'].rolling(LONG).mean().fillna(0)
    
    df['energy_lt_contrast'] = (df['energy_60s_mean'] - df['energy_180s_mean']).abs()
    df['contrast_lt_contrast'] = (df['contrast_60s_mean'] - df['contrast_180s_mean']).abs()
    
    # Filter columns based on scaler
    try:
        if hasattr(scaler, 'feature_names_in_'):
            X = df[scaler.feature_names_in_]
        else:
            X = df.select_dtypes(include=[np.number])
        X_scaled = scaler.transform(X)
        
        # Determine actual duration from processed frames
        processed_duration = len(df) * CONFIG['hop_length_sec']
        return X_scaled, processed_duration
    except Exception as e:
        print(f"Scaling Error: {e}")
        raise

# ============================================================
# PREDICTION (MEMORY OPTIMIZED)
# ============================================================
def predict_chapters(X_scaled, duration, threshold=0.75):
    gc.collect()
    
    seq_len = CONFIG['sequence_length']
    num_frames = len(X_scaled)
    n_features = X_scaled.shape[1]
    
    # Pad
    padding = np.zeros((seq_len - 1, n_features))
    X_padded = np.vstack([padding, X_scaled])
    
    # Sliding window view (Zero Copy!)
    from numpy.lib.stride_tricks import sliding_window_view
    sequences_view = sliding_window_view(X_padded, (seq_len, n_features)).squeeze(axis=1)
    
    batch_size = 128
    probs_list = []
    
    with torch.no_grad():
        for i in range(0, len(sequences_view), batch_size):
            batch_numpy = sequences_view[i : i+batch_size]
            X_tensor = torch.tensor(batch_numpy, dtype=torch.float32).to(device)
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs_list.extend(probs)
            del X_tensor
            
    probabilities = np.array(probs_list)
    is_boundary = (probabilities >= threshold).astype(int)
    boundary_indices = np.where(is_boundary == 1)[0]
    
    MIN_DIST_FRAMES = 150 
    final_indices = []
    last_idx = 0
    
    for idx in boundary_indices:
        if idx - last_idx >= MIN_DIST_FRAMES:
            final_indices.append(idx)
            last_idx = idx
            
    hop = CONFIG['hop_length_sec']
    timestamps = [i * hop for i in final_indices]
    
    chapters = []
    current_start = 0.0
    
    for i, ts in enumerate(timestamps):
        if ts > current_start:
            chapters.append({
                'start_time': float(round(current_start, 1)),
                'end_time': float(round(ts, 1)),
                'confidence': float(probabilities[final_indices[i]]),
                'title': f"Chapter {i+1}"
            })
            current_start = ts
            
    if duration > current_start:
        chapters.append({
            'start_time': float(round(current_start, 1)),
            'end_time': float(round(duration, 1)),
            'confidence': 0.0,
            'title': f"Chapter {len(chapters)+1}"
        })
    
    return chapters

# ============================================================
# API ROUTES
# ============================================================

# --- Database Connection Helper ---
import psycopg2
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname="podcasts", user="myuser", password="password", host="localhost"
        )
    except: return None

@app.route('/api/filter-options', methods=['GET'])
def filter_options_route():
    conn = get_db_connection()
    if conn is None: return jsonify({}), 500
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT genre FROM podcasts WHERE genre IS NOT NULL ORDER BY genre")
        genres = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT topics FROM podcasts WHERE topics IS NOT NULL")
        unique_topics = set()
        for row in cur.fetchall():
            if row[0]: unique_topics.update([t.strip() for t in row[0].split(',')])
        return jsonify({"genres": genres, "topics": sorted(list(unique_topics))})
    except: return jsonify({}), 500
    finally: conn.close()

@app.route('/api/random-chapters', methods=['GET'])
def random_chapters_route():
    genre = request.args.get('genre')
    topic = request.args.get('topic')
    conn = get_db_connection()
    if conn is None: return jsonify([]), 500
    
    try:
        cur = conn.cursor()
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=7)
        
        topic_param = f"%{topic}%" if topic else None
        
        query = """
            WITH RecentEpisodesWithChapters AS (
                SELECT DISTINCT e.id
                FROM episodes e
                JOIN chapters c ON e.id = c.episode_id
                JOIN podcasts p ON e.podcast_id = p.id
                WHERE e.pub_date >= %s
                AND (%s::text IS NULL OR p.genre = %s)
                AND (%s::text IS NULL OR p.topics ILIKE %s)
            )
            SELECT e.id, p.title, e.title, e.audio_url, p.artwork_url
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            JOIN RecentEpisodesWithChapters rec ON e.id = rec.id
            ORDER BY RANDOM()
            LIMIT 1;
        """
        cur.execute(query, (cutoff, genre, genre, topic, topic_param))
        ep_data = cur.fetchone()
        
        if not ep_data: return jsonify([])
        
        ep_id, p_title, e_title, audio, art = ep_data
        cur.execute("SELECT start_time, end_time, title FROM chapters WHERE episode_id = %s ORDER BY start_time", (ep_id,))
        chapters = []
        for s, e, t in cur.fetchall():
            chapters.append({
                "chapter_title": t, "podcast_title": p_title, "episode_title": e_title,
                "artwork_url": art, "audio_url": audio, "start_time": s, "end_time": e
            })
        return jsonify(chapters)
    except Exception as e:
        print(e)
        return jsonify([])
    finally: conn.close()

if __name__ == '__main__':
    load_model_artifacts()
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)
