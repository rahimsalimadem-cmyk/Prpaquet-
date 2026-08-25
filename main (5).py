# -----------------------------------------------------------------------------
# OVIN MANAGER PRO - Version 9.0 (Complète & Automatisée)
# Laboratoir - Université Laval
# Conçu pour répondre aux défis du Pr. Éric R. Paquet
# -----------------------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import hashlib
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from PIL import Image
import io
import time
import numpy as np
import statsmodels.api as sm
import zipfile
import os
import uuid
from scipy.optimize import linprog
import joblib
import tempfile
from io import BytesIO

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Pour l'analyse exploratoire (optionnel)
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    profiling_available = True
except ImportError:
    profiling_available = False

# Traitement d'image et IA
import cv2
import mediapipe as mp

# Outil pour les coordonnées de clics
from streamlit_image_coordinates import streamlit_image_coordinates

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PHOTO_DIR = "photos_brebis"
MODEL_DIR = "models"
DATASET_DIR = "dataset"
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Initialisation MediaPipe Pose (API "Tasks", remplace l'ancienne API "solutions"
# qui n'est plus incluse dans les distributions récentes de mediapipe)
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_pose_landmarker = None

def _ensure_pose_model() -> bool:
    """Télécharge le modèle de détection de pose s'il n'est pas déjà présent localement."""
    if os.path.exists(POSE_MODEL_PATH):
        return True
    try:
        response = requests.get(POSE_MODEL_URL, timeout=60)
        response.raise_for_status()
        with open(POSE_MODEL_PATH, "wb") as f:
            f.write(response.content)
        return True
    except Exception:
        return False

def get_pose_landmarker():
    """Retourne une instance (singleton, créée à la demande) du détecteur de pose MediaPipe."""
    global _pose_landmarker
    if _pose_landmarker is None:
        if not _ensure_pose_model():
            return None
        base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker

class Config:
    APP_NAME = "Ovin Manager Pro"
    LABORATOIRE = "GenApAgiE"
    VERSION = "9.0"
    
    VERT = "#2E7D32"
    ORANGE = "#FF6F00"
    BLEU = "#1565C0"
    ROUGE = "#C62828"
    VIOLET = "#6A1B9A"
    CYAN = "#00838F"
    
    NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    ETALONS = {
        "baton_1m": {"nom": "Bâton 1m", "largeur": 1000, "hauteur": None},
        "a4": {"nom": "Feuille A4", "largeur": 210, "hauteur": 297},
        "carte": {"nom": "Carte bancaire", "largeur": 85.6, "hauteur": 53.98},
        "piece_100da": {"nom": "Pièce 100 DA", "diametre": 29.5}
    }
    
    RACES = {
        "Hamra": {"origine": "Atlas saharien", "aptitude": "Mixte", "genes": ["BMP15", "GDF9"]},
        "Ouled Djellal": {"origine": "Steppes algériennes", "aptitude": "Viande", "genes": ["MSTN", "IGF2"]},
        "Sidahou": {"origine": "Aurès", "aptitude": "Lait", "genes": ["LALBA", "CSN3", "DGAT1"]},
        "Rembi": {"origine": "Tell", "aptitude": "Mixte", "genes": ["BMP15", "LALBA"]},
        "Autre": {"origine": "Inconnue", "aptitude": "Variable", "genes": []}
    }
    
    GENES_ECONOMIQUES = {
        "BMP15": {"nom": "Bone Morphogenetic Protein 15", "chr": "X", "effet": "Fécondité"},
        "GDF9": {"nom": "Growth Differentiation Factor 9", "chr": "5", "effet": "Fécondité"},
        "BMPR1B": {"nom": "BMP Receptor 1B", "chr": "6", "effet": "Prolificité (Booroola)"},
        "MSTN": {"nom": "Myostatin", "chr": "2", "effet": "Hypertrophie musculaire"},
        "IGF2": {"nom": "Insulin-like Growth Factor 2", "chr": "2", "effet": "Croissance"},
        "GH": {"nom": "Growth Hormone", "chr": "19", "effet": "Croissance"},
        "GHR": {"nom": "Growth Hormone Receptor", "chr": "16", "effet": "Efficacité alimentaire"},
        "LALBA": {"nom": "Alpha-Lactalbumin", "chr": "3", "effet": "Protéines lait"},
        "CSN3": {"nom": "Kappa-Casein", "chr": "6", "effet": "Qualité fromagère"},
        "DGAT1": {"nom": "Diacylglycerol Acyltransferase 1", "chr": "14", "effet": "Matière grasse lait"},
        "SCD": {"nom": "Stearoyl-CoA Desaturase", "chr": "22", "effet": "Acides gras insaturés"},
        "TLR4": {"nom": "Toll-like Receptor 4", "chr": "1", "effet": "Résistance infections"},
        "MHC": {"nom": "Major Histocompatibility Complex", "chr": "20", "effet": "Immunité"},
        "PRNP": {"nom": "Prion Protein", "chr": "13", "effet": "Résistance tremblante"},
        "CAST": {"nom": "Calpastatin", "chr": "7", "effet": "Tendreté viande"},
        "CAPN1": {"nom": "Calpain 1", "chr": "16", "effet": "Tendreté viande"},
        "FABP4": {"nom": "Fatty Acid Binding Protein 4", "chr": "8", "effet": "Marbling (gras intramusculaire)"}
    }
    
    ETATS_PHYSIO = [
        "Jeune", "Gestation début", "Gestation fin",
        "Lactation début", "Lactation milieu", "Lactation fin",
        "Tarie", "Engraissement"
    ]

# Seuils de qualité image
QUALITY_BLUR_THRESHOLD = 100.0
QUALITY_CONTRAST_THRESHOLD = 30

# -----------------------------------------------------------------------------
# BASE DE DONNÉES
# -----------------------------------------------------------------------------
@st.cache_resource
def get_database():
    return Database()

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("ovin_streamlit.db", check_same_thread=False)
        self.init_database()
    
    def init_database(self):
        cursor = self.conn.cursor()
        
        # Tables existantes (inchangées)
        tables = [
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT,
                nom_laboratoire TEXT DEFAULT 'GenApAgiE', date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS eleveurs (
                id INTEGER PRIMARY KEY, user_id INTEGER, nom TEXT, region TEXT,
                telephone TEXT, email TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS elevages (
                id INTEGER PRIMARY KEY, eleveur_id INTEGER, nom TEXT,
                localisation TEXT, superficie REAL
            )""",
            """CREATE TABLE IF NOT EXISTS brebis (
                id INTEGER PRIMARY KEY, elevage_id INTEGER, numero_id TEXT UNIQUE,
                nom TEXT, race TEXT, date_naissance TEXT, etat_physio TEXT,
                photo_profil TEXT, photo_mamelle TEXT, sequence_fasta TEXT,
                variants_snps TEXT, profil_genetique TEXT, poids_vif REAL
            )""",
            """CREATE TABLE IF NOT EXISTS mesures_morpho (
                id INTEGER PRIMARY KEY, brebis_id INTEGER, date_mesure TIMESTAMP,
                longueur_corps REAL, hauteur_garrot REAL, tour_poitrine REAL,
                circonference_canon REAL, largeur_bassin REAL, score_global REAL
            )""",
            """CREATE TABLE IF NOT EXISTS mesures_mamelles (
                id INTEGER PRIMARY KEY, brebis_id INTEGER, date_mesure TIMESTAMP,
                longueur_trayon REAL, diametre_trayon REAL, symetrie TEXT,
                attache TEXT, forme TEXT, score_total REAL
            )""",
            """CREATE TABLE IF NOT EXISTS composition_corporelle (
                id INTEGER PRIMARY KEY, brebis_id INTEGER, date_estimation TIMESTAMP,
                poids_vif REAL, poids_carcasse REAL, rendement_carcasse REAL,
                poids_viande REAL, pct_viande REAL, poids_graisse REAL,
                pct_graisse REAL, poids_os REAL, pct_os REAL,
                gigot_poids REAL, epaule_poids REAL, cotelette_poids REAL
            )""",
            """CREATE TABLE IF NOT EXISTS analyses_genomiques (
                id INTEGER PRIMARY KEY, brebis_id INTEGER, date_analyse TIMESTAMP,
                gene_cible TEXT, sequence_query TEXT, blast_hits TEXT,
                identite_pct REAL, e_value REAL
            )""",
            """CREATE TABLE IF NOT EXISTS productions (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date DATE,
                quantite REAL,
                ph REAL,
                mg REAL,
                proteine REAL,
                ag_satures REAL,
                densite REAL,
                extrait_sec REAL,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS genotypes (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                snp_name TEXT,
                genotype TEXT,
                chromosome TEXT,
                position INTEGER,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS phenotypes (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                trait TEXT,
                valeur REAL,
                date_mesure DATE,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date DATE,
                maladie TEXT,
                symptomes TEXT,
                traitement TEXT,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS aliments (
                id INTEGER PRIMARY KEY,
                nom TEXT UNIQUE,
                type TEXT,
                uem REAL,
                pdin REAL,
                ms REAL,
                prix_kg REAL
            )""",
            """CREATE TABLE IF NOT EXISTS rations (
                id INTEGER PRIMARY KEY,
                nom TEXT,
                etat_physio TEXT,
                description TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS ration_composition (
                id INTEGER PRIMARY KEY,
                ration_id INTEGER,
                aliment_id INTEGER,
                quantite_kg REAL,
                FOREIGN KEY (ration_id) REFERENCES rations(id),
                FOREIGN KEY (aliment_id) REFERENCES aliments(id)
            )""",
            """CREATE TABLE IF NOT EXISTS vaccinations (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_vaccin DATE,
                vaccin TEXT,
                rappel DATE,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS soins (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_soin DATE,
                type TEXT,
                diagnostic TEXT,
                traitement TEXT,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS chaleurs (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_debut DATE,
                date_fin DATE,
                methode_synchro TEXT,
                observation TEXT,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS saillies (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_saillie DATE,
                male_id TEXT,
                methode TEXT,
                resultat TEXT,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )""",
            """CREATE TABLE IF NOT EXISTS mises_bas (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_mise_bas DATE,
                nb_agneaux INTEGER,
                poids_portee REAL,
                remarques TEXT,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )"""
        ]
        
        # Nouvelles tables pour IoT et validation
        tables.append("""
            CREATE TABLE IF NOT EXISTS capteurs (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date TIMESTAMP,
                temperature REAL,
                activite REAL,
                rythme_cardiaque REAL,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )
        """)
        tables.append("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                brebis_id INTEGER,
                date_prediction TIMESTAMP,
                modele TEXT,
                variable TEXT,
                valeur_predite REAL,
                valeur_reelle REAL,
                FOREIGN KEY (brebis_id) REFERENCES brebis(id)
            )
        """)
        
        for table in tables:
            cursor.execute(table)
        
        # Aliments initiaux
        aliments_init = [
            ("Orge", "Concentré", 1.1, 80, 86, 25),
            ("Maïs", "Concentré", 1.3, 70, 86, 30),
            ("Son de blé", "Concentré", 0.9, 120, 87, 18),
            ("Tourteau de soja", "Concentré", 1.2, 400, 88, 45),
            ("Foin de luzerne", "Fourrage", 0.6, 120, 85, 15),
            ("Foin d'avoine", "Fourrage", 0.5, 70, 85, 12),
            ("Paille", "Fourrage", 0.3, 20, 88, 5),
            ("CMV", "Minéral", 0, 0, 100, 80)
        ]
        for alim in aliments_init:
            try:
                cursor.execute("INSERT OR IGNORE INTO aliments (nom, type, uem, pdin, ms, prix_kg) VALUES (?, ?, ?, ?, ?, ?)", alim)
            except:
                pass
        
        self.conn.commit()
    
    def execute(self, query: str, params: tuple = ()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor
    
    def fetchall(self, query: str, params: tuple = ()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def fetchone(self, query: str, params: tuple = ()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()

# -----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES (sauvegarde, filtrage, détection)
# -----------------------------------------------------------------------------
def save_uploaded_photo(uploaded_file):
    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(PHOTO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename
    return None

def filtrer_par_eleveur(query_base: str, params: list, join_eleveur: bool = True) -> tuple:
    if st.session_state.eleveur_id is not None:
        if join_eleveur:
            query_base += " AND el.id=?"
        else:
            query_base += " AND eleveur_id=?"
        params.append(st.session_state.eleveur_id)
    return query_base, tuple(params)

# -----------------------------------------------------------------------------
# FONCTIONS QUALITÉ IMAGE ET DÉTECTION
# -----------------------------------------------------------------------------
def is_image_blurry(image_gray, threshold=QUALITY_BLUR_THRESHOLD):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def is_image_low_contrast(image_gray, threshold=QUALITY_CONTRAST_THRESHOLD):
    std = np.std(image_gray)
    return std < threshold, std

def detect_keypoints_mediapipe(image_rgb):
    landmarker = get_pose_landmarker()
    if landmarker is None:
        st.warning("Modèle de détection de pose indisponible (téléchargement impossible). Vérifiez la connexion internet.")
        return None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    results = landmarker.detect(mp_image)
    if not results.pose_landmarks:
        return None
    h, w, _ = image_rgb.shape
    landmarks = results.pose_landmarks[0]
    pts = {}
    # Garrot : entre les épaules (indices 11 et 12 pour humain)
    x = (landmarks[11].x + landmarks[12].x) / 2 * w
    y = (landmarks[11].y + landmarks[12].y) / 2 * h
    pts['garrot'] = (int(x), int(y))
    # Épaule : épaule gauche
    pts['epaule'] = (int(landmarks[11].x * w), int(landmarks[11].y * h))
    # Fesse : hanche gauche (indice 23)
    pts['fesse'] = (int(landmarks[23].x * w), int(landmarks[23].y * h))
    return pts

def detect_keypoints_hybrid(image_rgb, custom_model_path=None):
    """
    Détection de points clés : utilise un modèle custom TensorFlow si présent,
    sinon MediaPipe.
    """
    if custom_model_path and os.path.exists(custom_model_path):
        try:
            model = tf.keras.models.load_model(custom_model_path)
            img_resized = cv2.resize(image_rgb, (256, 256))
            img_input = np.expand_dims(img_resized / 255.0, axis=0)
            pred = model.predict(img_input, verbose=0)[0]
            h, w, _ = image_rgb.shape
            # Les coordonnées prédites sont normalisées (0-1) pour 6 valeurs : garrot(x,y), epaule(x,y), fesse(x,y)
            pts = {
                'garrot': (int(pred[0]*w), int(pred[1]*h)),
                'epaule': (int(pred[2]*w), int(pred[3]*h)),
                'fesse': (int(pred[4]*w), int(pred[5]*h))
            }
            return pts
        except Exception as e:
            st.warning(f"Erreur chargement modèle custom: {e}. Utilisation de MediaPipe.")
    # Fallback MediaPipe
    return detect_keypoints_mediapipe(image_rgb)

def analyze_mamelle(image):
    """
    Analyse une image de mamelle pour détecter rougeur et asymétrie.
    Retourne un rapport.
    """
    h, w = image.shape[:2]
    # Détection de rouge en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = np.sum(red_mask > 0)
    total_pixels = h * w
    red_ratio = red_pixels / total_pixels

    # Asymétrie : diviser en deux verticalement, comparer luminosité
    left_half = image[:, :w//2]
    right_half = image[:, w//2:]
    left_mean = np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
    right_mean = np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))
    asym_ratio = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2 + 1e-6)

    alerts = []
    score = 0
    if red_ratio > 0.03:
        alerts.append("⚠️ Rougeur détectée (possible inflammation)")
        score += 2
    if asym_ratio > 0.2:
        alerts.append("⚠️ Asymétrie importante (possible œdème)")
        score += 2
    if score >= 3:
        level = "Élevé"
    elif score >= 1:
        level = "Modéré"
    else:
        level = "Faible"
    return {
        'red_ratio': round(red_ratio, 3),
        'asym_ratio': round(asym_ratio, 3),
        'score': score,
        'level': level,
        'alerts': alerts
    }

def extract_frames_from_video(video_bytes, n_frames=5):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        tmpfile.flush()
        cap = cv2.VideoCapture(tmpfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        step = max(1, total_frames // n_frames)
        frames = []
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= n_frames:
                break
        cap.release()
        os.unlink(tmpfile.name)
        return frames

def filter_best_images(frames):
    scored = []
    for idx, img in enumerate(frames):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurry, blur_val = is_image_blurry(gray)
        low_contrast, cont_val = is_image_low_contrast(gray)
        # Score : contraste / flou (évite division par zéro)
        score = cont_val / (blur_val + 1e-6)
        scored.append((idx, score, img, blurry, low_contrast))
    scored.sort(key=lambda x: x[1], reverse=True)
    best = [img for _, _, img, blurry, low_contrast in scored if not blurry and not low_contrast]
    if not best:
        best.append(scored[0][2])
    return best

def get_weather_data(api_key, lat, lon):
    if not api_key:
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description']
            }
    except:
        pass
    return None

# Détection étalons
def detecter_baton(image, seuil_canny1=50, seuil_canny2=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, seuil_canny1, seuil_canny2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        max_len = 0
        best_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)
        return best_line, max_len
    return None, 0

def detecter_feuille(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            long_cote = max(w, h)
            return approx, long_cote
    return None, 0

def detecter_piece(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        max_radius = 0
        best_circle = None
        for (x, y, r) in circles:
            if r > max_radius:
                max_radius = r
                best_circle = (x, y, r)
        return best_circle, 2 * max_radius
    return None, 0

# -----------------------------------------------------------------------------
# CLASSES MÉTIER
# -----------------------------------------------------------------------------
class OvinScience:
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def calcul_score_morpho(longueur: float, hauteur: float, poitrine: float, 
                          canon: float, bassin: float) -> float:
        try:
            indice_format = (longueur / hauteur) * 100 if hauteur > 0 else 0
            indice_corpulence = (poitrine / hauteur) * 100 if hauteur > 0 else 0
            score = 40
            if 100 <= indice_format <= 120: score += 20
            if 115 <= indice_corpulence <= 135: score += 20
            if canon > 7.0: score += 10
            if bassin > 18: score += 10
            return min(100, round(score, 2))
        except:
            return 0
    
    @staticmethod
    def calcul_score_mamelle(long_trayon: float, diametre: float,
                           symetrie: str, attache: str, forme: str) -> float:
        score = 5.0
        if 4 <= long_trayon <= 6: score += 1.5
        if 2 <= diametre <= 3: score += 1.5
        if symetrie == "Symétrique": score += 0.5
        if attache == "Solide": score += 0.5
        if forme == "Globuleuse": score += 0.5
        if attache != "Pendante": score += 0.5
        return min(10, round(score, 2))
    
    @staticmethod
    def estimer_composition(poids_vif: float, race: str, condition_corporelle: float) -> Dict:
        try:
            rendement = 0.48 if race == "Ouled Djellal" else 0.45 if race == "Sidahou" else 0.46
            rendement += (condition_corporelle - 3) * 0.01
            poids_carcasse = poids_vif * rendement
            if condition_corporelle >= 4:
                pct_viande, pct_graisse, pct_os = 0.55, 0.28, 0.17
            elif condition_corporelle <= 2:
                pct_viande, pct_graisse, pct_os = 0.62, 0.18, 0.20
            else:
                pct_viande, pct_graisse, pct_os = 0.58, 0.23, 0.19
            if race == "Ouled Djellal":
                pct_viande += 0.02
                pct_graisse -= 0.01
            return {
                "poids_vif": poids_vif,
                "poids_carcasse": round(poids_carcasse, 2),
                "rendement": round(rendement * 100, 1),
                "viande": {"kg": round(poids_carcasse * pct_viande, 2), "pct": round(pct_viande * 100, 1)},
                "graisse": {"kg": round(poids_carcasse * pct_graisse, 2), "pct": round(pct_graisse * 100, 1)},
                "os": {"kg": round(poids_carcasse * pct_os, 2), "pct": round(pct_os * 100, 1)},
                "decoupes": {
                    "gigot": round(poids_carcasse * 0.22, 2),
                    "epaule": round(poids_carcasse * 0.17, 2),
                    "cotelette": round(poids_carcasse * 0.14, 2),
                    "poitrine": round(poids_carcasse * 0.12, 2)
                },
                "qualite": {
                    "conformation": min(15, max(1, 8 + int((condition_corporelle - 3) * 1.5) + (2 if race == "Ouled Djellal" else 0))),
                    "gras": int(condition_corporelle)
                }
            }
        except Exception as e:
            return {"erreur": str(e)}
    
    @staticmethod
    def besoins_nutritionnels(poids: float, etat: str, lactation: float = 0) -> Dict:
        besoins = {
            "maintenance": {"uem": 0.5, "pdin": 45, "ms": 1.0},
            "gestation": {"uem": 0.7, "pdin": 70, "ms": 1.2},
            "lactation": {"uem": 1.2, "pdin": 120, "ms": 2.5},
            "tarie": {"uem": 0.55, "pdin": 50, "ms": 1.1},
            "engraissement": {"uem": 0.8, "pdin": 60, "ms": 1.5}
        }
        base = besoins.get("maintenance")
        for key in besoins:
            if key in etat.lower():
                base = besoins[key]
                break
        if lactation > 0:
            base["uem"] += lactation * 0.4
            base["pdin"] += lactation * 8
        return {k: round(v, 2) for k, v in base.items()}

class MachineLearning:
    @staticmethod
    def predire_lait(score_mam: float, score_morpho: float, race: str, age: int) -> Dict:
        base = 0.5
        if score_mam >= 8: base += 1.5
        elif score_mam >= 6: base += 0.8
        if score_morpho >= 80: base += 0.3
        if race == "Lacaune": base *= 1.3
        if 3 <= age <= 6: base *= 1.2
        return {
            "litres_jour": round(base, 2),
            "litres_lactation": round(base * 180, 2),
            "niveau": "Élite" if base > 1.5 else "Bon" if base > 1.0 else "Standard"
        }

class NCBIApi:
    def __init__(self):
        self.base_url = Config.NCBI_EUTILS_BASE
    
    def search_gene(self, gene_name: str, organism: str = "Ovis aries") -> List[Dict]:
        try:
            url = f"{self.base_url}/esearch.fcgi"
            params = {
                "db": "gene",
                "term": f"{gene_name}[Gene] AND {organism}[Organism]",
                "retmode": "json",
                "retmax": 5
            }
            with st.spinner(f"Recherche {gene_name} dans NCBI..."):
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
            gene_ids = data.get("esearchresult", {}).get("idlist", [])
            if gene_ids:
                return self.fetch_gene_details(gene_ids)
            return []
        except Exception as e:
            st.error(f"Erreur API NCBI: {e}")
            return []
    
    def fetch_gene_details(self, gene_ids: List[str]) -> List[Dict]:
        try:
            url = f"{self.base_url}/esummary.fcgi"
            params = {"db": "gene", "id": ",".join(gene_ids), "retmode": "json"}
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            results = []
            for gid in gene_ids:
                summary = data.get("result", {}).get(gid, {})
                results.append({
                    "gene_id": gid,
                    "name": summary.get("name", "N/A"),
                    "description": summary.get("description", "N/A"),
                    "chromosome": summary.get("chromosome", "N/A"),
                    "map_location": summary.get("maplocation", "N/A")
                })
            return results
        except Exception as e:
            st.error(f"Erreur détails gènes: {e}")
            return []
    
    def fetch_fasta(self, accession: str) -> Optional[str]:
        try:
            url = f"{self.base_url}/efetch.fcgi"
            params = {"db": "nucleotide", "id": accession, "rettype": "fasta", "retmode": "text"}
            response = requests.get(url, params=params, timeout=30)
            return response.text if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Erreur FASTA: {e}")
            return None

class GenomicAnalyzer:
    def __init__(self):
        self.ncbi = NCBIApi()
    
    def analyze_race_profile(self, race: str) -> Dict:
        genes_race = Config.RACES.get(race, {}).get("genes", [])
        results = {
            "race": race,
            "genes": [],
            "score_reproduction": 0,
            "score_croissance": 0,
            "score_lait": 0,
            "recommandations": []
        }
        for gene in genes_race:
            info = Config.GENES_ECONOMIQUES.get(gene, {})
            results["genes"].append({
                "symbole": gene,
                "nom": info.get("nom", ""),
                "effet": info.get("effet", ""),
                "chromosome": info.get("chr", "")
            })
            if gene in ["BMP15", "GDF9", "BMPR1B"]:
                results["score_reproduction"] += 33
            if gene in ["MSTN", "IGF2", "GH"]:
                results["score_croissance"] += 33
            if gene in ["LALBA", "CSN3", "DGAT1"]:
                results["score_lait"] += 33
        results["score_reproduction"] = min(100, results["score_reproduction"])
        results["score_croissance"] = min(100, results["score_croissance"])
        results["score_lait"] = min(100, results["score_lait"])
        if results["score_reproduction"] > 70:
            results["recommandations"].append("✅ Excellente valeur reproductive")
        if results["score_croissance"] > 70:
            results["recommandations"].append("✅ Excellente conformation viande")
        if results["score_lait"] > 70:
            results["recommandations"].append("✅ Excellent potentiel laitier")
        return results

# -----------------------------------------------------------------------------
# FONCTIONS ML
# -----------------------------------------------------------------------------
def train_lait_model():
    query = """
        SELECT p.quantite, b.race, b.date_naissance, 
               AVG(m.score_global) as score_morpho,
               AVG(m2.score_total) as score_mamelle,
               COUNT(DISTINCT p.id) as nb_mesures
        FROM productions p
        JOIN brebis b ON p.brebis_id = b.id
        LEFT JOIN mesures_morpho m ON b.id = m.brebis_id
        LEFT JOIN mesures_mamelles m2 ON b.id = m2.brebis_id
        GROUP BY b.id
        HAVING nb_mesures > 0
    """
    df = pd.read_sql_query(query, db.conn)
    if len(df) < 20:
        return None
    df['age'] = (datetime.now() - pd.to_datetime(df['date_naissance'])).dt.days / 365
    df = pd.get_dummies(df, columns=['race'], prefix='race')
    feature_cols = [c for c in df.columns if c not in ['quantite', 'date_naissance', 'nb_mesures']]
    X = df[feature_cols].fillna(0)
    y = df['quantite']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    joblib.dump(model, os.path.join(MODEL_DIR, 'lait_model.pkl'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'lait_features.pkl'))
    return model, score

def predict_lait_ml(brebis_id):
    model_path = os.path.join(MODEL_DIR, 'lait_model.pkl')
    features_path = os.path.join(MODEL_DIR, 'lait_features.pkl')
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    query = """
        SELECT b.race, b.date_naissance,
               AVG(m.score_global) as score_morpho,
               AVG(m2.score_total) as score_mamelle
        FROM brebis b
        LEFT JOIN mesures_morpho m ON b.id = m.brebis_id
        LEFT JOIN mesures_mamelles m2 ON b.id = m2.brebis_id
        WHERE b.id = ?
        GROUP BY b.id
    """
    row = db.fetchone(query, (brebis_id,))
    if not row:
        return None
    race, date_naiss, score_morpho, score_mamelle = row
    age = (datetime.now() - datetime.strptime(date_naiss, "%Y-%m-%d")).days / 365 if date_naiss else 0
    data = {'score_morpho': score_morpho or 0, 'score_mamelle': score_mamelle or 0, 'age': age}
    for col in feature_cols:
        if col.startswith('race_'):
            data[col] = 1 if col == f"race_{race}" else 0
        elif col not in data:
            data[col] = 0
    X = pd.DataFrame([data])[feature_cols].fillna(0)
    pred = model.predict(X)[0]
    return pred

def cluster_brebis(df, n_clusters=3):
    features = ['prod_moy (L/j)', 'score_morpho', 'poids', 'viande_estimee (kg)']
    avail = [f for f in features if f in df.columns]
    if len(avail) < 2:
        return None
    X = df[avail].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans.cluster_centers_, avail

def detect_anomalies(df, contamination=0.1):
    features = ['prod_moy (L/j)', 'score_morpho', 'poids', 'viande_estimee (kg)']
    avail = [f for f in features if f in df.columns]
    if len(avail) < 2:
        return None
    X = df[avail].fillna(0)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    return preds

# -----------------------------------------------------------------------------
# PAGES DE L'APPLICATION
# -----------------------------------------------------------------------------

# ---- PAGE LOGIN ----
def page_login():
    st.markdown('<p class="main-header">🐑 Ovin Manager Pro</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Laboratoire {Config.LABORATOIRE} - Système Expert de Génétique Ovine</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Connexion", "Inscription"])
        with tab1:
            username = st.text_input("Nom d'utilisateur", key="login_user")
            password = st.text_input("Mot de passe", type="password", key="login_pass")
            if st.button("Se connecter", use_container_width=True):
                user = db.fetchone(
                    "SELECT id FROM users WHERE username=? AND password_hash=?",
                    (username, OvinScience.hash_password(password))
                )
                if user:
                    st.session_state.user_id = user[0]
                    st.session_state.current_page = "dashboard"
                    st.rerun()
                else:
                    st.error("Identifiants incorrects")
        with tab2:
            new_user = st.text_input("Nouvel utilisateur", key="new_user")
            new_pass = st.text_input("Mot de passe", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirmer mot de passe", type="password")
            if st.button("Créer compte", use_container_width=True):
                if new_pass != confirm_pass:
                    st.error("Les mots de passe ne correspondent pas")
                elif not new_user or not new_pass:
                    st.error("Remplissez tous les champs")
                else:
                    try:
                        db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                                  (new_user, OvinScience.hash_password(new_pass)))
                        st.success("Compte créé ! Connectez-vous")
                    except:
                        st.error("Nom d'utilisateur déjà pris")

# ---- PAGE DASHBOARD ----
def page_dashboard():
    st.title(f"📊 Tableau de Bord - {Config.LABORATOIRE}")
    dash_stats = db.fetchone("""
        SELECT 
            (SELECT COUNT(*) FROM eleveurs WHERE user_id=?),
            (SELECT COUNT(*) FROM brebis b JOIN elevages e ON b.elevage_id = e.id 
             JOIN eleveurs el ON e.eleveur_id = el.id WHERE el.user_id=?),
            (SELECT COUNT(*) FROM composition_corporelle cc 
             JOIN brebis b ON cc.brebis_id = b.id JOIN elevages e ON b.elevage_id = e.id
             JOIN eleveurs el ON e.eleveur_id = el.id WHERE el.user_id=?)
    """, (st.session_state.user_id, st.session_state.user_id, st.session_state.user_id))
    cols = st.columns(4)
    metrics = [
        ("👨‍🌾 Éleveurs", dash_stats[0], Config.VERT),
        ("🐑 Brebis", dash_stats[1], Config.BLEU),
        ("🧬 Analyses", dash_stats[2], Config.CYAN),
        ("📈 Données", dash_stats[0] + dash_stats[1] + dash_stats[2], Config.ORANGE)
    ]
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background-color: {color}20; border-radius: 10px; padding: 20px; text-align: center; border-left: 5px solid {color}">
                <h3 style="color: {color}; margin: 0;">{value}</h3>
                <p style="margin: 0; color: #666;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    st.divider()
    st.subheader("🚀 Modules Génomiques & Analytiques")
    modules = [
        ("🧬 Analyse NCBI/GenBank", "Recherche gènes, SNPs, BLAST", "genomique", Config.CYAN),
        ("🥩 Composition Corporelle", "Estimation viande/graisse/os", "composition", Config.ORANGE),
        ("📸 Photogrammétrie auto", "Capture vidéo, IA, filtrage qualité", "analyse_auto", Config.VERT),
        ("🥛 Prédiction Lait", "ML potentiel laitier", "prediction", Config.VIOLET),
        ("🌾 Nutrition", "Formulation rations", "nutrition_avancee", Config.BLEU),
        ("🧠 IA & Data Mining", "Analyses avancées, clustering, anomalies", "ia", Config.ROUGE),
        ("📡 IoT & Capteurs", "Import données capteurs", "iot", Config.CYAN),
        ("📊 Validation", "Comparaison prédictions vs réel", "validation", Config.ORANGE),
    ]
    cols = st.columns(3)
    for i, (title, desc, page, color) in enumerate(modules):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div style="background-color: white; border-radius: 10px; padding: 20px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
                            border-top: 4px solid {color};">
                    <h4 style="color: {color}; margin-top: 0;">{title}</h4>
                    <p style="color: #666; font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Ouvrir →", key=f"btn_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()

# ---- PAGE GÉNOMIQUE NCBI ----
def page_genomique():
    st.title("🧬 Analyse Génomique - NCBI/GenBank")
    tab1, tab2, tab3 = st.tabs(["🔍 Recherche Gène", "🏆 Profil Race", "🧪 SNPs/QTN"])
    with tab1:
        st.subheader("Recherche dans NCBI Gene")
        col1, col2 = st.columns([2, 1])
        with col1:
            gene_search = st.text_input("Nom du gène", "BMP15")
        with col2:
            organism_label = st.selectbox("Organisme", ["Ovis aries (Mouton)", "Capra hircus (Chèvre)", "Bos taurus (Bovin)"])
            organism = organism_label.split(" (")[0]
        if st.button("🔍 Rechercher dans NCBI", use_container_width=True):
            results = genomic_analyzer.ncbi.search_gene(gene_search, organism)
            if results:
                for gene in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="gene-card">
                            <h4>🧬 {gene['name']} (ID: {gene['gene_id']})</h4>
                            <p><strong>Description:</strong> {gene['description']}</p>
                            <p><strong>Chromosome:</strong> {gene['chromosome']} | <strong>Position:</strong> {gene['map_location']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        local_info = Config.GENES_ECONOMIQUES.get(gene_search.upper())
                        if local_info:
                            st.info(f"**Effet économique:** {local_info['effet']}")
            else:
                local = Config.GENES_ECONOMIQUES.get(gene_search.upper())
                if local:
                    st.success("Informations depuis la base locale GenApAgiE")
                    st.json(local)
                else:
                    st.warning("Gène non trouvé. Essayez: BMP15, MSTN, DGAT1, CAST, CAPN1...")
    with tab2:
        st.subheader("Profil Génétique par Race")
        race_selected = st.selectbox("Sélectionner une race", list(Config.RACES.keys()))
        if st.button("🧬 Analyser le profil génétique"):
            analysis = genomic_analyzer.analyze_race_profile(race_selected)
            fig = go.Figure(data=go.Scatterpolar(
                r=[analysis['score_reproduction'], analysis['score_croissance'], 
                   analysis['score_lait'], analysis['score_reproduction']],
                theta=['Reproduction', 'Croissance/Viande', 'Lait', 'Reproduction'],
                fill='toself',
                name=race_selected
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title=f"Profil Génétique: {race_selected}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Gènes Majeurs")
            for gene in analysis['genes']:
                with st.expander(f"🧬 {gene['symbole']} - {gene['nom'][:40]}..."):
                    st.write(f"**Effet:** {gene['effet']}")
                    st.write(f"**Chromosome:** {gene['chromosome']}")
            if analysis['recommandations']:
                st.success("### ✅ Recommandations")
                for rec in analysis['recommandations']:
                    st.write(rec)
    with tab3:
        st.subheader("Base de données SNPs et QTN économiques")
        categorie = st.selectbox("Filtrer par catégorie", 
                                ["Tous", "Reproduction", "Croissance/Viande", "Lait", "Résistance", "Qualité viande"])
        genes_filtres = []
        for sym, info in Config.GENES_ECONOMIQUES.items():
            if categorie == "Tous":
                genes_filtres.append((sym, info))
            elif categorie == "Reproduction" and any(x in sym for x in ["BMP", "GDF"]):
                genes_filtres.append((sym, info))
            elif categorie == "Croissance/Viande" and any(x in sym for x in ["MSTN", "IGF", "GH"]):
                genes_filtres.append((sym, info))
            elif categorie == "Lait" and any(x in sym for x in ["LALBA", "CSN", "DGAT", "SCD"]):
                genes_filtres.append((sym, info))
            elif categorie == "Résistance" and any(x in sym for x in ["TLR", "MHC", "PRNP"]):
                genes_filtres.append((sym, info))
            elif categorie == "Qualité viande" and any(x in sym for x in ["CAST", "CAPN", "FABP"]):
                genes_filtres.append((sym, info))
        df_genes = pd.DataFrame([
            {
                "Symbole": sym,
                "Nom": info["nom"][:50] + "...",
                "Chr": info["chr"],
                "Effet": info["effet"][:60] + "...",
                "Type": "QTN" if sym in ["BMP15", "MSTN", "DGAT1", "BMPR1B"] else "SNP"
            }
            for sym, info in genes_filtres
        ])
        st.dataframe(df_genes, use_container_width=True, hide_index=True)
        gene_detail = st.selectbox("Voir détails", [sym for sym, _ in genes_filtres])
        if gene_detail:
            info = Config.GENES_ECONOMIQUES[gene_detail]
            st.json(info)

# ---- PAGE COMPOSITION ----
def page_composition():
    st.title("🥩 Composition Corporelle Estimée")
    st.markdown("Estimation détaillée de la répartition viande/graisse/os basée sur les équations zootechniques")
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom, b.race, e.nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis_list = db.fetchall(query_brebis, params)
    brebis_options = {f"{b[0]} - {b[1]} {b[2]} ({b[4]})": b[0] for b in brebis_list}
    brebis_options["Saisie manuelle (animal non enregistré)"] = None
    mode = st.radio("Mode de saisie", ["Sélectionner une brebis existante", "Saisie manuelle"])
    if mode == "Sélectionner une brebis existante":
        selected = st.selectbox("Choisir une brebis", list(brebis_options.keys()))
        brebis_id = brebis_options[selected]
        if brebis_id is not None:
            info = db.fetchone("SELECT poids_vif, race, etat_physio FROM brebis WHERE id=?", (brebis_id,))
            if info:
                poids_def = info[0] if info[0] is not None else 45.0
                race_def = info[1] if info[1] else "Autre"
            else:
                poids_def = 45.0
                race_def = "Autre"
        else:
            poids_def = 45.0
            race_def = "Autre"
    else:
        brebis_id = None
        poids_def = 45.0
        race_def = "Autre"
    col1, col2, col3 = st.columns(3)
    with col1:
        poids_vif = st.number_input("Poids vif (kg)", min_value=10.0, max_value=150.0, value=poids_def, step=0.5)
    with col2:
        race = st.selectbox("Race", list(Config.RACES.keys()), index=list(Config.RACES.keys()).index(race_def) if race_def in Config.RACES else 0)
    with col3:
        cc = st.slider("Condition Corporelle (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
    if st.button("🧮 Calculer la composition", use_container_width=True):
        comp = OvinScience.estimer_composition(poids_vif, race, cc)
        if "erreur" in comp:
            st.error(comp["erreur"])
            return
        st.subheader("📊 Résultats")
        cols = st.columns(4)
        metrics = [
            ("🥩 Viande", comp['viande']['kg'], comp['viande']['pct'], Config.VERT),
            ("🥓 Graisse", comp['graisse']['kg'], comp['graisse']['pct'], Config.ORANGE),
            ("🦴 Os", comp['os']['kg'], comp['os']['pct'], "grey"),
            ("📦 Carcasse", comp['poids_carcasse'], comp['rendement'], Config.BLEU)
        ]
        for col, (label, kg, pct, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div style="background-color: {color}15; border-radius: 10px; padding: 20px; 
                            text-align: center; border-left: 4px solid {color};">
                    <h4 style="color: {color}; margin: 0;">{kg} kg</h4>
                    <p style="margin: 0; font-size: 0.9rem;">{label}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">{pct}%</p>
                </div>
                """, unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=['Viande', 'Graisse', 'Os'],
            values=[comp['viande']['kg'], comp['graisse']['kg'], comp['os']['kg']],
            marker_colors=[Config.VERT, Config.ORANGE, 'grey'],
            hole=0.4
        )])
        fig.update_layout(title="Composition de la carcasse (kg)")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔪 Détails des découpes"):
            decoupes_data = {
                "Découpe": ["Gigot", "Épaule", "Côtelettes", "Poitrine"],
                "Poids (kg)": [comp['decoupes']['gigot'], comp['decoupes']['epaule'],
                              comp['decoupes']['cotelette'], comp['decoupes']['poitrine']],
                "% Carcasse": [22, 17, 14, 12]
            }
            df_decoupes = pd.DataFrame(decoupes_data)
            st.dataframe(df_decoupes, hide_index=True, use_container_width=True)
        if brebis_id is not None:
            if st.button("💾 Enregistrer cette composition dans la base"):
                db.execute("""
                    INSERT INTO composition_corporelle 
                    (brebis_id, date_estimation, poids_vif, poids_carcasse, rendement_carcasse,
                     poids_viande, pct_viande, poids_graisse, pct_graisse, poids_os, pct_os,
                     gigot_poids, epaule_poids, cotelette_poids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    brebis_id, datetime.now().isoformat(),
                    poids_vif, comp['poids_carcasse'], comp['rendement'],
                    comp['viande']['kg'], comp['viande']['pct'],
                    comp['graisse']['kg'], comp['graisse']['pct'],
                    comp['os']['kg'], comp['os']['pct'],
                    comp['decoupes']['gigot'], comp['decoupes']['epaule'], comp['decoupes']['cotelette']
                ))
                st.success("Composition enregistrée pour cette brebis !")
    st.divider()
    st.subheader("🔍 Comparer plusieurs brebis")
    if len(brebis_list) >= 2:
        selected_ids = st.multiselect(
            "Choisir les brebis à comparer",
            options=list(brebis_options.keys()),
            default=list(brebis_options.keys())[:min(2, len(brebis_options))]
        )
        selected_ids = [brebis_options[id_str] for id_str in selected_ids if brebis_options[id_str] is not None]
        if len(selected_ids) >= 2:
            comp_data = []
            for bid in selected_ids:
                row = db.fetchone("""
                    SELECT poids_vif, poids_carcasse, rendement_carcasse,
                           poids_viande, poids_graisse, poids_os, date_estimation
                    FROM composition_corporelle
                    WHERE brebis_id=?
                    ORDER BY date_estimation DESC
                    LIMIT 1
                """, (bid,))
                if row:
                    name = db.fetchone("SELECT numero_id, nom FROM brebis WHERE id=?", (bid,))
                    label = f"{name[0]} {name[1]}" if name else f"Brebis {bid}"
                    comp_data.append({
                        "id": bid,
                        "nom": label,
                        "poids_vif": row[0],
                        "poids_carcasse": row[1],
                        "rendement": row[2],
                        "viande": row[3],
                        "graisse": row[4],
                        "os": row[5],
                        "date": row[6]
                    })
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                fig_comp = go.Figure()
                for animal in comp_data:
                    fig_comp.add_trace(go.Bar(
                        name=animal['nom'],
                        x=['Viande', 'Graisse', 'Os'],
                        y=[animal['viande'], animal['graisse'], animal['os']],
                        text=[f"{animal['viande']} kg", f"{animal['graisse']} kg", f"{animal['os']} kg"],
                        textposition='auto'
                    ))
                fig_comp.update_layout(
                    title="Comparaison des compositions (kg)",
                    barmode='group',
                    yaxis_title="Poids (kg)"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(df_comp[['nom', 'poids_vif', 'poids_carcasse', 'rendement', 'viande', 'graisse', 'os']].round(2),
                           use_container_width=True, hide_index=True)
            else:
                st.warning("Aucune composition enregistrée pour ces brebis. Calculez d'abord une composition et enregistrez-la.")
    else:
        st.info("Ajoutez au moins deux brebis et enregistrez leurs compositions pour activer la comparaison.")

# ---- PAGE PRÉDICTION ----
def page_prediction():
    st.title("🔮 Prédiction par Machine Learning")
    st.subheader("Potentiel laitier estimé")
    col1, col2 = st.columns(2)
    with col1:
        score_mam = st.slider("Score mamelles", 1.0, 10.0, 7.0, 0.5)
        score_morpho = st.slider("Score morphologique", 0, 100, 75)
    with col2:
        race = st.selectbox("Race", list(Config.RACES.keys()))
        age = st.number_input("Âge (années)", 1, 15, 4)
    if st.button("🔮 Prédire production (formule simple)"):
        pred = MachineLearning.predire_lait(score_mam, score_morpho, race, age)
        cols = st.columns(3)
        cols[0].metric("Production/jour", f"{pred['litres_jour']} L")
        cols[1].metric("Production/lactation", f"{pred['litres_lactation']} L")
        cols[2].metric("Niveau", pred['niveau'])
        fig = px.bar(
            x=["Potentiel estimé", "Moyenne race", "Record élite"],
            y=[pred['litres_jour'], 1.2, 2.5],
            color=[pred['niveau'], "Moyenne", "Élite"],
            title="Comparaison production laitière (L/jour)"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Prédiction avancée par modèle ML")
    model_path = os.path.join(MODEL_DIR, 'lait_model.pkl')
    if os.path.exists(model_path):
        st.success("Un modèle ML est disponible.")
        params = [st.session_state.user_id]
        query_brebis = """
            SELECT b.id, b.numero_id, b.nom, e.nom
            FROM brebis b
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
        """
        query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
        brebis_list = db.fetchall(query_brebis, params)
        brebis_dict = {f"{b[0]} - {b[1]} {b[2]} ({b[3]})": b[0] for b in brebis_list}
        if brebis_dict:
            selected = st.selectbox("Choisir une brebis", list(brebis_dict.keys()), key="ml_brebis")
            bid = brebis_dict[selected]
            if st.button("Prédire avec ML"):
                pred = predict_lait_ml(bid)
                if pred is not None:
                    st.metric("Production prédite (L/j)", f"{pred:.2f}")
                else:
                    st.warning("Impossible de faire la prédiction (données manquantes).")
        else:
            st.warning("Aucune brebis disponible.")
    else:
        st.info("Aucun modèle ML entraîné. Vous pouvez en entraîner un si vous avez suffisamment de données de production.")
        if st.button("Entraîner un modèle ML"):
            with st.spinner("Entraînement en cours..."):
                result = train_lait_model()
                if result is None:
                    st.error("Pas assez de données (minimum 20 brebis avec productions).")
                else:
                    model, score = result
                    st.success(f"Modèle entraîné avec un score R² de {score:.2f} sur le test.")

# ---- PAGE PHOTOGRAMMÉTRIE AUTOMATISÉE (avec analyse mamelle) ----
def page_analyse_auto():
    st.title("📸 Photogrammétrie Automatisée (IA)")
    st.markdown("""
    **Cette version automatise la collecte de données massives :**
    - Capture vidéo en rafale (plusieurs images en une prise)
    - Filtrage qualité (flou, contraste)
    - Détection automatique des points clés par IA (MediaPipe ou modèle custom)
    - Validation manuelle possible si nécessaire
    - Enregistrement des métadonnées (qualité, météo)
    - **Nouveau :** Analyse automatique de la mamelle (rougeur, asymétrie)
    """)
    mode = st.radio("Mode de capture", ["Vidéo (rafale)", "Photo unique"], index=0)
    etalon = st.selectbox("Étalon de calibration", list(Config.ETALONS.keys()),
                         format_func=lambda x: Config.ETALONS[x]['nom'])
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude (optionnel)", value=36.0, format="%.4f")
    with col2:
        lon = st.number_input("Longitude (optionnel)", value=2.0, format="%.4f")
    st.subheader("Calibration automatique")
    use_auto_scale = st.checkbox("Détecter l'étalon automatiquement", value=True)
    facteur = None
    if not use_auto_scale:
        facteur = st.number_input("Facteur d'échelle (px/cm)", value=10.0, step=0.1)
    
    # Upload pour photo corps
    uploaded_file = st.file_uploader("Choisir une vidéo ou une photo (corps)", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
    # Upload pour photo mamelle (optionnel)
    uploaded_mamelle = st.file_uploader("Photo de la mamelle (optionnelle)", type=['jpg', 'png', 'jpeg'], key="mamelle")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if uploaded_file.type.startswith('video'):
            with st.spinner("Extraction des images de la vidéo..."):
                frames = extract_frames_from_video(file_bytes, n_frames=10)
            if not frames:
                st.error("Impossible d'extraire des images de la vidéo.")
                return
            st.success(f"{len(frames)} images extraites.")
            with st.spinner("Filtrage des images de qualité..."):
                best_frames = filter_best_images(frames)
            st.success(f"{len(best_frames)} images retenues après filtrage.")
            if not best_frames:
                st.warning("Aucune image de qualité suffisante. Utilisation de la première image.")
                best_frames = [frames[0]]
            selected_img = best_frames[0]
            st.image(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB), caption="Meilleure image extraite", use_column_width=True)
        else:
            img_pil = Image.open(BytesIO(file_bytes))
            selected_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(selected_img, cv2.COLOR_BGR2GRAY)
            blurry, blur_val = is_image_blurry(gray)
            low_contrast, cont_val = is_image_low_contrast(gray)
            if blurry or low_contrast:
                st.warning(f"Image potentiellement floue (var={blur_val:.1f}) ou faible contraste (std={cont_val:.1f}).")
            st.image(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB), caption="Photo chargée", use_column_width=True)
        
        # Détection des points clés (hybride)
        st.subheader("Détection IA des points anatomiques")
        custom_model_path = os.path.join(MODEL_DIR, 'keypoints_model_custom.h5')
        with st.spinner("Analyse par IA..."):
            img_rgb = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
            keypoints = detect_keypoints_hybrid(img_rgb, custom_model_path)
        
        if keypoints is None:
            st.warning("Aucun point détecté par l'IA. Veuillez passer en mode manuel.")
            coord = streamlit_image_coordinates(img_rgb, key="manual_fallback")
            if coord:
                x, y = coord["x"], coord["y"]
                st.write(f"Point cliqué : ({x}, {y})")
        else:
            st.success(f"Points détectés : Garrot {keypoints['garrot']}, Épaule {keypoints['epaule']}, Fesse {keypoints['fesse']}")
            img_with_kp = selected_img.copy()
            for pt in keypoints.values():
                cv2.circle(img_with_kp, pt, 6, (0,255,0), -1)
            st.image(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB), caption="Points détectés", use_column_width=True)
            if st.checkbox("Corriger manuellement les points ?"):
                st.info("Cliquez sur l'image pour repositionner chaque point.")
                coord = streamlit_image_coordinates(img_rgb, key="correction")
                if coord:
                    new_x, new_y = coord["x"], coord["y"]
                    pt_to_replace = st.selectbox("Quel point remplacer ?", list(keypoints.keys()))
                    if st.button("Remplacer ce point"):
                        keypoints[pt_to_replace] = (new_x, new_y)
                        st.rerun()
            
            # Analyse de la mamelle si une photo est fournie
            if uploaded_mamelle is not None:
                st.subheader("🔬 Analyse automatique de la mamelle")
                mamelle_img = Image.open(uploaded_mamelle)
                mamelle_cv = cv2.cvtColor(np.array(mamelle_img), cv2.COLOR_RGB2BGR)
                st.image(mamelle_img, caption="Photo de la mamelle", use_column_width=True)
                with st.spinner("Analyse en cours..."):
                    mamelle_report = analyze_mamelle(mamelle_cv)
                st.write(f"**Niveau d'alerte :** {mamelle_report['level']}")
                st.write(f"**Ratio de rougeur :** {mamelle_report['red_ratio']}")
                st.write(f"**Ratio d'asymétrie :** {mamelle_report['asym_ratio']}")
                if mamelle_report['alerts']:
                    for alert in mamelle_report['alerts']:
                        st.warning(alert)
                else:
                    st.success("✅ Aucune anomalie détectée.")
            
            if st.button("💾 Enregistrer cette image et ses points dans le dataset"):
                if use_auto_scale:
                    if etalon == "baton_1m":
                        line, len_px = detecter_baton(selected_img)
                        if line is not None:
                            facteur = len_px / 100
                    elif etalon == "a4":
                        rect, long_px = detecter_feuille(selected_img)
                        if rect is not None:
                            facteur = long_px / 29.7
                    elif etalon == "piece_100da":
                        circle, diam_px = detecter_piece(selected_img)
                        if circle is not None:
                            facteur = diam_px / 2.95
                    if facteur is None:
                        st.error("Impossible de détecter l'étalon automatiquement. Veuillez saisir manuellement.")
                        facteur = st.number_input("Facteur d'échelle (px/cm)", value=10.0, step=0.1)
                else:
                    facteur = facteur
                if facteur is None:
                    st.warning("Facteur d'échelle non défini.")
                    return
                weather_data = None
                if lat and lon:
                    api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
                    if api_key:
                        weather_data = get_weather_data(api_key, lat, lon)
                        if weather_data:
                            st.info(f"Météo : {weather_data['temp']}°C, {weather_data['humidity']}% humidité")
                img_resized = cv2.resize(selected_img, (256, 256))
                h, w = selected_img.shape[:2]
                points_norm = {k: (v[0]/w, v[1]/h) for k, v in keypoints.items()}
                metadata = {
                    'race': st.session_state.get('race_brebis', 'Inconnue'),
                    'age_mois': st.session_state.get('age_mois', 24),
                    'facteur_echelle': facteur,
                    'qualite': {'blur_var': blur_val if 'blur_val' in locals() else 0,
                               'contrast_std': cont_val if 'cont_val' in locals() else 0},
                    'meteo': weather_data,
                    'date': datetime.now().isoformat(),
                    'mamelle_analysis': mamelle_report if uploaded_mamelle else None
                }
                filename = os.path.join(DATASET_DIR, f"{uuid.uuid4().hex}.npz")
                np.savez(filename, image=img_resized, points_norm=points_norm, metadata=metadata)
                st.success(f"Données enregistrées dans {filename}")
                st.info("Données sauvegardées dans le dataset d'entraînement.")
    else:
        st.info("Veuillez charger une vidéo ou une photo du corps.")
    st.divider()
    with st.expander("📊 Voir les données collectées"):
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.npz')]
        if files:
            st.write(f"{len(files)} échantillons dans le dataset.")
            data_summary = []
            for f in files[-10:]:
                data = np.load(os.path.join(DATASET_DIR, f), allow_pickle=True)
                meta = data['metadata'].item()
                data_summary.append({
                    'Fichier': f,
                    'Race': meta.get('race', '?'),
                    'Âge (mois)': meta.get('age_mois', '?'),
                    'Facteur': meta.get('facteur_echelle', '?'),
                    'Météo': meta.get('meteo', {}).get('temp', '?')
                })
            if data_summary:
                st.dataframe(pd.DataFrame(data_summary), use_container_width=True)
        else:
            st.info("Aucune donnée collectée.")

# ---- PAGE NUTRITION AVANCÉE (inchangée mais complète) ----
def page_nutrition_avancee():
    st.title("🌾 Nutrition avancée et formulation")
    tab1, tab2, tab3 = st.tabs(["📦 Catalogue aliments", "📋 Rations types", "🧮 Calcul ration personnalisée"])
    with tab1:
        st.subheader("Gestion des aliments")
        with st.expander("➕ Ajouter un aliment"):
            with st.form("form_aliment"):
                nom = st.text_input("Nom de l'aliment")
                type_alim = st.selectbox("Type", ["Fourrage", "Concentré", "Minéral", "Autre"])
                uem = st.number_input("UEM (MJ/kg)", min_value=0.0, step=0.1, format="%.2f")
                pdin = st.number_input("PDIN (g/kg)", min_value=0.0, step=1.0)
                ms = st.number_input("Matière sèche (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
                prix = st.number_input("Prix (DA/kg)", min_value=0.0, step=1.0, format="%.2f")
                if st.form_submit_button("Ajouter"):
                    try:
                        db.execute(
                            "INSERT INTO aliments (nom, type, uem, pdin, ms, prix_kg) VALUES (?, ?, ?, ?, ?, ?)",
                            (nom, type_alim, uem, pdin, ms, prix)
                        )
                        st.success("Aliment ajouté")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Cet aliment existe déjà.")
        aliments = db.fetchall("SELECT id, nom, type, uem, pdin, ms, prix_kg FROM aliments")
        if aliments:
            df_alim = pd.DataFrame(aliments, columns=["ID", "Nom", "Type", "UEM", "PDIN", "MS%", "Prix DA/kg"])
            st.dataframe(df_alim, use_container_width=True, hide_index=True)
            with st.expander("💰 Modifier un prix"):
                choix = st.selectbox("Choisir un aliment", [f"{a[0]} - {a[1]}" for a in aliments])
                aid = int(choix.split(" - ")[0])
                nouveau_prix = st.number_input("Nouveau prix (DA/kg)", min_value=0.0, step=1.0)
                if st.button("Mettre à jour"):
                    db.execute("UPDATE aliments SET prix_kg=? WHERE id=?", (nouveau_prix, aid))
                    st.success("Prix mis à jour")
                    st.rerun()
        else:
            st.info("Aucun aliment enregistré. Commencez par en ajouter.")
    with tab2:
        st.subheader("Rations types par état physiologique")
        etat_physio = st.selectbox("État physiologique", Config.ETATS_PHYSIO)
        ration_existante = db.fetchone("SELECT id, nom, description FROM rations WHERE etat_physio=?", (etat_physio,))
        if ration_existante:
            st.success(f"Ration existante : {ration_existante[1]}")
            compo = db.fetchall("""
                SELECT a.nom, rc.quantite_kg, a.prix_kg
                FROM ration_composition rc
                JOIN aliments a ON rc.aliment_id = a.id
                WHERE rc.ration_id=?
            """, (ration_existante[0],))
            if compo:
                df_compo = pd.DataFrame(compo, columns=["Aliment", "Quantité (kg/jour)", "Prix/kg"])
                df_compo["Coût (DA/jour)"] = df_compo["Quantité (kg/jour)"] * df_compo["Prix/kg"]
                st.dataframe(df_compo, use_container_width=True, hide_index=True)
                total_journalier = df_compo["Coût (DA/jour)"].sum()
                st.metric("Coût total journalier", f"{total_journalier:.2f} DA")
            else:
                st.info("Cette ration n'a pas d'aliments associés.")
        else:
            st.info("Aucune ration définie pour cet état.")
        with st.expander("⚙️ Configurer une ration pour cet état"):
            aliments = db.fetchall("SELECT id, nom FROM aliments")
            if not aliments:
                st.warning("Ajoutez d'abord des aliments.")
            else:
                if ration_existante:
                    ration_id = ration_existante[0]
                    st.markdown("**Modifier la ration existante**")
                else:
                    nom_ration = st.text_input("Nom de la ration", value=f"Ration {etat_physio}")
                    desc = st.text_area("Description")
                    if st.button("Créer la ration"):
                        db.execute(
                            "INSERT INTO rations (nom, etat_physio, description) VALUES (?, ?, ?)",
                            (nom_ration, etat_physio, desc)
                        )
                        st.success("Ration créée, vous pouvez maintenant ajouter des aliments.")
                        st.rerun()
                    ration_id = None
                if ration_id:
                    st.subheader("Ajouter un aliment à cette ration")
                    aliment_choix = st.selectbox("Choisir un aliment", [f"{a[0]} - {a[1]}" for a in aliments])
                    aid = int(aliment_choix.split(" - ")[0])
                    quantite = st.number_input("Quantité (kg/jour)", min_value=0.0, step=0.1, format="%.2f")
                    if st.button("Ajouter à la ration"):
                        existing = db.fetchone(
                            "SELECT id FROM ration_composition WHERE ration_id=? AND aliment_id=?",
                            (ration_id, aid)
                        )
                        if existing:
                            db.execute(
                                "UPDATE ration_composition SET quantite_kg=? WHERE id=?",
                                (quantite, existing[0])
                            )
                        else:
                            db.execute(
                                "INSERT INTO ration_composition (ration_id, aliment_id, quantite_kg) VALUES (?, ?, ?)",
                                (ration_id, aid, quantite)
                            )
                        st.success("Aliment ajouté/modifié")
                        st.rerun()
                    with st.expander("🗑️ Supprimer un aliment de la ration"):
                        compo = db.fetchall("""
                            SELECT rc.id, a.nom FROM ration_composition rc
                            JOIN aliments a ON rc.aliment_id = a.id
                            WHERE rc.ration_id=?
                        """, (ration_id,))
                        if compo:
                            choix_suppr = st.selectbox("Aliment à retirer", [f"{c[0]} - {c[1]}" for c in compo])
                            suppr_id = int(choix_suppr.split(" - ")[0])
                            if st.button("Retirer"):
                                db.execute("DELETE FROM ration_composition WHERE id=?", (suppr_id,))
                                st.success("Aliment retiré")
                                st.rerun()
    with tab3:
        st.subheader("Calcul de ration personnalisée")
        params = [st.session_state.user_id]
        query_brebis = """
            SELECT b.id, b.numero_id, b.nom, b.etat_physio, b.poids_vif
            FROM brebis b
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
        """
        query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
        brebis_list = db.fetchall(query_brebis, params)
        brebis_dict = {f"{b[0]} - {b[1]} {b[2]}": b[0] for b in brebis_list}
        if brebis_dict:
            choix = st.selectbox("Choisir une brebis (ou personnaliser)", ["Personnalisé"] + list(brebis_dict.keys()))
            if choix != "Personnalisé":
                bid = brebis_dict[choix]
                infos = db.fetchone("SELECT poids_vif, etat_physio FROM brebis WHERE id=?", (bid,))
                if infos:
                    poids_def = infos[0] or 50.0
                    etat_def = infos[1] or "Tarie"
                else:
                    poids_def = 50.0
                    etat_def = "Tarie"
            else:
                poids_def = 50.0
                etat_def = "Tarie"
            col1, col2 = st.columns(2)
            with col1:
                poids = st.number_input("Poids vif (kg)", min_value=10.0, max_value=150.0, value=poids_def)
            with col2:
                etat = st.selectbox("État physiologique", Config.ETATS_PHYSIO, index=Config.ETATS_PHYSIO.index(etat_def) if etat_def in Config.ETATS_PHYSIO else 0)
            lactation = st.number_input("Production laitière (L/j)", min_value=0.0, value=0.0, step=0.5)
            besoins = OvinScience.besoins_nutritionnels(poids, etat, lactation)
            st.info(f"**Besoins journaliers** : UEM = {besoins['uem']} MJ, PDIN = {besoins['pdin']} g, MS = {besoins['ms']} kg")
            aliments = db.fetchall("SELECT id, nom, type, uem, pdin, ms, prix_kg FROM aliments")
            if not aliments:
                st.warning("Ajoutez d'abord des aliments.")
            else:
                mode_ration = st.radio("Mode de composition", ["Manuel", "Optimisation automatique (coût minimum)"])
                if mode_ration == "Manuel":
                    st.subheader("Composition de la ration")
                    ration_temp = {}
                    for alim in aliments:
                        with st.expander(f"{alim[1]} ({alim[2]}) - {alim[6]} DA/kg"):
                            qte = st.number_input("Quantité (kg MS)", min_value=0.0, step=0.1, key=f"qte_{alim[0]}")
                            if qte > 0:
                                ration_temp[alim[0]] = {
                                    "nom": alim[1],
                                    "qte": qte,
                                    "uem": alim[3],
                                    "pdin": alim[4],
                                    "ms": alim[5],
                                    "prix": alim[6]
                                }
                    if ration_temp and st.button("Calculer la ration"):
                        total_uem = sum(v["qte"] * v["uem"] for v in ration_temp.values())
                        total_pdin = sum(v["qte"] * v["pdin"] for v in ration_temp.values())
                        total_ms = sum(v["qte"] for v in ration_temp.values())
                        total_prix = sum(v["qte"] * v["prix"] for v in ration_temp.values())
                        st.subheader("Résultats")
                        cola, colb, colc = st.columns(3)
                        cola.metric("UEM apportée", f"{total_uem:.2f} MJ", delta=f"{total_uem - besoins['uem']:.2f}")
                        colb.metric("PDIN apportée", f"{total_pdin:.2f} g", delta=f"{total_pdin - besoins['pdin']:.2f}")
                        colc.metric("MS apportée", f"{total_ms:.2f} kg", delta=f"{total_ms - besoins['ms']:.2f}")
                        st.metric("Coût journalier", f"{total_prix:.2f} DA")
                        if total_uem < besoins['uem'] * 0.9:
                            st.warning("⚠️ Apport énergétique insuffisant")
                        elif total_uem > besoins['uem'] * 1.1:
                            st.warning("⚠️ Excès d'énergie")
                        else:
                            st.success("✅ Énergie équilibrée")
                        if total_pdin < besoins['pdin'] * 0.9:
                            st.warning("⚠️ Apport protéique insuffisant")
                        elif total_pdin > besoins['pdin'] * 1.1:
                            st.warning("⚠️ Excès de protéines")
                        else:
                            st.success("✅ Protéines équilibrées")
                else:
                    st.subheader("Optimisation de la ration (coût minimum)")
                    n = len(aliments)
                    c = [a[6] for a in aliments]
                    A_ub = []
                    b_ub = []
                    A_ub.append([-a[3] for a in aliments])
                    b_ub.append(-besoins['uem'])
                    A_ub.append([-a[4] for a in aliments])
                    b_ub.append(-besoins['pdin'])
                    A_ub.append([a[5]/100 for a in aliments])
                    b_ub.append(besoins['ms'])
                    bounds = [(0, None) for _ in range(n)]
                    tolerance = st.slider("Tolérance sur les besoins (%)", 0, 20, 10) / 100
                    b_ub[0] = -besoins['uem'] * (1 - tolerance)
                    b_ub[1] = -besoins['pdin'] * (1 - tolerance)
                    b_ub[2] = besoins['ms'] * (1 + tolerance)
                    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    if res.success:
                        quantites = res.x
                        ration_opt = []
                        for i, q in enumerate(quantites):
                            if q > 0.01:
                                ration_opt.append({
                                    "nom": aliments[i][1],
                                    "qte": q,
                                    "uem": aliments[i][3],
                                    "pdin": aliments[i][4],
                                    "ms": aliments[i][5],
                                    "prix": aliments[i][6]
                                })
                        if ration_opt:
                            df_opt = pd.DataFrame(ration_opt)
                            df_opt["Coût (DA/jour)"] = df_opt["qte"] * df_opt["prix"]
                            st.dataframe(df_opt[["nom", "qte", "Coût (DA/jour)"]].round(2), use_container_width=True, hide_index=True)
                            total_opt = df_opt["Coût (DA/jour)"].sum()
                            st.metric("Coût optimal journalier", f"{total_opt:.2f} DA")
                            uem_tot = sum(q * aliments[i][3] for i, q in enumerate(quantites))
                            pdin_tot = sum(q * aliments[i][4] for i, q in enumerate(quantites))
                            ms_tot = sum(q * aliments[i][5]/100 for i, q in enumerate(quantites))
                            st.write(f"UEM apportée : {uem_tot:.2f} MJ (besoin {besoins['uem']})")
                            st.write(f"PDIN apportée : {pdin_tot:.2f} g (besoin {besoins['pdin']})")
                            st.write(f"MS apportée : {ms_tot:.2f} kg (max {besoins['ms']* (1+tolerance):.2f})")
                        else:
                            st.warning("Aucun aliment sélectionné par l'optimisation.")
                    else:
                        st.error("Impossible de trouver une solution optimale. Vérifiez les contraintes ou ajoutez des aliments.")
        else:
            st.info("Aucune brebis disponible. Vous pouvez utiliser 'Personnalisé'.")

# ---- PAGE PRODUCTION LAITIÈRE ----
def page_production():
    st.title("🥛 Production laitière et analyses biochimiques")
    tab1, tab2 = st.tabs(["📈 Suivi production", "🧪 Analyses biochimiques"])
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom, e.nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis_list = db.fetchall(query_brebis, params)
    brebis_dict = {f"{b[0]} - {b[1]} {b[2]} ({b[3]})": b[0] for b in brebis_list}
    if not brebis_dict:
        st.warning("Aucune brebis disponible pour cet éleveur.")
        return
    with tab1:
        st.subheader("Saisie d'une production")
        with st.form("form_prod"):
            brebis_choice = st.selectbox("Brebis", list(brebis_dict.keys()))
            date_prod = st.date_input("Date", value=datetime.today().date())
            quantite = st.number_input("Quantité de lait (L)", min_value=0.0, step=0.1)
            if st.form_submit_button("Enregistrer production"):
                brebis_id = brebis_dict[brebis_choice]
                db.execute(
                    "INSERT INTO productions (brebis_id, date, quantite) VALUES (?, ?, ?)",
                    (brebis_id, date_prod.isoformat(), quantite)
                )
                st.success("Production enregistrée")
                st.rerun()
        st.subheader("Évolution de la production")
        brebis_graph = st.selectbox("Choisir une brebis pour le graphique", list(brebis_dict.keys()), key="graph_brebis")
        bid = brebis_dict[brebis_graph]
        data = db.fetchall(
            "SELECT date, quantite FROM productions WHERE brebis_id=? ORDER BY date",
            (bid,)
        )
        if data:
            df = pd.DataFrame(data, columns=["Date", "Quantité (L)"])
            df["Date"] = pd.to_datetime(df["Date"])
            fig = px.line(df, x="Date", y="Quantité (L)", title=f"Production de {brebis_graph}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée pour cette brebis.")
        st.subheader("Production par éleveur")
        data_all = db.fetchall("""
            SELECT el.nom AS eleveur, b.numero_id, p.date, p.quantite
            FROM productions p
            JOIN brebis b ON p.brebis_id = b.id
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
            ORDER BY p.date
        """, (st.session_state.user_id,))
        if data_all:
            df_all = pd.DataFrame(data_all, columns=["Éleveur", "Brebis", "Date", "Quantité"])
            df_all["Date"] = pd.to_datetime(df_all["Date"])
            fig2 = px.line(df_all, x="Date", y="Quantité", color="Brebis", line_group="Brebis",
                          title="Production par brebis")
            st.plotly_chart(fig2, use_container_width=True)
            total_par_eleveur = df_all.groupby("Éleveur")["Quantité"].sum().reset_index()
            fig3 = px.bar(total_par_eleveur, x="Éleveur", y="Quantité", title="Production totale par éleveur")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Aucune donnée de production.")
    with tab2:
        st.subheader("Analyses biochimiques du lait")
        with st.form("form_biochimie"):
            brebis_choice2 = st.selectbox("Brebis", list(brebis_dict.keys()), key="bio_brebis")
            date_bio = st.date_input("Date de l'analyse", value=datetime.today().date())
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.7, step=0.1)
            mg = st.number_input("Matière grasse (g/L)", min_value=0.0, value=65.0, step=0.1)
            proteine = st.number_input("Protéines (g/L)", min_value=0.0, value=55.0, step=0.1)
            ag_satures = st.number_input("Acides gras saturés (g/L)", min_value=0.0, value=35.0, step=0.1)
            densite = st.number_input("Densité", min_value=1.0, max_value=1.1, value=1.035, step=0.001, format="%.3f")
            extrait_sec = st.number_input("Extrait sec (g/L)", min_value=0.0, value=180.0, step=0.1)
            if st.form_submit_button("Enregistrer analyse"):
                brebis_id = brebis_dict[brebis_choice2]
                existing = db.fetchone(
                    "SELECT id FROM productions WHERE brebis_id=? AND date=?",
                    (brebis_id, date_bio.isoformat())
                )
                if existing:
                    db.execute("""
                        UPDATE productions SET ph=?, mg=?, proteine=?, ag_satures=?, densite=?, extrait_sec=?
                        WHERE id=?
                    """, (ph, mg, proteine, ag_satures, densite, extrait_sec, existing[0]))
                else:
                    db.execute("""
                        INSERT INTO productions 
                        (brebis_id, date, ph, mg, proteine, ag_satures, densite, extrait_sec)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (brebis_id, date_bio.isoformat(), ph, mg, proteine, ag_satures, densite, extrait_sec))
                st.success("Analyse enregistrée")
                st.rerun()
        st.subheader("Dernières analyses enregistrées")
        data_bio = db.fetchall("""
            SELECT b.numero_id, b.nom, p.date, p.ph, p.mg, p.proteine, p.ag_satures, p.densite, p.extrait_sec
            FROM productions p
            JOIN brebis b ON p.brebis_id = b.id
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=? AND (p.ph IS NOT NULL OR p.mg IS NOT NULL)
            ORDER BY p.date DESC LIMIT 20
        """, (st.session_state.user_id,))
        if data_bio:
            df_bio = pd.DataFrame(data_bio, columns=["Numéro", "Nom", "Date", "pH", "MG", "Protéines", "AGS", "Densité", "Extrait sec"])
            st.dataframe(df_bio, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune analyse biochimique.")

# ---- PAGE GÉNOMIQUE AVANCÉE ----
def page_genomique_avancee():
    st.title("🧬 Génomique avancée")
    tab1, tab2, tab3 = st.tabs(["🔍 BLAST", "🧬 SNPs d'intérêt", "📊 GWAS"])
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis_list = db.fetchall(query_brebis, params)
    brebis_dict = {f"{b[0]} - {b[1]} {b[2]}": b[0] for b in brebis_list}
    with tab1:
        st.subheader("Alignement BLAST sur NCBI")
        default_seq = ""
        if brebis_dict:
            blast_brebis = st.selectbox("Sélectionner une brebis (pour utiliser sa séquence FASTA)", 
                                        ["Nouvelle séquence"] + list(brebis_dict.keys()))
            if blast_brebis != "Nouvelle séquence":
                bid = brebis_dict[blast_brebis]
                seq_result = db.fetchone("SELECT sequence_fasta FROM brebis WHERE id=?", (bid,))
                if seq_result and seq_result[0]:
                    default_seq = seq_result[0]
        seq_input = st.text_area("Séquence FASTA", value=default_seq, height=150)
        database = st.selectbox("Base de données", ["nr", "nt", "refseq_rna", "refseq_protein"])
        if st.button("Lancer BLAST"):
            if not seq_input:
                st.error("Veuillez entrer une séquence.")
            else:
                with st.spinner("Recherche BLAST en cours..."):
                    try:
                        url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
                        params = {
                            "CMD": "Put",
                            "PROGRAM": "blastn",
                            "DATABASE": database,
                            "QUERY": seq_input,
                            "FORMAT_TYPE": "JSON2"
                        }
                        requests.post(url, data=params)
                        st.warning("Le BLAST en ligne est complexe à intégrer. Pour une démonstration, nous affichons un résultat factice.")
                        time.sleep(2)
                        st.success("BLAST terminé (simulation)")
                        mock_results = [
                            {"accession": "XM_004012345.1", "description": "Ovis aries BMP15 mRNA", "score": 1234, "evalue": 1e-150},
                            {"accession": "NM_001009345.1", "description": "Ovis aries MSTN mRNA", "score": 1100, "evalue": 1e-140},
                        ]
                        df_mock = pd.DataFrame(mock_results)
                        st.dataframe(df_mock)
                        if st.button("Enregistrer ce résultat"):
                            st.info("Fonctionnalité à implémenter (sauvegarde en base)")
                    except Exception as e:
                        st.error(f"Erreur BLAST: {e}")
    with tab2:
        st.subheader("SNPs d'intérêt économique")
        st.markdown("**Gènes d'intérêt et SNPs associés**")
        df_genes = pd.DataFrame([
            {"Gène": sym, "Nom": info["nom"], "Effet": info["effet"]}
            for sym, info in Config.GENES_ECONOMIQUES.items()
        ])
        st.dataframe(df_genes, use_container_width=True, hide_index=True)
        if brebis_dict:
            selected = st.selectbox("Charger les SNPs d'une brebis", list(brebis_dict.keys()))
            bid = brebis_dict[selected]
            variants = db.fetchone("SELECT variants_snps FROM brebis WHERE id=?", (bid,))
            if variants and variants[0]:
                try:
                    snps = json.loads(variants[0])
                    st.json(snps)
                except:
                    st.info("Les SNPs ne sont pas au format JSON valide.")
            else:
                st.info("Aucun SNP enregistré pour cette brebis.")
            with st.expander("Ajouter / modifier les SNPs"):
                snps_json = st.text_area("SNPs au format JSON (ex: {'BMP15': 'AA', 'MSTN': 'GG'})", height=150)
                if st.button("Enregistrer"):
                    db.execute("UPDATE brebis SET variants_snps=? WHERE id=?", (snps_json, bid))
                    st.success("SNPs enregistrés")
                    st.rerun()
    with tab3:
        st.subheader("Analyse d'association GWAS")
        st.markdown("""
        Cette section permet de réaliser une étude d'association pangénomique simplifiée.
        Vous devez fournir deux fichiers CSV :
        - **Génotypes** : avec une colonne `brebis_id` et une colonne par SNP (valeurs 0,1,2 pour le dosage allélique).
        - **Phénotypes** : avec les colonnes `brebis_id` et un trait quantitatif (ex: production laitière, poids...).
        """)
        upload_geno = st.file_uploader("Fichier génotypes (CSV)", type="csv", key="geno")
        upload_pheno = st.file_uploader("Fichier phénotypes (CSV)", type="csv", key="pheno")
        if upload_geno and upload_pheno:
            try:
                df_geno = pd.read_csv(upload_geno)
                df_pheno = pd.read_csv(upload_pheno)
                if 'brebis_id' not in df_geno.columns or 'brebis_id' not in df_pheno.columns:
                    st.error("Les fichiers doivent contenir une colonne 'brebis_id'.")
                else:
                    df_merged = pd.merge(df_geno, df_pheno, on='brebis_id')
                    trait_col = st.selectbox("Sélectionner le trait phénotypique", 
                                             [c for c in df_pheno.columns if c != 'brebis_id'])
                    snp_cols = [c for c in df_geno.columns if c != 'brebis_id' and df_geno[c].dtype in ['int64', 'float64']]
                    if len(snp_cols) == 0:
                        st.error("Aucune colonne SNP numérique trouvée.")
                    else:
                        st.write(f"Nombre de SNPs analysés : {len(snp_cols)}")
                        results = []
                        pbar = st.progress(0)
                        for i, snp in enumerate(snp_cols):
                            X = df_merged[snp].values
                            y = df_merged[trait_col].values
                            X = sm.add_constant(X)
                            model = sm.OLS(y, X).fit()
                            p_value = model.pvalues[1]
                            beta = model.params[1]
                            results.append({
                                'SNP': snp,
                                'Beta': beta,
                                'P_value': p_value,
                                '-log10(p)': -np.log10(p_value) if p_value > 0 else 10
                            })
                            pbar.progress((i+1)/len(snp_cols))
                        df_res = pd.DataFrame(results)
                        fig = px.scatter(df_res, x='SNP', y='-log10(p)', 
                                         title="Manhattan plot",
                                         labels={'-log10(p)': '-log10(p-value)'},
                                         hover_data=['Beta', 'P_value'])
                        fig.add_hline(y=-np.log10(0.05/len(snp_cols)), line_dash="dash", 
                                      annotation_text="Bonferroni threshold")
                        st.plotly_chart(fig, use_container_width=True)
                        sig = df_res[df_res['P_value'] < 0.05]
                        if not sig.empty:
                            st.subheader("SNPs suggestifs (p < 0.05)")
                            st.dataframe(sig.sort_values('P_value'), use_container_width=True, hide_index=True)
                        else:
                            st.info("Aucun SNP significatif au seuil de 0.05.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")

# ---- PAGE GESTION ÉLEVAGE (inchangée, mais reprise complète) ----
def page_gestion_elevage():
    st.title("🐑 Gestion des élevages")
    if st.session_state.eleveur_id is not None:
        eleveur = db.fetchone("SELECT nom, region FROM eleveurs WHERE id=?", (st.session_state.eleveur_id,))
        if eleveur:
            st.subheader(f"📊 Résumé de l'éleveur : {eleveur[0]} ({eleveur[1]})")
            nb_elevages = db.fetchone("SELECT COUNT(*) FROM elevages WHERE eleveur_id=?", (st.session_state.eleveur_id,))[0]
            nb_brebis = db.fetchone("""
                SELECT COUNT(*) FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                WHERE e.eleveur_id=?
            """, (st.session_state.eleveur_id,))[0]
            prod_moy = db.fetchone("""
                SELECT AVG(p.quantite)
                FROM productions p
                JOIN brebis b ON p.brebis_id = b.id
                JOIN elevages e ON b.elevage_id = e.id
                WHERE e.eleveur_id=? AND p.date >= date('now', '-30 days')
            """, (st.session_state.eleveur_id,))[0]
            poids_moy = db.fetchone("""
                SELECT AVG(b.poids_vif)
                FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                WHERE e.eleveur_id=?
            """, (st.session_state.eleveur_id,))[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏡 Élevages", nb_elevages)
            col2.metric("🐑 Brebis", nb_brebis)
            col3.metric("🥛 Production moy. (L/j)", f"{prod_moy:.2f}" if prod_moy else "N/A")
            col4.metric("⚖️ Poids moy. (kg)", f"{poids_moy:.1f}" if poids_moy else "N/A")
            races = db.fetchall("""
                SELECT b.race, COUNT(*) 
                FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                WHERE e.eleveur_id=?
                GROUP BY b.race
            """, (st.session_state.eleveur_id,))
            if races:
                df_races = pd.DataFrame(races, columns=["Race", "Nombre"])
                fig = px.pie(df_races, values="Nombre", names="Race", title="Répartition des races")
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.info("👈 Sélectionnez un éleveur dans la barre latérale pour voir un résumé.")
    tab1, tab2, tab3 = st.tabs(["👨‍🌾 Éleveurs", "🏡 Élevages", "🐑 Brebis"])
    with tab1:
        st.subheader("Liste des éleveurs")
        with st.expander("➕ Ajouter un éleveur", expanded=True):
            with st.form("form_eleveur"):
                nom = st.text_input("Nom")
                region = st.text_input("Région")
                telephone = st.text_input("Téléphone")
                email = st.text_input("Email")
                submitted = st.form_submit_button("Ajouter")
                if submitted:
                    db.execute(
                        "INSERT INTO eleveurs (user_id, nom, region, telephone, email) VALUES (?, ?, ?, ?, ?)",
                        (st.session_state.user_id, nom, region, telephone, email)
                    )
                    st.success("Éleveur ajouté")
                    st.rerun()
        eleveurs = db.fetchall(
            "SELECT id, nom, region, telephone, email FROM eleveurs WHERE user_id=?",
            (st.session_state.user_id,)
        )
        if eleveurs:
            df = pd.DataFrame(eleveurs, columns=["ID", "Nom", "Région", "Téléphone", "Email"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("🗑️ Supprimer un éleveur"):
                del_id = st.selectbox("Choisir l'éleveur", [f"{e[0]} - {e[1]}" for e in eleveurs], key="del_eleveur_select")
                if st.button("Supprimer", key="del_eleveur_btn"):
                    eid = int(del_id.split(" - ")[0])
                    count = db.fetchone("SELECT COUNT(*) FROM elevages WHERE eleveur_id=?", (eid,))[0]
                    if count > 0:
                        st.error("Cet éleveur a encore des élevages. Supprimez d'abord les élevages.")
                    else:
                        db.execute("DELETE FROM eleveurs WHERE id=?", (eid,))
                        st.success("Éleveur supprimé")
                        st.rerun()
        else:
            st.info("Aucun éleveur enregistré.")
    with tab2:
        st.subheader("Liste des élevages")
        eleveurs_list = db.fetchall(
            "SELECT id, nom FROM eleveurs WHERE user_id=?", (st.session_state.user_id,)
        )
        eleveurs_dict = {f"{e[0]} - {e[1]}": e[0] for e in eleveurs_list}
        if not eleveurs_dict:
            st.warning("Vous devez d'abord ajouter un éleveur.")
        else:
            with st.expander("➕ Ajouter un élevage", expanded=True):
                with st.form("form_elevage"):
                    eleveur_choice = st.selectbox("Éleveur", list(eleveurs_dict.keys()))
                    nom_elevage = st.text_input("Nom de l'élevage")
                    localisation = st.text_input("Localisation")
                    superficie = st.number_input("Superficie (ha)", min_value=0.0, step=0.1)
                    submitted = st.form_submit_button("Ajouter")
                    if submitted:
                        eleveur_id = eleveurs_dict[eleveur_choice]
                        db.execute(
                            "INSERT INTO elevages (eleveur_id, nom, localisation, superficie) VALUES (?, ?, ?, ?)",
                            (eleveur_id, nom_elevage, localisation, superficie)
                        )
                        st.success("Élevage ajouté")
                        st.rerun()
            params = [st.session_state.user_id]
            query = """
                SELECT e.id, e.nom, e.localisation, e.superficie, el.nom
                FROM elevages e
                JOIN eleveurs el ON e.eleveur_id = el.id
                WHERE el.user_id=?
            """
            query, params = filtrer_par_eleveur(query, params, join_eleveur=True)
            elevages = db.fetchall(query, params)
            if not elevages:
                st.info("Aucun élevage pour cet éleveur.")
            else:
                df = pd.DataFrame(elevages, columns=["ID", "Nom", "Localisation", "Superficie", "Éleveur"])
                st.dataframe(df, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Liste des brebis")
        params_elev = [st.session_state.user_id]
        query_elev = """
            SELECT e.id, e.nom, el.nom
            FROM elevages e
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
        """
        query_elev, params_elev = filtrer_par_eleveur(query_elev, params_elev, join_eleveur=True)
        elevages_list = db.fetchall(query_elev, params_elev)
        elevages_dict = {f"{e[0]} - {e[1]} ({e[2]})": e[0] for e in elevages_list}
        if not elevages_dict:
            st.warning("Aucun élevage pour cet éleveur. Veuillez d'abord ajouter un élevage.")
        else:
            with st.expander("➕ Ajouter une brebis", expanded=False):
                with st.form("form_brebis"):
                    elevage_choice = st.selectbox("Élevage", list(elevages_dict.keys()))
                    numero_id = st.text_input("Numéro d'identification (obligatoire)")
                    age_mode = st.radio("Mode de saisie de l'âge", ["Âge en mois", "Dentition"])
                    date_naissance = None
                    if age_mode == "Âge en mois":
                        age_mois = st.number_input("Âge en mois", min_value=0, max_value=200, value=24, step=1)
                        date_naissance = datetime.today().date() - timedelta(days=age_mois * 30)
                        st.date_input("Date estimée (d'après âge)", value=date_naissance, disabled=True)
                    else:
                        dentition = st.selectbox("Dentition", ["Dents de lait", "2 dents", "4 dents", "6 dents ou plus"])
                        if dentition == "Dents de lait":
                            age_estime_mois = 6
                        elif dentition == "2 dents":
                            age_estime_mois = 18
                        elif dentition == "4 dents":
                            age_estime_mois = 30
                        else:
                            age_estime_mois = 48
                        date_naissance = datetime.today().date() - timedelta(days=age_estime_mois * 30)
                        st.date_input("Date estimée (d'après dentition)", value=date_naissance, disabled=True)
                    race = st.selectbox("Race", list(Config.RACES.keys()))
                    etat_physio = st.selectbox("État physiologique", Config.ETATS_PHYSIO)
                    photo_profil = st.file_uploader("Photo de profil (optionnelle)", type=['jpg','png','jpeg'])
                    photo_mamelle = st.file_uploader("Photo mamelle (optionnelle)", type=['jpg','png','jpeg'])
                    poids_vif = st.number_input("Poids vif (kg) (optionnel)", min_value=0.0, value=0.0, step=0.5)
                    submitted = st.form_submit_button("Ajouter")
                    if submitted and numero_id:
                        existing = db.fetchone("SELECT id FROM brebis WHERE numero_id=?", (numero_id,))
                        if existing:
                            st.error(f"Une brebis avec le numéro {numero_id} existe déjà.")
                        else:
                            elevage_id = elevages_dict[elevage_choice]
                            profil_filename = save_uploaded_photo(photo_profil)
                            mamelle_filename = save_uploaded_photo(photo_mamelle)
                            db.execute("""
                                INSERT INTO brebis 
                                (elevage_id, numero_id, nom, race, date_naissance, etat_physio, photo_profil, photo_mamelle, poids_vif)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                elevage_id, numero_id, "", race,
                                date_naissance.isoformat(), etat_physio,
                                profil_filename, mamelle_filename, poids_vif if poids_vif > 0 else None
                            ))
                            st.success("Brebis ajoutée")
                            st.rerun()
                    elif submitted and not numero_id:
                        st.error("Le numéro d'identification est obligatoire.")
            params_brebis = [st.session_state.user_id]
            query_brebis = """
                SELECT b.id, b.numero_id, b.nom, b.race, b.date_naissance, b.etat_physio, e.nom, b.poids_vif, b.photo_profil, b.photo_mamelle
                FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                JOIN eleveurs el ON e.eleveur_id = el.id
                WHERE el.user_id=?
            """
            query_brebis, params_brebis = filtrer_par_eleveur(query_brebis, params_brebis, join_eleveur=True)
            brebis = db.fetchall(query_brebis, params_brebis)
            if brebis:
                df_brebis = pd.DataFrame(brebis, columns=["ID", "Numéro", "Nom", "Race", "Naissance", "État", "Élevage", "Poids vif (kg)", "Photo profil", "Photo mamelle"])
                st.dataframe(df_brebis[["Numéro", "Race", "Naissance", "État", "Élevage", "Poids vif (kg)"]], use_container_width=True, hide_index=True)
                st.divider()
                st.subheader("🐑 Suivi individuel")
                selected_brebis = st.selectbox("Choisir une brebis", [f"{b[0]} - {b[1]}" for b in brebis], key="suivi_select")
                bid = int(selected_brebis.split(" - ")[0])
                brebis_info = db.fetchone("SELECT numero_id, nom, race, date_naissance, poids_vif, photo_profil, photo_mamelle FROM brebis WHERE id=?", (bid,))
                if brebis_info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Numéro", brebis_info[0])
                    col2.metric("Nom", brebis_info[1] if brebis_info[1] else "Non renseigné")
                    col3.metric("Race", brebis_info[2])
                    if brebis_info[3]:
                        naiss = datetime.strptime(brebis_info[3], "%Y-%m-%d").date()
                        age_jours = (datetime.today().date() - naiss).days
                        age_mois = age_jours // 30
                        age_ans = age_jours // 365
                        st.metric("Âge", f"{age_ans} ans ({age_mois} mois)")
                    else:
                        st.metric("Âge", "Non renseigné")
                    st.metric("Dernier poids connu", f"{brebis_info[4]} kg" if brebis_info[4] else "Non renseigné")
                tab_hist1, tab_hist2, tab_hist3, tab_hist4 = st.tabs(["📈 Poids", "🥛 Production", "📏 Morphométrie", "📝 Notes"])
                with tab_hist1:
                    poids_data = db.fetchall("""
                        SELECT date_estimation, poids_vif FROM composition_corporelle 
                        WHERE brebis_id=? ORDER BY date_estimation
                    """, (bid,))
                    if poids_data:
                        df_poids = pd.DataFrame(poids_data, columns=["Date", "Poids (kg)"])
                        df_poids["Date"] = pd.to_datetime(df_poids["Date"])
                        fig_poids = px.line(df_poids, x="Date", y="Poids (kg)", title="Évolution du poids")
                        st.plotly_chart(fig_poids, use_container_width=True)
                    else:
                        st.info("Aucune donnée de poids historique.")
                    with st.form("form_poids"):
                        new_poids = st.number_input("Nouveau poids (kg)", min_value=0.0, step=0.1)
                        if st.form_submit_button("Ajouter ce poids"):
                            db.execute("""
                                INSERT INTO composition_corporelle 
                                (brebis_id, date_estimation, poids_vif, poids_carcasse, rendement_carcasse,
                                 poids_viande, pct_viande, poids_graisse, pct_graisse, poids_os, pct_os,
                                 gigot_poids, epaule_poids, cotelette_poids)
                                VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                            """, (bid, datetime.now().isoformat(), new_poids))
                            st.success("Poids enregistré !")
                            st.rerun()
                with tab_hist2:
                    prod_data = db.fetchall("""
                        SELECT date, quantite FROM productions WHERE brebis_id=? ORDER BY date
                    """, (bid,))
                    if prod_data:
                        df_prod = pd.DataFrame(prod_data, columns=["Date", "Lait (L)"])
                        df_prod["Date"] = pd.to_datetime(df_prod["Date"])
                        fig_prod = px.line(df_prod, x="Date", y="Lait (L)", title="Production laitière")
                        st.plotly_chart(fig_prod, use_container_width=True)
                    else:
                        st.info("Aucune donnée de production.")
                    with st.form("form_prod_suivi"):
                        date_prod = st.date_input("Date", value=datetime.today().date())
                        quantite = st.number_input("Quantité (L)", min_value=0.0, step=0.1)
                        if st.form_submit_button("Enregistrer production"):
                            db.execute("INSERT INTO productions (brebis_id, date, quantite) VALUES (?, ?, ?)",
                                      (bid, date_prod.isoformat(), quantite))
                            st.success("Production enregistrée !")
                            st.rerun()
                with tab_hist3:
                    morpho_data = db.fetchall("""
                        SELECT date_mesure, longueur_corps, hauteur_garrot, tour_poitrine, 
                               circonference_canon, largeur_bassin, score_global
                        FROM mesures_morpho WHERE brebis_id=? ORDER BY date_mesure
                    """, (bid,))
                    if morpho_data:
                        df_morpho = pd.DataFrame(morpho_data, columns=["Date", "Longueur", "Hauteur", "Poitrine", "Canon", "Bassin", "Score"])
                        df_morpho["Date"] = pd.to_datetime(df_morpho["Date"])
                        st.dataframe(df_morpho.drop(columns=["Date"]), use_container_width=True, hide_index=True)
                        fig_score = px.line(df_morpho, x="Date", y="Score", title="Évolution du score morphologique")
                        st.plotly_chart(fig_score, use_container_width=True)
                    else:
                        st.info("Aucune mesure morphométrique.")
                    if st.button("📸 Aller à la photogrammétrie pour cette brebis"):
                        st.session_state.brebis_analyse_id = bid
                        st.session_state.current_page = "analyse_auto"
                        st.rerun()
                with tab_hist4:
                    diag_data = db.fetchall("""
                        SELECT date, maladie, symptomes, traitement FROM diagnostics WHERE brebis_id=? ORDER BY date DESC
                    """, (bid,))
                    if diag_data:
                        df_diag = pd.DataFrame(diag_data, columns=["Date", "Maladie", "Symptômes", "Traitement"])
                        st.dataframe(df_diag, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune note de diagnostic.")
                    with st.form("form_diag"):
                        date_diag = st.date_input("Date", value=datetime.today().date())
                        maladie = st.text_input("Maladie / Observation")
                        symptomes = st.text_area("Symptômes")
                        traitement = st.text_area("Traitement")
                        if st.form_submit_button("Enregistrer"):
                            db.execute("""
                                INSERT INTO diagnostics (brebis_id, date, maladie, symptomes, traitement)
                                VALUES (?, ?, ?, ?, ?)
                            """, (bid, date_diag.isoformat(), maladie, symptomes, traitement))
                            st.success("Note enregistrée !")
                            st.rerun()
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Supprimer cette brebis", key="del_brebis_suivi"):
                        photos = db.fetchone("SELECT photo_profil, photo_mamelle FROM brebis WHERE id=?", (bid,))
                        if photos:
                            for p in photos:
                                if p:
                                    try:
                                        os.remove(os.path.join(PHOTO_DIR, p))
                                    except:
                                        pass
                        db.execute("DELETE FROM brebis WHERE id=?", (bid,))
                        st.success("Brebis supprimée")
                        st.rerun()
                with col2:
                    if st.button("📋 Voir détails complets", key="details_brebis_suivi"):
                        b = db.fetchone("SELECT * FROM brebis WHERE id=?", (bid,))
                        cols = [col[0] for col in db.conn.execute("PRAGMA table_info(brebis)").fetchall()]
                        data = dict(zip(cols, b))
                        with st.expander("Détails de la brebis", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Numéro", data.get('numero_id', 'N/A'))
                                st.metric("Race", data.get('race', 'N/A'))
                                st.metric("État physiologique", data.get('etat_physio', 'N/A'))
                                if data.get('date_naissance'):
                                    naiss = datetime.strptime(data['date_naissance'], "%Y-%m-%d").date()
                                    age_jours = (datetime.today().date() - naiss).days
                                    age_mois = age_jours // 30
                                    st.metric("Âge", f"{age_mois} mois")
                            with col_b:
                                st.metric("Poids vif (kg)", data.get('poids_vif', 'Non renseigné'))
                                if data.get('photo_profil'):
                                    st.image(os.path.join(PHOTO_DIR, data['photo_profil']), caption="Photo de profil", width=200)
                                if data.get('photo_mamelle'):
                                    st.image(os.path.join(PHOTO_DIR, data['photo_mamelle']), caption="Photo mamelle", width=200)
            else:
                st.info("Aucune brebis enregistrée.")

# ---- PAGE SANTÉ ----
def page_sante():
    st.title("🏥 Suivi sanitaire et vaccinal")
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom, e.nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis_list = db.fetchall(query_brebis, params)
    brebis_dict = {f"{b[0]} - {b[1]} {b[2]} ({b[3]})": b[0] for b in brebis_list}
    if not brebis_dict:
        st.warning("Aucune brebis disponible.")
        return
    selected = st.selectbox("Choisir une brebis", list(brebis_dict.keys()), key="sante_brebis")
    bid = brebis_dict[selected]
    brebis_infos = db.fetchone("SELECT nom, numero_id, date_naissance, race FROM brebis WHERE id=?", (bid,))
    if brebis_infos:
        nom, numero, naiss, race = brebis_infos
        age = (datetime.now() - datetime.strptime(naiss, "%Y-%m-%d")).days // 365 if naiss else 0
        st.info(f"**{nom}** ({numero}) - {race}, {age} ans")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📜 Historique", 
        "⏰ Rappels", 
        "📊 Statistiques", 
        "🤖 IA & Prédictions", 
        "📤 Export"
    ])
    with tab1:
        st.subheader("Historique des soins et vaccins")
        vaccins = db.fetchall("""
            SELECT date_vaccin, vaccin, rappel, 'Vaccin' as type
            FROM vaccinations WHERE brebis_id=?
        """, (bid,))
        soins = db.fetchall("""
            SELECT date_soin, diagnostic, traitement, type as type
            FROM soins WHERE brebis_id=?
        """, (bid,))
        historique = []
        for v in vaccins:
            historique.append({
                "Date": v[0],
                "Type": v[3],
                "Description": f"{v[1]} (rappel le {v[2]})" if v[2] else v[1],
                "Détails": ""
            })
        for s in soins:
            historique.append({
                "Date": s[0],
                "Type": s[3],
                "Description": s[1],
                "Détails": s[2]
            })
        if historique:
            df_hist = pd.DataFrame(historique)
            df_hist["Date"] = pd.to_datetime(df_hist["Date"])
            df_hist = df_hist.sort_values("Date", ascending=False)
            types = df_hist["Type"].unique().tolist()
            selected_types = st.multiselect("Filtrer par type", types, default=types)
            df_filtre = df_hist[df_hist["Type"].isin(selected_types)]
            st.dataframe(df_filtre, use_container_width=True, hide_index=True)
            df_count = df_filtre.groupby([df_filtre["Date"].dt.to_period("M"), "Type"]).size().reset_index(name="Nombre")
            df_count["Date"] = df_count["Date"].astype(str)
            fig = px.bar(df_count, x="Date", y="Nombre", color="Type", title="Événements par mois")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun événement enregistré pour cette brebis.")
        with st.expander("➕ Ajouter un événement"):
            type_evt = st.radio("Type", ["Soin", "Vaccin"])
            if type_evt == "Vaccin":
                with st.form("form_vaccin_rapide"):
                    date_vaccin = st.date_input("Date du vaccin", value=datetime.today().date())
                    vaccin = st.text_input("Nom du vaccin")
                    rappel = st.date_input("Date de rappel (optionnelle)", value=None)
                    if st.form_submit_button("Enregistrer"):
                        db.execute(
                            "INSERT INTO vaccinations (brebis_id, date_vaccin, vaccin, rappel) VALUES (?, ?, ?, ?)",
                            (bid, date_vaccin.isoformat(), vaccin, rappel.isoformat() if rappel else None)
                        )
                        st.success("Vaccin enregistré")
                        st.rerun()
            else:
                with st.form("form_soin_rapide"):
                    date_soin = st.date_input("Date du soin", value=datetime.today().date())
                    type_soin = st.selectbox("Type", ["Maladie", "Parasite", "Blessure", "Autre"])
                    diagnostic = st.text_area("Diagnostic / Symptômes")
                    traitement = st.text_area("Traitement administré")
                    if st.form_submit_button("Enregistrer"):
                        db.execute(
                            "INSERT INTO soins (brebis_id, date_soin, type, diagnostic, traitement) VALUES (?, ?, ?, ?, ?)",
                            (bid, date_soin.isoformat(), type_soin, diagnostic, traitement)
                        )
                        st.success("Soin enregistré")
                        st.rerun()
    with tab2:
        st.subheader("Rappels à venir")
        rappels = db.fetchall("""
            SELECT vaccin, rappel FROM vaccinations
            WHERE brebis_id=? AND rappel IS NOT NULL AND rappel >= date('now')
            ORDER BY rappel
        """, (bid,))
        if rappels:
            df_rappels = pd.DataFrame(rappels, columns=["Vaccin", "Date de rappel"])
            df_rappels["Jours restants"] = (pd.to_datetime(df_rappels["Date de rappel"]) - datetime.now()).dt.days
            st.dataframe(df_rappels, use_container_width=True, hide_index=True)
            imminents = df_rappels[df_rappels["Jours restants"] <= 7]
            if not imminents.empty:
                st.warning("⚠️ Certains rappels sont imminents !")
                st.dataframe(imminents)
        else:
            st.info("Aucun rappel programmé.")
        soins_recents = db.fetchall("""
            SELECT date_soin, type, diagnostic, traitement
            FROM soins
            WHERE brebis_id=? AND date_soin >= date('now', '-30 days')
            ORDER BY date_soin DESC
        """, (bid,))
        if soins_recents:
            st.subheader("Traitements récents (mois en cours)")
            df_recents = pd.DataFrame(soins_recents, columns=["Date", "Type", "Diagnostic", "Traitement"])
            st.dataframe(df_recents, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Statistiques sanitaires")
        soins_stats = db.fetchall("""
            SELECT type, COUNT(*) FROM soins WHERE brebis_id=? GROUP BY type
        """, (bid,))
        if soins_stats:
            df_stats = pd.DataFrame(soins_stats, columns=["Type", "Nombre"])
            fig = px.pie(df_stats, values="Nombre", names="Type", title="Répartition des soins par type")
            st.plotly_chart(fig, use_container_width=True)
        soins_temp = db.fetchall("""
            SELECT strftime('%Y-%m', date_soin) as mois, COUNT(*) 
            FROM soins WHERE brebis_id=?
            GROUP BY mois
            ORDER BY mois
        """, (bid,))
        if soins_temp:
            df_temp = pd.DataFrame(soins_temp, columns=["Mois", "Nombre"])
            fig2 = px.line(df_temp, x="Mois", y="Nombre", title="Évolution du nombre de soins")
            st.plotly_chart(fig2, use_container_width=True)
        dernier_vaccin = db.fetchone("""
            SELECT MAX(date_vaccin) FROM vaccinations WHERE brebis_id=?
        """, (bid,))[0]
        if dernier_vaccin:
            jours_depuis = (datetime.now() - datetime.strptime(dernier_vaccin, "%Y-%m-%d")).days
            st.metric("Dernier vaccin", f"il y a {jours_depuis} jours")
        else:
            st.info("Aucun vaccin enregistré.")
    with tab4:
        st.subheader("Intelligence Artificielle – Analyses prédictives")
        st.caption(
            "Score de risque sanitaire calculé à partir de l'historique réel de l'animal "
            "(fréquence des soins récents, retard vaccinal, diagnostics antérieurs)."
        )
        if st.button("Évaluer le risque pour cette brebis"):
            nb_soins_90j = db.fetchone(
                "SELECT COUNT(*) FROM soins WHERE brebis_id=? AND date_soin >= date('now', '-90 days')",
                (bid,)
            )[0]
            nb_diagnostics = db.fetchone(
                "SELECT COUNT(*) FROM diagnostics WHERE brebis_id=?", (bid,)
            )[0]
            dernier_vaccin_row = db.fetchone(
                "SELECT MAX(date_vaccin) FROM vaccinations WHERE brebis_id=?", (bid,)
            )
            dernier_vaccin = dernier_vaccin_row[0] if dernier_vaccin_row else None
            retard_vaccin_jours = None
            if dernier_vaccin:
                retard_vaccin_jours = (datetime.now() - datetime.strptime(dernier_vaccin, "%Y-%m-%d")).days

            score = 0
            score += min(nb_soins_90j, 5) * 10          # soins fréquents récents
            score += min(nb_diagnostics, 5) * 6          # historique de maladies
            if retard_vaccin_jours is None or retard_vaccin_jours > 365:
                score += 25                               # jamais vacciné ou rappel très en retard
            elif retard_vaccin_jours > 180:
                score += 10

            if score >= 45:
                risque = "Élevé"
            elif score >= 20:
                risque = "Modéré"
            else:
                risque = "Faible"

            st.metric("Risque estimé", risque, help=f"Score composite : {score}/100")
            with st.expander("Détail du calcul"):
                st.write(f"- Soins sur 90 jours : **{nb_soins_90j}**")
                st.write(f"- Diagnostics enregistrés : **{nb_diagnostics}**")
                st.write(
                    f"- Dernier vaccin : **{'il y a ' + str(retard_vaccin_jours) + ' jours' if retard_vaccin_jours is not None else 'aucun enregistré'}**"
                )
        prod_recentes = db.fetchall("""
            SELECT quantite FROM productions 
            WHERE brebis_id=? AND date >= date('now', '-60 days')
            ORDER BY date
        """, (bid,))
        if len(prod_recentes) >= 5:
            valeurs = np.array([p[0] for p in prod_recentes], dtype=float)
            moyenne, ecart_type = valeurs[:-1].mean(), valeurs[:-1].std()
            derniere = valeurs[-1]
            if ecart_type > 0:
                z_score = (derniere - moyenne) / ecart_type
                if abs(z_score) > 2:
                    st.warning(
                        f"⚠️ Anomalie détectée : dernière valeur ({derniere:.2f} L) "
                        f"s'écarte fortement de la moyenne récente ({moyenne:.2f} L, z-score={z_score:.1f})."
                    )
                else:
                    st.success("Production laitière normale (pas d'écart significatif).")
            else:
                st.info("Production stable, pas de variance pour évaluer une anomalie.")
        else:
            st.info("Pas assez de données pour la détection d'anomalies (minimum 5 mesures).")
        st.subheader("Recommandations vaccinales")
        dernier_vaccin_annuel = db.fetchone("""
            SELECT date_vaccin FROM vaccinations 
            WHERE brebis_id=? AND vaccin LIKE '%entéro%' OR vaccin LIKE '%annuel%'
            ORDER BY date_vaccin DESC LIMIT 1
        """, (bid,))
        if dernier_vaccin_annuel:
            date_dernier = datetime.strptime(dernier_vaccin_annuel[0], "%Y-%m-%d")
            if (datetime.now() - date_dernier).days > 365:
                st.warning("⚠️ Le vaccin annuel est à renouveler (plus d'un an).")
            else:
                mois_restants = 12 - ((datetime.now() - date_dernier).days // 30)
                st.info(f"Prochain rappel annuel dans environ {mois_restants} mois.")
        else:
            st.info("Aucun vaccin annuel enregistré. Il est recommandé de vacciner.")
        if age < 1:
            st.info("Les agneaux de moins d'un an doivent être vaccinés contre la pasteurellose.")
    with tab5:
        st.subheader("Exporter l'historique")
        if st.button("Générer le rapport CSV"):
            vaccins_all = db.fetchall("""
                SELECT date_vaccin, vaccin, rappel FROM vaccinations WHERE brebis_id=?
            """, (bid,))
            soins_all = db.fetchall("""
                SELECT date_soin, type, diagnostic, traitement FROM soins WHERE brebis_id=?
            """, (bid,))
            data = []
            for v in vaccins_all:
                data.append({
                    "Date": v[0],
                    "Type": "Vaccin",
                    "Description": v[1],
                    "Rappel": v[2] if v[2] else "",
                    "Détails": ""
                })
            for s in soins_all:
                data.append({
                    "Date": s[0],
                    "Type": s[1],
                    "Description": s[2],
                    "Rappel": "",
                    "Détails": s[3]
                })
            if data:
                df_export = pd.DataFrame(data)
                df_export = df_export.sort_values("Date", ascending=False)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name=f"sante_{numero}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Aucune donnée à exporter.")

# ---- PAGE REPRODUCTION ----
def page_reproduction():
    st.title("🤰 Gestion de la reproduction")
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom, e.nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis_list = db.fetchall(query_brebis, params)
    brebis_dict = {f"{b[0]} - {b[1]} {b[2]} ({b[3]})": b[0] for b in brebis_list}
    if not brebis_dict:
        st.warning("Aucune brebis disponible.")
        return
    selected = st.selectbox("Choisir une brebis", list(brebis_dict.keys()))
    bid = brebis_dict[selected]
    tab1, tab2, tab3 = st.tabs(["🔥 Chaleurs", "🐏 Saillies", "🐑 Mises bas"])
    with tab1:
        st.subheader("Observations des chaleurs / synchronisation")
        with st.form("form_chaleur"):
            date_debut = st.date_input("Date de début", value=datetime.today().date())
            date_fin = st.date_input("Date de fin (optionnelle)", value=None)
            methode = st.selectbox("Méthode", ["Naturelle", "Progestagène", "Autre"])
            obs = st.text_area("Observations")
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO chaleurs (brebis_id, date_debut, date_fin, methode_synchro, observation) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_debut.isoformat(), date_fin.isoformat() if date_fin else None, methode, obs)
                )
                st.success("Chaleurs enregistrées")
                st.rerun()
        chaleurs = db.fetchall(
            "SELECT date_debut, date_fin, methode_synchro, observation FROM chaleurs WHERE brebis_id=? ORDER BY date_debut DESC",
            (bid,)
        )
        if chaleurs:
            df = pd.DataFrame(chaleurs, columns=["Début", "Fin", "Méthode", "Observations"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with tab2:
        st.subheader("Saillies / Inséminations")
        with st.form("form_saillie"):
            date_saillie = st.date_input("Date de saillie", value=datetime.today().date())
            male_id = st.text_input("Identifiant du bélier")
            methode = st.selectbox("Méthode", ["Naturelle", "Insémination artificielle"])
            resultat = st.selectbox("Résultat", ["En attente", "Gestante", "Non gestante"])
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO saillies (brebis_id, date_saillie, male_id, methode, resultat) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_saillie.isoformat(), male_id, methode, resultat)
                )
                st.success("Saillie enregistrée")
                st.rerun()
        saillies = db.fetchall(
            "SELECT date_saillie, male_id, methode, resultat FROM saillies WHERE brebis_id=? ORDER BY date_saillie DESC",
            (bid,)
        )
        if saillies:
            df = pd.DataFrame(saillies, columns=["Date", "Bélier", "Méthode", "Résultat"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            last_gest = db.fetchone(
                "SELECT date_saillie FROM saillies WHERE brebis_id=? AND resultat='Gestante' ORDER BY date_saillie DESC",
                (bid,)
            )
            if last_gest:
                date_saillie = datetime.strptime(last_gest[0], "%Y-%m-%d").date()
                date_mb = date_saillie + timedelta(days=150)
                st.success(f"📅 Mise bas prévue autour du : {date_mb.strftime('%d/%m/%Y')}")
    with tab3:
        st.subheader("Mises bas enregistrées")
        with st.form("form_mb"):
            date_mb = st.date_input("Date de mise bas", value=datetime.today().date())
            nb_agneaux = st.number_input("Nombre d'agneaux", min_value=1, step=1)
            poids_portee = st.number_input("Poids total de la portée (kg)", min_value=0.0, step=0.1)
            remarques = st.text_area("Remarques")
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO mises_bas (brebis_id, date_mise_bas, nb_agneaux, poids_portee, remarques) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_mb.isoformat(), nb_agneaux, poids_portee, remarques)
                )
                st.success("Mise bas enregistrée")
                st.rerun()
        mbas = db.fetchall(
            "SELECT date_mise_bas, nb_agneaux, poids_portee, remarques FROM mises_bas WHERE brebis_id=? ORDER BY date_mise_bas DESC",
            (bid,)
        )
        if mbas:
            df = pd.DataFrame(mbas, columns=["Date", "Agneaux", "Poids portée (kg)", "Remarques"])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ---- PAGE EXPORT ----
def page_export():
    st.title("📤 Export des données")
    st.markdown("Téléchargez l'ensemble de vos données au format CSV ou Excel pour les partager avec votre professeur.")
    format_export = st.radio("Format", ["CSV (dossier compressé)", "Excel (fichier unique)"])
    inclure_photos = st.checkbox("Inclure les photos dans l'archive (pour CSV uniquement)", value=True)
    if st.button("Générer l'export"):
        all_tables = [
            "eleveurs", "elevages", "brebis", 
            "productions", "mesures_morpho", "mesures_mamelles", "composition_corporelle",
            "vaccinations", "soins", "chaleurs", "saillies", "mises_bas",
            "aliments", "rations", "ration_composition",
            "capteurs", "predictions"
        ]
        cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        data_frames = {}
        for table in all_tables:
            if table in existing_tables:
                cursor = db.conn.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()
                columns = [col[1] for col in columns_info]
            else:
                st.warning(f"La table {table} n'existe pas. Elle sera ignorée.")
                data_frames[table] = pd.DataFrame()
                continue
            df_empty = pd.DataFrame(columns=columns)
            try:
                if table == "eleveurs":
                    df_data = pd.read_sql_query(f"SELECT * FROM {table} WHERE user_id=?", db.conn, params=(st.session_state.user_id,))
                elif table == "elevages":
                    df_data = pd.read_sql_query("""
                        SELECT e.* FROM elevages e
                        JOIN eleveurs el ON e.eleveur_id = el.id
                        WHERE el.user_id=?
                    """, db.conn, params=(st.session_state.user_id,))
                elif table == "brebis":
                    df_data = pd.read_sql_query("""
                        SELECT b.* FROM brebis b
                        JOIN elevages e ON b.elevage_id = e.id
                        JOIN eleveurs el ON e.eleveur_id = el.id
                        WHERE el.user_id=?
                    """, db.conn, params=(st.session_state.user_id,))
                elif table in ["productions", "vaccinations", "soins", "chaleurs", "saillies", "mises_bas"]:
                    df_data = pd.read_sql_query(f"""
                        SELECT t.* FROM {table} t
                        JOIN brebis b ON t.brebis_id = b.id
                        JOIN elevages e ON b.elevage_id = e.id
                        JOIN eleveurs el ON e.eleveur_id = el.id
                        WHERE el.user_id=?
                    """, db.conn, params=(st.session_state.user_id,))
                elif table in ["mesures_morpho", "mesures_mamelles", "composition_corporelle"]:
                    df_data = pd.read_sql_query(f"""
                        SELECT t.* FROM {table} t
                        JOIN brebis b ON t.brebis_id = b.id
                        JOIN elevages e ON b.elevage_id = e.id
                        JOIN eleveurs el ON e.eleveur_id = el.id
                        WHERE el.user_id=?
                    """, db.conn, params=(st.session_state.user_id,))
                elif table in ["capteurs", "predictions"]:
                    df_data = pd.read_sql_query(f"""
                        SELECT t.* FROM {table} t
                        JOIN brebis b ON t.brebis_id = b.id
                        JOIN elevages e ON b.elevage_id = e.id
                        JOIN eleveurs el ON e.eleveur_id = el.id
                        WHERE el.user_id=?
                    """, db.conn, params=(st.session_state.user_id,))
                else:
                    df_data = pd.read_sql_query(f"SELECT * FROM {table}", db.conn)
                df_combined = pd.concat([df_empty, df_data], ignore_index=True)
                data_frames[table] = df_combined
            except Exception as e:
                st.error(f"Erreur lors de l'export de la table {table}: {e}")
                data_frames[table] = df_empty
        if format_export.startswith("Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for name, df in data_frames.items():
                    sheet_name = name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            output.seek(0)
            st.download_button(
                label="📥 Télécharger Excel",
                data=output,
                file_name=f"ovin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                for name, df in data_frames.items():
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    zip_file.writestr(f"{name}.csv", csv_data)
                if inclure_photos and os.path.exists(PHOTO_DIR):
                    for root, dirs, files in os.walk(PHOTO_DIR):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zip_file.write(file_path, arcname=os.path.join("photos", file))
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Télécharger ZIP (CSV + photos)",
                data=zip_buffer,
                file_name=f"ovin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

# ---- PAGE ÉLITE ----
def page_elite():
    st.title("🏆 Élite et comparaison")
    params = [st.session_state.user_id]
    query_brebis = """
        SELECT b.id, b.numero_id, b.nom, b.race, b.date_naissance, b.poids_vif,
               e.nom as elevage_nom, el.nom as eleveur_nom
        FROM brebis b
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
    """
    query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
    brebis = db.fetchall(query_brebis, params)
    if not brebis:
        st.warning("Aucune brebis trouvée pour le contexte sélectionné.")
        return
    df = pd.DataFrame(brebis, columns=["id", "numero", "nom", "race", "naissance", "poids", "elevage", "eleveur"])
    prod_moy = []
    for bid in df["id"]:
        prod = db.fetchone("""
            SELECT AVG(quantite) FROM productions 
            WHERE brebis_id=? AND date >= date('now', '-30 days')
        """, (bid,))
        prod_moy.append(prod[0] if prod and prod[0] else 0)
    df["prod_moy (L/j)"] = prod_moy
    score_morpho = []
    for bid in df["id"]:
        score = db.fetchone("""
            SELECT score_global FROM mesures_morpho 
            WHERE brebis_id=? ORDER BY date_mesure DESC LIMIT 1
        """, (bid,))
        score_morpho.append(score[0] if score else 0)
    df["score_morpho"] = score_morpho
    df["viande_estimee (kg)"] = df["poids"] * 0.45
    rendement = []
    for bid in df["id"]:
        comp = db.fetchone("""
            SELECT rendement_carcasse FROM composition_corporelle 
            WHERE brebis_id=? ORDER BY date_estimation DESC LIMIT 1
        """, (bid,))
        rendement.append(comp[0] if comp else None)
    df["rendement (%)"] = rendement
    st.subheader("📊 Tableau des brebis")
    colonnes_affichees = ["numero", "nom", "eleveur", "elevage", "race", "poids", "prod_moy (L/j)", "score_morpho", "viande_estimee (kg)", "rendement (%)"]
    st.dataframe(df[colonnes_affichees].round(2))
    st.subheader("🏆 Classement")
    critere = st.selectbox("Critère de classement", 
                           ["prod_moy (L/j)", "score_morpho", "viande_estimee (kg)", "poids", "rendement (%)"])
    top_n = st.slider("Nombre de brebis à afficher", 5, 50, 10)
    ascending = st.checkbox("Ordre croissant", False)
    df[critere] = pd.to_numeric(df[critere], errors='coerce')
    df_class = df[df[critere].notna()].copy()
    if df_class.empty:
        st.warning(f"Aucune valeur numérique valide pour le critère {critere}.")
    else:
        if ascending:
            top = df_class.nsmallest(top_n, critere)
        else:
            top = df_class.nlargest(top_n, critere)
        st.dataframe(top[["numero", "nom", "eleveur", "elevage", critere]].round(2))
        fig = px.bar(top, x="nom", y=critere, color="eleveur", title=f"Top {top_n} - {critere}")
        st.plotly_chart(fig, use_container_width=True)
    if st.session_state.eleveur_id is None and len(df["eleveur"].unique()) > 1:
        st.subheader("📈 Comparaison par éleveur")
        numeric_cols = ["prod_moy (L/j)", "score_morpho", "poids", "viande_estimee (kg)", "rendement (%)"]
        df_eleveur = df.groupby("eleveur")[numeric_cols].mean().reset_index()
        for col in numeric_cols:
            df_eleveur[col] = pd.to_numeric(df_eleveur[col], errors='coerce').fillna(0)
        st.dataframe(df_eleveur.round(2))
        fig2 = px.bar(df_eleveur, x="eleveur", y=["prod_moy (L/j)", "score_morpho", "rendement (%)"], 
                     barmode="group", title="Performances moyennes par éleveur")
        st.plotly_chart(fig2, use_container_width=True)

# ---- PAGE IA ----
def page_ia():
    st.title("🧠 Intelligence Artificielle & Data Mining")
    st.markdown("Analyses avancées basées sur les données de votre élevage.")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prédiction laitière avancée",
        "🔍 Détection d'anomalies",
        "📊 Clustering des brebis",
        "📂 Analyse exploratoire (import)"
    ])
    with tab1:
        st.subheader("Prédiction de production laitière par modèle ML")
        model_path = os.path.join(MODEL_DIR, 'lait_model.pkl')
        if os.path.exists(model_path):
            st.success("Un modèle ML est disponible.")
            params = [st.session_state.user_id]
            query_brebis = """
                SELECT b.id, b.numero_id, b.nom, e.nom
                FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                JOIN eleveurs el ON e.eleveur_id = el.id
                WHERE el.user_id=?
            """
            query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
            brebis_list = db.fetchall(query_brebis, params)
            brebis_dict = {f"{b[0]} - {b[1]} {b[2]} ({b[3]})": b[0] for b in brebis_list}
            if brebis_dict:
                selected = st.selectbox("Choisir une brebis", list(brebis_dict.keys()), key="ia_brebis")
                bid = brebis_dict[selected]
                if st.button("Prédire avec ML"):
                    pred = predict_lait_ml(bid)
                    if pred is not None:
                        st.metric("Production prédite (L/j)", f"{pred:.2f}")
                    else:
                        st.warning("Impossible de faire la prédiction (données manquantes).")
            else:
                st.warning("Aucune brebis disponible.")
        else:
            st.info("Aucun modèle ML entraîné. Vous pouvez en entraîner un si vous avez suffisamment de données de production.")
            if st.button("Entraîner un modèle ML"):
                with st.spinner("Entraînement en cours..."):
                    result = train_lait_model()
                    if result is None:
                        st.error("Pas assez de données (minimum 20 brebis avec productions).")
                    else:
                        model, score = result
                        st.success(f"Modèle entraîné avec un score R² de {score:.2f} sur le test.")
    with tab2:
        st.subheader("Détection d'anomalies (Isolation Forest)")
        params = [st.session_state.user_id]
        query_brebis = """
            SELECT b.id, b.numero_id, b.nom, b.poids_vif,
                   AVG(p.quantite) as prod_moy,
                   AVG(m.score_global) as score_morpho
            FROM brebis b
            LEFT JOIN productions p ON b.id = p.brebis_id AND p.date >= date('now', '-30 days')
            LEFT JOIN mesures_morpho m ON b.id = m.brebis_id
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
            GROUP BY b.id
        """
        query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
        df = pd.read_sql_query(query_brebis, db.conn, params=params)
        if df.empty:
            st.warning("Aucune donnée disponible.")
        else:
            df['viande_estimee'] = df['poids_vif'] * 0.45
            df['prod_moy'] = df['prod_moy'].fillna(0)
            df['score_morpho'] = df['score_morpho'].fillna(0)
            features = ['prod_moy', 'score_morpho', 'poids_vif', 'viande_estimee']
            X = df[features].fillna(0)
            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(X)
            df['anomalie'] = preds
            anomalies = df[df['anomalie'] == -1]
            st.write(f"**{len(anomalies)}** brebis potentiellement anormales détectées.")
            if not anomalies.empty:
                st.dataframe(anomalies[['numero_id', 'nom', 'prod_moy', 'score_morpho', 'poids_vif']])
            else:
                st.success("Aucune anomalie détectée.")
    with tab3:
        st.subheader("Clustering des brebis (K-Means)")
        params = [st.session_state.user_id]
        query_brebis = """
            SELECT b.id, b.numero_id, b.nom, b.poids_vif,
                   AVG(p.quantite) as prod_moy,
                   AVG(m.score_global) as score_morpho
            FROM brebis b
            LEFT JOIN productions p ON b.id = p.brebis_id AND p.date >= date('now', '-30 days')
            LEFT JOIN mesures_morpho m ON b.id = m.brebis_id
            JOIN elevages e ON b.elevage_id = e.id
            JOIN eleveurs el ON e.eleveur_id = el.id
            WHERE el.user_id=?
            GROUP BY b.id
        """
        query_brebis, params = filtrer_par_eleveur(query_brebis, params, join_eleveur=True)
        df = pd.read_sql_query(query_brebis, db.conn, params=params)
        if df.empty:
            st.warning("Aucune donnée disponible pour le clustering.")
        else:
            df['viande_estimee'] = df['poids_vif'] * 0.45
            df['prod_moy'] = df['prod_moy'].fillna(0)
            df['score_morpho'] = df['score_morpho'].fillna(0)
            n_brebis = len(df)
            max_clusters = min(5, n_brebis)
            if max_clusters < 2:
                st.warning(f"Pas assez de brebis ({n_brebis}) pour effectuer un clustering (minimum 2).")
            else:
                n_clusters = st.slider("Nombre de clusters", 2, max_clusters, min(3, max_clusters))
                features = ['prod_moy', 'score_morpho', 'poids_vif', 'viande_estimee']
                X = df[features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                df['cluster'] = clusters
                fig = px.scatter_3d(df, x='prod_moy', y='score_morpho', z='poids_vif', color='cluster',
                                     hover_data=['numero_id', 'nom'], title="Clusters des brebis")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.groupby('cluster')[features].mean().round(2))
    with tab4:
        st.subheader("Analyse exploratoire d'un fichier externe")
        uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success("Fichier chargé avec succès.")
                st.dataframe(df.head())
                if profiling_available:
                    analyse_mode = st.radio("Type d'analyse", ["Statistiques descriptives", "Rapport complet (ydata-profiling)"])
                else:
                    st.info("Module ydata-profiling non installé. Utilisation des statistiques descriptives.")
                    analyse_mode = "Statistiques descriptives"
                if analyse_mode == "Statistiques descriptives":
                    st.subheader("Statistiques descriptives")
                    st.dataframe(df.describe(include='all').transpose())
                    st.subheader("Informations sur les colonnes")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                else:
                    if profiling_available:
                        if st.button("Générer le rapport d'analyse"):
                            with st.spinner("Génération du rapport..."):
                                profile = ProfileReport(df, title="Rapport d'analyse", explorative=True)
                                st_profile_report(profile)
                    else:
                        st.warning("Le module ydata-profiling n'est pas disponible.")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

# ---- PAGE APPRENTISSAGE ----
def page_apprentissage():
    st.title("🧠 Apprentissage automatique")
    st.markdown("Cette page permet d'entraîner un modèle de deep learning pour la détection automatique des points anatomiques.")
    nb_images = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.npz')])
    st.write(f"Images collectées pour l'entraînement : **{nb_images}**")
    if nb_images < 10:
        st.warning("Il faut au moins 10 images pour un premier entraînement significatif. Continuez à utiliser la photogrammétrie et à contribuer.")
    else:
        if st.button("🚀 Lancer l'entraînement"):
            with st.spinner("Entraînement en cours... (cela peut prendre plusieurs minutes)"):
                model, history = entrainer_modele()
                if model is None:
                    st.error(history)
                else:
                    st.success("Entraînement terminé ! Modèle sauvegardé dans models/keypoints_model_custom.h5")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
                    ax1.plot(history.history['loss'], label='Train')
                    ax1.plot(history.history['val_loss'], label='Validation')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss (MSE)')
                    ax1.legend()
                    ax2.plot(history.history['mae'], label='Train')
                    ax2.plot(history.history['val_mae'], label='Validation')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('MAE')
                    ax2.legend()
                    st.pyplot(fig)
    st.subheader("Tester le modèle")
    custom_model_path = os.path.join(MODEL_DIR, 'keypoints_model_custom.h5')
    if os.path.exists(custom_model_path):
        model = tf.keras.models.load_model(custom_model_path)
        uploaded_test = st.file_uploader("Choisir une image de test", type=['jpg','png','jpeg'])
        if uploaded_test is not None:
            img_pil = Image.open(uploaded_test)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_cv, (256, 256))
            img_input = np.expand_dims(img_resized / 255.0, axis=0)
            pred = model.predict(img_input)[0]
            h, w = img_cv.shape[:2]
            facteur_x = w / 256
            facteur_y = h / 256
            points = {
                'garrot': (int(pred[0]*facteur_x), int(pred[1]*facteur_y)),
                'epaule': (int(pred[2]*facteur_x), int(pred[3]*facteur_y)),
                'fesse': (int(pred[4]*facteur_x), int(pred[5]*facteur_y))
            }
            img_copy = img_cv.copy()
            for pt in points.values():
                cv2.circle(img_copy, pt, 5, (0,255,0), -1)
            st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), caption="Points prédits")
    else:
        st.info("Aucun modèle personnalisé entraîné. Utilisez la photogrammétrie pour collecter des données et entraîner le modèle.")

# ---- PAGE IoT ----
def page_iot():
    st.title("📡 Import de données IoT (capteurs)")
    st.markdown("Chargez un fichier CSV contenant des données de capteurs (température, activité, rythme cardiaque).")
    st.info("Format attendu : colonnes 'brebis_id', 'date', 'temperature', 'activite', 'rythme_cardiaque'.")
    uploaded = st.file_uploader("Choisir un fichier CSV", type='csv')
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required = ['brebis_id', 'date', 'temperature', 'activite', 'rythme_cardiaque']
            if not all(col in df.columns for col in required):
                st.error(f"Colonnes requises : {required}")
                return
            df['date'] = pd.to_datetime(df['date'])
            inserted = 0
            for _, row in df.iterrows():
                try:
                    db.execute("""
                        INSERT INTO capteurs (brebis_id, date, temperature, activite, rythme_cardiaque)
                        VALUES (?, ?, ?, ?, ?)
                    """, (int(row['brebis_id']), row['date'].isoformat(), 
                          float(row['temperature']), float(row['activite']), float(row['rythme_cardiaque'])))
                    inserted += 1
                except:
                    pass
            st.success(f"{inserted} lignes importées.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur: {e}")

# ---- PAGE VALIDATION ----
def page_validation():
    st.title("📊 Validation des modèles")
    st.markdown("Compare les prédictions IA avec les mesures réelles.")
    params = [st.session_state.user_id]
    query = """
        SELECT p.variable, AVG(ABS(p.valeur_predite - p.valeur_reelle)) as erreur_moyenne,
               COUNT(*) as nb
        FROM predictions p
        JOIN brebis b ON p.brebis_id = b.id
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=?
        GROUP BY p.variable
    """
    query, params = filtrer_par_eleveur(query, params, join_eleveur=True)
    df = pd.read_sql_query(query, db.conn, params=params)
    if df.empty:
        st.info("Aucune donnée de validation disponible.")
        return
    st.dataframe(df, use_container_width=True)
    fig = px.bar(df, x='variable', y='erreur_moyenne', color='variable',
                 title="Erreur moyenne par variable")
    st.plotly_chart(fig, use_container_width=True)
    var = st.selectbox("Choisir une variable", df['variable'].unique())
    query2 = """
        SELECT date_prediction, valeur_predite, valeur_reelle
        FROM predictions p
        JOIN brebis b ON p.brebis_id = b.id
        JOIN elevages e ON b.elevage_id = e.id
        JOIN eleveurs el ON e.eleveur_id = el.id
        WHERE el.user_id=? AND p.variable=?
        ORDER BY date_prediction
    """
    params2 = [st.session_state.user_id, var]
    query2, params2 = filtrer_par_eleveur(query2, params2, join_eleveur=True)
    df2 = pd.read_sql_query(query2, db.conn, params=params2)
    if not df2.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df2['date_prediction'], y=df2['valeur_predite'],
                                  mode='lines+markers', name='Prédit'))
        fig2.add_trace(go.Scatter(x=df2['date_prediction'], y=df2['valeur_reelle'],
                                  mode='lines+markers', name='Réel'))
        fig2.update_layout(title=f"Évolution {var} – Prédit vs Réel")
        st.plotly_chart(fig2, use_container_width=True)

# ---- FONCTION ENTRAINEMENT MODÈLE CUSTOM (pour clés points) ----
def entrainer_modele():
    # Fonction similaire à celle déjà présente, mais sauvegarde sous keypoints_model_custom.h5
    X, y, _ = charger_dataset()
    if X is None or len(X) < 10:
        return None, "Pas assez de données (minimum 10 images)."
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(6)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
    model.save(os.path.join(MODEL_DIR, 'keypoints_model_custom.h5'))
    return model, history

def charger_dataset():
    images = []
    points_list = []
    metadatas = []
    for f in os.listdir(DATASET_DIR):
        if f.endswith('.npz'):
            data = np.load(os.path.join(DATASET_DIR, f), allow_pickle=True)
            images.append(data['image'])
            pts = data['points_norm'].item()
            vec = [pts['garrot'][0], pts['garrot'][1],
                   pts['epaule'][0], pts['epaule'][1],
                   pts['fesse'][0], pts['fesse'][1]]
            points_list.append(vec)
            metadatas.append(data['metadata'].item())
    if len(images) == 0:
        return None, None, None
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(points_list, dtype=np.float32)
    return X, y, metadatas

# -----------------------------------------------------------------------------
# SIDEBAR ET MAIN
# -----------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/sheep.png", width=80)
        st.title(f"🐑 {Config.APP_NAME}")
        st.caption(f"**{Config.LABORATOIRE}** v{Config.VERSION}")
        st.divider()
        if st.session_state.user_id:
            eleveurs = db.fetchall(
                "SELECT id, nom FROM eleveurs WHERE user_id=? ORDER BY nom",
                (st.session_state.user_id,)
            )
            eleveurs_options = {"Tous les éleveurs": None}
            eleveurs_options.update({f"{e[1]} (ID {e[0]})": e[0] for e in eleveurs})
            current = st.session_state.get("eleveur_id", None)
            default_index = 0
            for i, (label, eid) in enumerate(eleveurs_options.items()):
                if eid == current:
                    default_index = i
                    break
            selected_label = st.selectbox(
                "👨‍🌾 Éleveur actif",
                options=list(eleveurs_options.keys()),
                index=default_index,
                key="eleveur_selector"
            )
            st.session_state.eleveur_id = eleveurs_options[selected_label]
            st.divider()
            menu = st.radio(
                "Navigation",
                ["📊 Tableau de bord", 
                 "🐑 Gestion élevage",
                 "🧬 Génomique NCBI", 
                 "🥩 Composition", 
                 "📸 Photogrammétrie auto", 
                 "🔮 Prédictions", 
                 "🌾 Nutrition avancée",
                 "🥛 Production laitière",
                 "🧬 Génomique avancée",
                 "🏥 Santé",
                 "🤰 Reproduction",
                 "📤 Export données",
                 "🏆 Élite et comparaison",
                 "🧠 IA & Data Mining",
                 "🧠 Apprentissage automatique",
                 "📡 IoT",
                 "📊 Validation",
                 "🚪 Déconnexion"],
                label_visibility="collapsed"
            )
            st.divider()
            if st.button("💾 Sauvegarde rapide", use_container_width=True):
                st.download_button(
                    label="Télécharger JSON",
                    data=json.dumps({"user_id": st.session_state.user_id, "date": datetime.now().isoformat()}),
                    file_name=f"ovin_backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            page_map = {
                "📊 Tableau de bord": "dashboard",
                "🐑 Gestion élevage": "gestion_elevage",
                "🧬 Génomique NCBI": "genomique",
                "🥩 Composition": "composition",
                "📸 Photogrammétrie auto": "analyse_auto",
                "🔮 Prédictions": "prediction",
                "🌾 Nutrition avancée": "nutrition_avancee",
                "🥛 Production laitière": "production",
                "🧬 Génomique avancée": "genomique_avancee",
                "🏥 Santé": "sante",
                "🤰 Reproduction": "reproduction",
                "📤 Export données": "export",
                "🏆 Élite et comparaison": "elite",
                "🧠 IA & Data Mining": "ia",
                "🧠 Apprentissage automatique": "apprentissage",
                "📡 IoT": "iot",
                "📊 Validation": "validation",
                "🚪 Déconnexion": "logout"
            }
            selected_page = page_map.get(menu, "dashboard")
            if selected_page == "logout":
                st.session_state.user_id = None
                st.session_state.current_page = "login"
                st.rerun()
            elif selected_page != st.session_state.current_page:
                st.session_state.current_page = selected_page
                st.rerun()

def main():
    sidebar()
    if st.session_state.current_page == "login":
        page_login()
    elif st.session_state.current_page == "dashboard":
        page_dashboard()
    elif st.session_state.current_page == "genomique":
        page_genomique()
    elif st.session_state.current_page == "composition":
        page_composition()
    elif st.session_state.current_page == "analyse_auto":
        page_analyse_auto()
    elif st.session_state.current_page == "prediction":
        page_prediction()
    elif st.session_state.current_page == "nutrition_avancee":
        page_nutrition_avancee()
    elif st.session_state.current_page == "production":
        page_production()
    elif st.session_state.current_page == "genomique_avancee":
        page_genomique_avancee()
    elif st.session_state.current_page == "gestion_elevage":
        page_gestion_elevage()
    elif st.session_state.current_page == "sante":
        page_sante()
    elif st.session_state.current_page == "reproduction":
        page_reproduction()
    elif st.session_state.current_page == "export":
        page_export()
    elif st.session_state.current_page == "elite":
        page_elite()
    elif st.session_state.current_page == "ia":
        page_ia()
    elif st.session_state.current_page == "apprentissage":
        page_apprentissage()
    elif st.session_state.current_page == "iot":
        page_iot()
    elif st.session_state.current_page == "validation":
        page_validation()

# -----------------------------------------------------------------------------
# POINT D'ENTRÉE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    db = get_database()
    genomic_analyzer = GenomicAnalyzer()
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.current_page = "login"
        st.session_state.eleveur_id = None
    st.set_page_config(
        page_title="Ovin Manager Pro",
        page_icon="🐑",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .gene-card {
            background-color: #e3f2fd;
            border-left: 5px solid #00838F;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .meat-card {
            background-color: #fff3e0;
            border-left: 5px solid #FF6F00;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    main()
