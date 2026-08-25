# -----------------------------------------------------------------------------
# OVIN MANAGER PRO - Version 9.0 (ComplÃ¨te & AutomatisÃ©e)
# Laboratoir - UniversitÃ© Laval
# ConÃ§u pour rÃ©pondre aux dÃ©fis du Pr. Ã‰ric R. Paquet
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

# Outil pour les coordonnÃ©es de clics
from streamlit_image_coordinates import streamlit_image_coordinates

# Deep learning (optionnel : TensorFlow n'a pas toujours de wheel disponible
# pour la version de Python utilisÃ©e par la plateforme d'hÃ©bergement. L'app
# fonctionne sans lui grÃ¢ce au repli automatique sur MediaPipe pour la
# dÃ©tection de points anatomiques ; seule la page "Apprentissage" (entraÃ®nement
# d'un modÃ¨le custom) est dÃ©sactivÃ©e si tensorflow est absent.)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
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
# qui n'est plus incluse dans les distributions rÃ©centes de mediapipe)
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_pose_landmarker = None

def _ensure_pose_model() -> bool:
    """TÃ©lÃ©charge le modÃ¨le de dÃ©tection de pose s'il n'est pas dÃ©jÃ  prÃ©sent localement."""
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
    """Retourne une instance (singleton, crÃ©Ã©e Ã  la demande) du dÃ©tecteur de pose MediaPipe."""
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
        "baton_1m": {"nom": "BÃ¢ton 1m", "largeur": 1000, "hauteur": None},
        "a4": {"nom": "Feuille A4", "largeur": 210, "hauteur": 297},
        "carte": {"nom": "Carte bancaire", "largeur": 85.6, "hauteur": 53.98},
        "piece_100da": {"nom": "PiÃ¨ce 100 DA", "diametre": 29.5}
    }
    
    RACES = {
        "Hamra": {"origine": "Atlas saharien", "aptitude": "Mixte", "genes": ["BMP15", "GDF9"]},
        "Ouled Djellal": {"origine": "Steppes algÃ©riennes", "aptitude": "Viande", "genes": ["MSTN", "IGF2"]},
        "Sidahou": {"origine": "AurÃ¨s", "aptitude": "Lait", "genes": ["LALBA", "CSN3", "DGAT1"]},
        "Rembi": {"origine": "Tell", "aptitude": "Mixte", "genes": ["BMP15", "LALBA"]},
        "Autre": {"origine": "Inconnue", "aptitude": "Variable", "genes": []}
    }
    
    GENES_ECONOMIQUES = {
        "BMP15": {"nom": "Bone Morphogenetic Protein 15", "chr": "X", "effet": "FÃ©conditÃ©"},
        "GDF9": {"nom": "Growth Differentiation Factor 9", "chr": "5", "effet": "FÃ©conditÃ©"},
        "BMPR1B": {"nom": "BMP Receptor 1B", "chr": "6", "effet": "ProlificitÃ© (Booroola)"},
        "MSTN": {"nom": "Myostatin", "chr": "2", "effet": "Hypertrophie musculaire"},
        "IGF2": {"nom": "Insulin-like Growth Factor 2", "chr": "2", "effet": "Croissance"},
        "GH": {"nom": "Growth Hormone", "chr": "19", "effet": "Croissance"},
        "GHR": {"nom": "Growth Hormone Receptor", "chr": "16", "effet": "EfficacitÃ© alimentaire"},
        "LALBA": {"nom": "Alpha-Lactalbumin", "chr": "3", "effet": "ProtÃ©ines lait"},
        "CSN3": {"nom": "Kappa-Casein", "chr": "6", "effet": "QualitÃ© fromagÃ¨re"},
        "DGAT1": {"nom": "Diacylglycerol Acyltransferase 1", "chr": "14", "effet": "MatiÃ¨re grasse lait"},
        "SCD": {"nom": "Stearoyl-CoA Desaturase", "chr": "22", "effet": "Acides gras insaturÃ©s"},
        "TLR4": {"nom": "Toll-like Receptor 4", "chr": "1", "effet": "RÃ©sistance infections"},
        "MHC": {"nom": "Major Histocompatibility Complex", "chr": "20", "effet": "ImmunitÃ©"},
        "PRNP": {"nom": "Prion Protein", "chr": "13", "effet": "RÃ©sistance tremblante"},
        "CAST": {"nom": "Calpastatin", "chr": "7", "effet": "TendretÃ© viande"},
        "CAPN1": {"nom": "Calpain 1", "chr": "16", "effet": "TendretÃ© viande"},
        "FABP4": {"nom": "Fatty Acid Binding Protein 4", "chr": "8", "effet": "Marbling (gras intramusculaire)"}
    }
    
    ETATS_PHYSIO = [
        "Jeune", "Gestation dÃ©but", "Gestation fin",
        "Lactation dÃ©but", "Lactation milieu", "Lactation fin",
        "Tarie", "Engraissement"
    ]

# Seuils de qualitÃ© image
QUALITY_BLUR_THRESHOLD = 100.0
QUALITY_CONTRAST_THRESHOLD = 30

# -----------------------------------------------------------------------------
# BASE DE DONNÃ‰ES
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
        
        # Tables existantes (inchangÃ©es)
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
            ("Orge", "ConcentrÃ©", 1.1, 80, 86, 25),
            ("MaÃ¯s", "ConcentrÃ©", 1.3, 70, 86, 30),
            ("Son de blÃ©", "ConcentrÃ©", 0.9, 120, 87, 18),
            ("Tourteau de soja", "ConcentrÃ©", 1.2, 400, 88, 45),
            ("Foin de luzerne", "Fourrage", 0.6, 120, 85, 15),
            ("Foin d'avoine", "Fourrage", 0.5, 70, 85, 12),
            ("Paille", "Fourrage", 0.3, 20, 88, 5),
            ("CMV", "MinÃ©ral", 0, 0, 100, 80)
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
# FONCTIONS UTILITAIRES (sauvegarde, filtrage, dÃ©tection)
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
# FONCTIONS QUALITÃ‰ IMAGE ET DÃ‰TECTION
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
        st.warning("ModÃ¨le de dÃ©tection de pose indisponible (tÃ©lÃ©chargement impossible). VÃ©rifiez la connexion internet.")
        return None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    results = landmarker.detect(mp_image)
    if not results.pose_landmarks:
        return None
    h, w, _ = image_rgb.shape
    landmarks = results.pose_landmarks[0]
    pts = {}
    # Garrot : entre les Ã©paules (indices 11 et 12 pour humain)
    x = (landmarks[11].x + landmarks[12].x) / 2 * w
    y = (landmarks[11].y + landmarks[12].y) / 2 * h
    pts['garrot'] = (int(x), int(y))
    # Ã‰paule : Ã©paule gauche
    pts['epaule'] = (int(landmarks[11].x * w), int(landmarks[11].y * h))
    # Fesse : hanche gauche (indice 23)
    pts['fesse'] = (int(landmarks[23].x * w), int(landmarks[23].y * h))
    return pts

def detect_keypoints_hybrid(image_rgb, custom_model_path=None):
    """
    DÃ©tection de points clÃ©s : utilise un modÃ¨le custom TensorFlow si prÃ©sent,
    sinon MediaPipe.
    """
    if tensorflow_available and custom_model_path and os.path.exists(custom_model_path):
        try:
            model = tf.keras.models.load_model(custom_model_path)
            img_resized = cv2.resize(image_rgb, (256, 256))
            img_input = np.expand_dims(img_resized / 255.0, axis=0)
            pred = model.predict(img_input, verbose=0)[0]
            h, w, _ = image_rgb.shape
            # Les coordonnÃ©es prÃ©dites sont normalisÃ©es (0-1) pour 6 valeurs : garrot(x,y), epaule(x,y), fesse(x,y)
            pts = {
                'garrot': (int(pred[0]*w), int(pred[1]*h)),
                'epaule': (int(pred[2]*w), int(pred[3]*h)),
                'fesse': (int(pred[4]*w), int(pred[5]*h))
            }
            return pts
        except Exception as e:
            st.warning(f"Erreur chargement modÃ¨le custom: {e}. Utilisation de MediaPipe.")
    # Fallback MediaPipe
    return detect_keypoints_mediapipe(image_rgb)

def analyze_mamelle(image):
    """
    Analyse une image de mamelle pour dÃ©tecter rougeur et asymÃ©trie.
    Retourne un rapport.
    """
    h, w = image.shape[:2]
    # DÃ©tection de rouge en HSV
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

    # AsymÃ©trie : diviser en deux verticalement, comparer luminositÃ©
    left_half = image[:, :w//2]
    right_half = image[:, w//2:]
    left_mean = np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
    right_mean = np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))
    asym_ratio = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2 + 1e-6)

    alerts = []
    score = 0
    if red_ratio > 0.03:
        alerts.append("âš ï¸ Rougeur dÃ©tectÃ©e (possible inflammation)")
        score += 2
    if asym_ratio > 0.2:
        alerts.append("âš ï¸ AsymÃ©trie importante (possible Å“dÃ¨me)")
        score += 2
    if score >= 3:
        level = "Ã‰levÃ©"
    elif score >= 1:
        level = "ModÃ©rÃ©"
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
        # Score : contraste / flou (Ã©vite division par zÃ©ro)
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

# DÃ©tection Ã©talons
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
# CLASSES MÃ‰TIER
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
        if symetrie == "SymÃ©trique": score += 0.5
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
            "niveau": "Ã‰lite" if base > 1.5 else "Bon" if base > 1.0 else "Standard"
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
            st.error(f"Erreur dÃ©tails gÃ¨nes: {e}")
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
            results["recommandations"].append("âœ… Excellente valeur reproductive")
        if results["score_croissance"] > 70:
            results["recommandations"].append("âœ… Excellente conformation viande")
        if results["score_lait"] > 70:
            results["recommandations"].append("âœ… Excellent potentiel laitier")
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
    st.markdown('<p class="main-header">ðŸ‘ Ovin Manager Pro</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Laboratoire {Config.LABORATOIRE} - SystÃ¨me Expert de GÃ©nÃ©tique Ovine</p>', unsafe_allow_html=True)
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
            if st.button("CrÃ©er compte", use_container_width=True):
                if new_pass != confirm_pass:
                    st.error("Les mots de passe ne correspondent pas")
                elif not new_user or not new_pass:
                    st.error("Remplissez tous les champs")
                else:
                    try:
                        db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                                  (new_user, OvinScience.hash_password(new_pass)))
                        st.success("Compte crÃ©Ã© ! Connectez-vous")
                    except:
                        st.error("Nom d'utilisateur dÃ©jÃ  pris")

# ---- PAGE DASHBOARD ----
def page_dashboard():
    st.title(f"ðŸ“Š Tableau de Bord - {Config.LABORATOIRE}")
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
        ("ðŸ‘¨â€ðŸŒ¾ Ã‰leveurs", dash_stats[0], Config.VERT),
        ("ðŸ‘ Brebis", dash_stats[1], Config.BLEU),
        ("ðŸ§¬ Analyses", dash_stats[2], Config.CYAN),
        ("ðŸ“ˆ DonnÃ©es", dash_stats[0] + dash_stats[1] + dash_stats[2], Config.ORANGE)
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
    st.subheader("ðŸš€ Modules GÃ©nomiques & Analytiques")
    modules = [
        ("ðŸ§¬ Analyse NCBI/GenBank", "Recherche gÃ¨nes, SNPs, BLAST", "genomique", Config.CYAN),
        ("ðŸ¥© Composition Corporelle", "Estimation viande/graisse/os", "composition", Config.ORANGE),
        ("ðŸ“¸ PhotogrammÃ©trie auto", "Capture vidÃ©o, IA, filtrage qualitÃ©", "analyse_auto", Config.VERT),
        ("ðŸ¥› PrÃ©diction Lait", "ML potentiel laitier", "prediction", Config.VIOLET),
        ("ðŸŒ¾ Nutrition", "Formulation rations", "nutrition_avancee", Config.BLEU),
        ("ðŸ§  IA & Data Mining", "Analyses avancÃ©es, clustering, anomalies", "ia", Config.ROUGE),
        ("ðŸ“¡ IoT & Capteurs", "Import donnÃ©es capteurs", "iot", Config.CYAN),
        ("ðŸ“Š Validation", "Comparaison prÃ©dictions vs rÃ©el", "validation", Config.ORANGE),
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
                if st.button("Ouvrir â†’", key=f"btn_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()

# ---- PAGE GÃ‰NOMIQUE NCBI ----
def page_genomique():
    st.title("ðŸ§¬ Analyse GÃ©nomique - NCBI/GenBank")
    tab1, tab2, tab3 = st.tabs(["ðŸ” Recherche GÃ¨ne", "ðŸ† Profil Race", "ðŸ§ª SNPs/QTN"])
    with tab1:
        st.subheader("Recherche dans NCBI Gene")
        col1, col2 = st.columns([2, 1])
        with col1:
            gene_search = st.text_input("Nom du gÃ¨ne", "BMP15")
        with col2:
            organism_label = st.selectbox("Organisme", ["Ovis aries (Mouton)", "Capra hircus (ChÃ¨vre)", "Bos taurus (Bovin)"])
            organism = organism_label.split(" (")[0]
        if st.button("ðŸ” Rechercher dans NCBI", use_container_width=True):
            results = genomic_analyzer.ncbi.search_gene(gene_search, organism)
            if results:
                for gene in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="gene-card">
                            <h4>ðŸ§¬ {gene['name']} (ID: {gene['gene_id']})</h4>
                            <p><strong>Description:</strong> {gene['description']}</p>
                            <p><strong>Chromosome:</strong> {gene['chromosome']} | <strong>Position:</strong> {gene['map_location']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        local_info = Config.GENES_ECONOMIQUES.get(gene_search.upper())
                        if local_info:
                            st.info(f"**Effet Ã©conomique:** {local_info['effet']}")
            else:
                local = Config.GENES_ECONOMIQUES.get(gene_search.upper())
                if local:
                    st.success("Informations depuis la base locale GenApAgiE")
                    st.json(local)
                else:
                    st.warning("GÃ¨ne non trouvÃ©. Essayez: BMP15, MSTN, DGAT1, CAST, CAPN1...")
    with tab2:
        st.subheader("Profil GÃ©nÃ©tique par Race")
        race_selected = st.selectbox("SÃ©lectionner une race", list(Config.RACES.keys()))
        if st.button("ðŸ§¬ Analyser le profil gÃ©nÃ©tique"):
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
                title=f"Profil GÃ©nÃ©tique: {race_selected}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("GÃ¨nes Majeurs")
            for gene in analysis['genes']:
                with st.expander(f"ðŸ§¬ {gene['symbole']} - {gene['nom'][:40]}..."):
                    st.write(f"**Effet:** {gene['effet']}")
                    st.write(f"**Chromosome:** {gene['chromosome']}")
            if analysis['recommandations']:
                st.success("### âœ… Recommandations")
                for rec in analysis['recommandations']:
                    st.write(rec)
    with tab3:
        st.subheader("Base de donnÃ©es SNPs et QTN Ã©conomiques")
        categorie = st.selectbox("Filtrer par catÃ©gorie", 
                                ["Tous", "Reproduction", "Croissance/Viande", "Lait", "RÃ©sistance", "QualitÃ© viande"])
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
            elif categorie == "RÃ©sistance" and any(x in sym for x in ["TLR", "MHC", "PRNP"]):
                genes_filtres.append((sym, info))
            elif categorie == "QualitÃ© viande" and any(x in sym for x in ["CAST", "CAPN", "FABP"]):
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
        gene_detail = st.selectbox("Voir dÃ©tails", [sym for sym, _ in genes_filtres])
        if gene_detail:
            info = Config.GENES_ECONOMIQUES[gene_detail]
            st.json(info)

# ---- PAGE COMPOSITION ----
def page_composition():
    st.title("ðŸ¥© Composition Corporelle EstimÃ©e")
    st.markdown("Estimation dÃ©taillÃ©e de la rÃ©partition viande/graisse/os basÃ©e sur les Ã©quations zootechniques")
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
    brebis_options["Saisie manuelle (animal non enregistrÃ©)"] = None
    mode = st.radio("Mode de saisie", ["SÃ©lectionner une brebis existante", "Saisie manuelle"])
    if mode == "SÃ©lectionner une brebis existante":
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
    if st.button("ðŸ§® Calculer la composition", use_container_width=True):
        comp = OvinScience.estimer_composition(poids_vif, race, cc)
        if "erreur" in comp:
            st.error(comp["erreur"])
            return
        st.subheader("ðŸ“Š RÃ©sultats")
        cols = st.columns(4)
        metrics = [
            ("ðŸ¥© Viande", comp['viande']['kg'], comp['viande']['pct'], Config.VERT),
            ("ðŸ¥“ Graisse", comp['graisse']['kg'], comp['graisse']['pct'], Config.ORANGE),
            ("ðŸ¦´ Os", comp['os']['kg'], comp['os']['pct'], "grey"),
            ("ðŸ“¦ Carcasse", comp['poids_carcasse'], comp['rendement'], Config.BLEU)
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
        with st.expander("ðŸ”ª DÃ©tails des dÃ©coupes"):
            decoupes_data = {
                "DÃ©coupe": ["Gigot", "Ã‰paule", "CÃ´telettes", "Poitrine"],
                "Poids (kg)": [comp['decoupes']['gigot'], comp['decoupes']['epaule'],
                              comp['decoupes']['cotelette'], comp['decoupes']['poitrine']],
                "% Carcasse": [22, 17, 14, 12]
            }
            df_decoupes = pd.DataFrame(decoupes_data)
            st.dataframe(df_decoupes, hide_index=True, use_container_width=True)
        if brebis_id is not None:
            if st.button("ðŸ’¾ Enregistrer cette composition dans la base"):
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
                st.success("Composition enregistrÃ©e pour cette brebis !")
    st.divider()
    st.subheader("ðŸ” Comparer plusieurs brebis")
    if len(brebis_list) >= 2:
        selected_ids = st.multiselect(
            "Choisir les brebis Ã  comparer",
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
                st.warning("Aucune composition enregistrÃ©e pour ces brebis. Calculez d'abord une composition et enregistrez-la.")
    else:
        st.info("Ajoutez au moins deux brebis et enregistrez leurs compositions pour activer la comparaison.")

# ---- PAGE PRÃ‰DICTION ----
def page_prediction():
    st.title("ðŸ”® PrÃ©diction par Machine Learning")
    st.subheader("Potentiel laitier estimÃ©")
    col1, col2 = st.columns(2)
    with col1:
        score_mam = st.slider("Score mamelles", 1.0, 10.0, 7.0, 0.5)
        score_morpho = st.slider("Score morphologique", 0, 100, 75)
    with col2:
        race = st.selectbox("Race", list(Config.RACES.keys()))
        age = st.number_input("Ã‚ge (annÃ©es)", 1, 15, 4)
    if st.button("ðŸ”® PrÃ©dire production (formule simple)"):
        pred = MachineLearning.predire_lait(score_mam, score_morpho, race, age)
        cols = st.columns(3)
        cols[0].metric("Production/jour", f"{pred['litres_jour']} L")
        cols[1].metric("Production/lactation", f"{pred['litres_lactation']} L")
        cols[2].metric("Niveau", pred['niveau'])
        fig = px.bar(
            x=["Potentiel estimÃ©", "Moyenne race", "Record Ã©lite"],
            y=[pred['litres_jour'], 1.2, 2.5],
            color=[pred['niveau'], "Moyenne", "Ã‰lite"],
            title="Comparaison production laitiÃ¨re (L/jour)"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("PrÃ©diction avancÃ©e par modÃ¨le ML")
    model_path = os.path.join(MODEL_DIR, 'lait_model.pkl')
    if os.path.exists(model_path):
        st.success("Un modÃ¨le ML est disponible.")
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
            if st.button("PrÃ©dire avec ML"):
                pred = predict_lait_ml(bid)
                if pred is not None:
                    st.metric("Production prÃ©dite (L/j)", f"{pred:.2f}")
                else:
                    st.warning("Impossible de faire la prÃ©diction (donnÃ©es manquantes).")
        else:
            st.warning("Aucune brebis disponible.")
    else:
        st.info("Aucun modÃ¨le ML entraÃ®nÃ©. Vous pouvez en entraÃ®ner un si vous avez suffisamment de donnÃ©es de production.")
        if st.button("EntraÃ®ner un modÃ¨le ML"):
            with st.spinner("EntraÃ®nement en cours..."):
                result = train_lait_model()
                if result is None:
                    st.error("Pas assez de donnÃ©es (minimum 20 brebis avec productions).")
                else:
                    model, score = result
                    st.success(f"ModÃ¨le entraÃ®nÃ© avec un score RÂ² de {score:.2f} sur le test.")

# ---- PAGE PHOTOGRAMMÃ‰TRIE AUTOMATISÃ‰E (avec analyse mamelle) ----
def page_analyse_auto():
    st.title("ðŸ“¸ PhotogrammÃ©trie AutomatisÃ©e (IA)")
    st.markdown("""
    **Cette version automatise la collecte de donnÃ©es massives :**
    - Capture vidÃ©o en rafale (plusieurs images en une prise)
    - Filtrage qualitÃ© (flou, contraste)
    - DÃ©tection automatique des points clÃ©s par IA (MediaPipe ou modÃ¨le custom)
    - Validation manuelle possible si nÃ©cessaire
    - Enregistrement des mÃ©tadonnÃ©es (qualitÃ©, mÃ©tÃ©o)
    - **Nouveau :** Analyse automatique de la mamelle (rougeur, asymÃ©trie)
    """)
    mode = st.radio("Mode de capture", ["VidÃ©o (rafale)", "Photo unique"], index=0)
    etalon = st.selectbox("Ã‰talon de calibration", list(Config.ETALONS.keys()),
                         format_func=lambda x: Config.ETALONS[x]['nom'])
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude (optionnel)", value=36.0, format="%.4f")
    with col2:
        lon = st.number_input("Longitude (optionnel)", value=2.0, format="%.4f")
    st.subheader("Calibration automatique")
    use_auto_scale = st.checkbox("DÃ©tecter l'Ã©talon automatiquement", value=True)
    facteur = None
    if not use_auto_scale:
        facteur = st.number_input("Facteur d'Ã©chelle (px/cm)", value=10.0, step=0.1)
    
    # Upload pour photo corps
    uploaded_file = st.file_uploader("Choisir une vidÃ©o ou une photo (corps)", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
    # Upload pour photo mamelle (optionnel)
    uploaded_mamelle = st.file_uploader("Photo de la mamelle (optionnelle)", type=['jpg', 'png', 'jpeg'], key="mamelle")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if uploaded_file.type.startswith('video'):
            with st.spinner("Extraction des images de la vidÃ©o..."):
                frames = extract_frames_from_video(file_bytes, n_frames=10)
            if not frames:
                st.error("Impossible d'extraire des images de la vidÃ©o.")
                return
            st.success(f"{len(frames)} images extraites.")
            with st.spinner("Filtrage des images de qualitÃ©..."):
                best_frames = filter_best_images(frames)
            st.success(f"{len(best_frames)} images retenues aprÃ¨s filtrage.")
            if not best_frames:
                st.warning("Aucune image de qualitÃ© suffisante. Utilisation de la premiÃ¨re image.")
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
            st.image(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB), caption="Photo chargÃ©e", use_column_width=True)
        
        # DÃ©tection des points clÃ©s (hybride)
        st.subheader("DÃ©tection IA des points anatomiques")
        custom_model_path = os.path.join(MODEL_DIR, 'keypoints_model_custom.h5')
        with st.spinner("Analyse par IA..."):
            img_rgb = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
            keypoints = detect_keypoints_hybrid(img_rgb, custom_model_path)
        
        if keypoints is None:
            st.warning("Aucun point dÃ©tectÃ© par l'IA. Veuillez passer en mode manuel.")
            coord = streamlit_image_coordinates(img_rgb, key="manual_fallback")
            if coord:
                x, y = coord["x"], coord["y"]
                st.write(f"Point cliquÃ© : ({x}, {y})")
        else:
            st.success(f"Points dÃ©tectÃ©s : Garrot {keypoints['garrot']}, Ã‰paule {keypoints['epaule']}, Fesse {keypoints['fesse']}")
            img_with_kp = selected_img.copy()
            for pt in keypoints.values():
                cv2.circle(img_with_kp, pt, 6, (0,255,0), -1)
            st.image(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB), caption="Points dÃ©tectÃ©s", use_column_width=True)
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
                st.subheader("ðŸ”¬ Analyse automatique de la mamelle")
                mamelle_img = Image.open(uploaded_mamelle)
                mamelle_cv = cv2.cvtColor(np.array(mamelle_img), cv2.COLOR_RGB2BGR)
                st.image(mamelle_img, caption="Photo de la mamelle", use_column_width=True)
                with st.spinner("Analyse en cours..."):
                    mamelle_report = analyze_mamelle(mamelle_cv)
                st.write(f"**Niveau d'alerte :** {mamelle_report['level']}")
                st.write(f"**Ratio de rougeur :** {mamelle_report['red_ratio']}")
                st.write(f"**Ratio d'asymÃ©trie :** {mamelle_report['asym_ratio']}")
                if mamelle_report['alerts']:
                    for alert in mamelle_report['alerts']:
                        st.warning(alert)
                else:
                    st.success("âœ… Aucune anomalie dÃ©tectÃ©e.")
            
            if st.button("ðŸ’¾ Enregistrer cette image et ses points dans le dataset"):
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
                        st.error("Impossible de dÃ©tecter l'Ã©talon automatiquement. Veuillez saisir manuellement.")
                        facteur = st.number_input("Facteur d'Ã©chelle (px/cm)", value=10.0, step=0.1)
                else:
                    facteur = facteur
                if facteur is None:
                    st.warning("Facteur d'Ã©chelle non dÃ©fini.")
                    return
                weather_data = None
                if lat and lon:
                    api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
                    if api_key:
                        weather_data = get_weather_data(api_key, lat, lon)
                        if weather_data:
                            st.info(f"MÃ©tÃ©o : {weather_data['temp']}Â°C, {weather_data['humidity']}% humiditÃ©")
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
                st.success(f"DonnÃ©es enregistrÃ©es dans {filename}")
                st.info("DonnÃ©es sauvegardÃ©es dans le dataset d'entraÃ®nement.")
    else:
        st.info("Veuillez charger une vidÃ©o ou une photo du corps.")
    st.divider()
    with st.expander("ðŸ“Š Voir les donnÃ©es collectÃ©es"):
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.npz')]
        if files:
            st.write(f"{len(files)} Ã©chantillons dans le dataset.")
            data_summary = []
            for f in files[-10:]:
                data = np.load(os.path.join(DATASET_DIR, f), allow_pickle=True)
                meta = data['metadata'].item()
                data_summary.append({
                    'Fichier': f,
                    'Race': meta.get('race', '?'),
                    'Ã‚ge (mois)': meta.get('age_mois', '?'),
                    'Facteur': meta.get('facteur_echelle', '?'),
                    'MÃ©tÃ©o': meta.get('meteo', {}).get('temp', '?')
                })
            if data_summary:
                st.dataframe(pd.DataFrame(data_summary), use_container_width=True)
        else:
            st.info("Aucune donnÃ©e collectÃ©e.")

# ---- PAGE NUTRITION AVANCÃ‰E (inchangÃ©e mais complÃ¨te) ----
def page_nutrition_avancee():
    st.title("ðŸŒ¾ Nutrition avancÃ©e et formulation")
    tab1, tab2, tab3 = st.tabs(["ðŸ“¦ Catalogue aliments", "ðŸ“‹ Rations types", "ðŸ§® Calcul ration personnalisÃ©e"])
    with tab1:
        st.subheader("Gestion des aliments")
        with st.expander("âž• Ajouter un aliment"):
            with st.form("form_aliment"):
                nom = st.text_input("Nom de l'aliment")
                type_alim = st.selectbox("Type", ["Fourrage", "ConcentrÃ©", "MinÃ©ral", "Autre"])
                uem = st.number_input("UEM (MJ/kg)", min_value=0.0, step=0.1, format="%.2f")
                pdin = st.number_input("PDIN (g/kg)", min_value=0.0, step=1.0)
                ms = st.number_input("MatiÃ¨re sÃ¨che (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
                prix = st.number_input("Prix (DA/kg)", min_value=0.0, step=1.0, format="%.2f")
                if st.form_submit_button("Ajouter"):
                    try:
                        db.execute(
                            "INSERT INTO aliments (nom, type, uem, pdin, ms, prix_kg) VALUES (?, ?, ?, ?, ?, ?)",
                            (nom, type_alim, uem, pdin, ms, prix)
                        )
                        st.success("Aliment ajoutÃ©")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Cet aliment existe dÃ©jÃ .")
        aliments = db.fetchall("SELECT id, nom, type, uem, pdin, ms, prix_kg FROM aliments")
        if aliments:
            df_alim = pd.DataFrame(aliments, columns=["ID", "Nom", "Type", "UEM", "PDIN", "MS%", "Prix DA/kg"])
            st.dataframe(df_alim, use_container_width=True, hide_index=True)
            with st.expander("ðŸ’° Modifier un prix"):
                choix = st.selectbox("Choisir un aliment", [f"{a[0]} - {a[1]}" for a in aliments])
                aid = int(choix.split(" - ")[0])
                nouveau_prix = st.number_input("Nouveau prix (DA/kg)", min_value=0.0, step=1.0)
                if st.button("Mettre Ã  jour"):
                    db.execute("UPDATE aliments SET prix_kg=? WHERE id=?", (nouveau_prix, aid))
                    st.success("Prix mis Ã  jour")
                    st.rerun()
        else:
            st.info("Aucun aliment enregistrÃ©. Commencez par en ajouter.")
    with tab2:
        st.subheader("Rations types par Ã©tat physiologique")
        etat_physio = st.selectbox("Ã‰tat physiologique", Config.ETATS_PHYSIO)
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
                df_compo = pd.DataFrame(compo, columns=["Aliment", "QuantitÃ© (kg/jour)", "Prix/kg"])
                df_compo["CoÃ»t (DA/jour)"] = df_compo["QuantitÃ© (kg/jour)"] * df_compo["Prix/kg"]
                st.dataframe(df_compo, use_container_width=True, hide_index=True)
                total_journalier = df_compo["CoÃ»t (DA/jour)"].sum()
                st.metric("CoÃ»t total journalier", f"{total_journalier:.2f} DA")
            else:
                st.info("Cette ration n'a pas d'aliments associÃ©s.")
        else:
            st.info("Aucune ration dÃ©finie pour cet Ã©tat.")
        with st.expander("âš™ï¸ Configurer une ration pour cet Ã©tat"):
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
                    if st.button("CrÃ©er la ration"):
                        db.execute(
                            "INSERT INTO rations (nom, etat_physio, description) VALUES (?, ?, ?)",
                            (nom_ration, etat_physio, desc)
                        )
                        st.success("Ration crÃ©Ã©e, vous pouvez maintenant ajouter des aliments.")
                        st.rerun()
                    ration_id = None
                if ration_id:
                    st.subheader("Ajouter un aliment Ã  cette ration")
                    aliment_choix = st.selectbox("Choisir un aliment", [f"{a[0]} - {a[1]}" for a in aliments])
                    aid = int(aliment_choix.split(" - ")[0])
                    quantite = st.number_input("QuantitÃ© (kg/jour)", min_value=0.0, step=0.1, format="%.2f")
                    if st.button("Ajouter Ã  la ration"):
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
                        st.success("Aliment ajoutÃ©/modifiÃ©")
                        st.rerun()
                    with st.expander("ðŸ—‘ï¸ Supprimer un aliment de la ration"):
                        compo = db.fetchall("""
                            SELECT rc.id, a.nom FROM ration_composition rc
                            JOIN aliments a ON rc.aliment_id = a.id
                            WHERE rc.ration_id=?
                        """, (ration_id,))
                        if compo:
                            choix_suppr = st.selectbox("Aliment Ã  retirer", [f"{c[0]} - {c[1]}" for c in compo])
                            suppr_id = int(choix_suppr.split(" - ")[0])
                            if st.button("Retirer"):
                                db.execute("DELETE FROM ration_composition WHERE id=?", (suppr_id,))
                                st.success("Aliment retirÃ©")
                                st.rerun()
    with tab3:
        st.subheader("Calcul de ration personnalisÃ©e")
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
            choix = st.selectbox("Choisir une brebis (ou personnaliser)", ["PersonnalisÃ©"] + list(brebis_dict.keys()))
            if choix != "PersonnalisÃ©":
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
                etat = st.selectbox("Ã‰tat physiologique", Config.ETATS_PHYSIO, index=Config.ETATS_PHYSIO.index(etat_def) if etat_def in Config.ETATS_PHYSIO else 0)
            lactation = st.number_input("Production laitiÃ¨re (L/j)", min_value=0.0, value=0.0, step=0.5)
            besoins = OvinScience.besoins_nutritionnels(poids, etat, lactation)
            st.info(f"**Besoins journaliers** : UEM = {besoins['uem']} MJ, PDIN = {besoins['pdin']} g, MS = {besoins['ms']} kg")
            aliments = db.fetchall("SELECT id, nom, type, uem, pdin, ms, prix_kg FROM aliments")
            if not aliments:
                st.warning("Ajoutez d'abord des aliments.")
            else:
                mode_ration = st.radio("Mode de composition", ["Manuel", "Optimisation automatique (coÃ»t minimum)"])
                if mode_ration == "Manuel":
                    st.subheader("Composition de la ration")
                    ration_temp = {}
                    for alim in aliments:
                        with st.expander(f"{alim[1]} ({alim[2]}) - {alim[6]} DA/kg"):
                            qte = st.number_input("QuantitÃ© (kg MS)", min_value=0.0, step=0.1, key=f"qte_{alim[0]}")
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
                        st.subheader("RÃ©sultats")
                        cola, colb, colc = st.columns(3)
                        cola.metric("UEM apportÃ©e", f"{total_uem:.2f} MJ", delta=f"{total_uem - besoins['uem']:.2f}")
                        colb.metric("PDIN apportÃ©e", f"{total_pdin:.2f} g", delta=f"{total_pdin - besoins['pdin']:.2f}")
                        colc.metric("MS apportÃ©e", f"{total_ms:.2f} kg", delta=f"{total_ms - besoins['ms']:.2f}")
                        st.metric("CoÃ»t journalier", f"{total_prix:.2f} DA")
                        if total_uem < besoins['uem'] * 0.9:
                            st.warning("âš ï¸ Apport Ã©nergÃ©tique insuffisant")
                        elif total_uem > besoins['uem'] * 1.1:
                            st.warning("âš ï¸ ExcÃ¨s d'Ã©nergie")
                        else:
                            st.success("âœ… Ã‰nergie Ã©quilibrÃ©e")
                        if total_pdin < besoins['pdin'] * 0.9:
                            st.warning("âš ï¸ Apport protÃ©ique insuffisant")
                        elif total_pdin > besoins['pdin'] * 1.1:
                            st.warning("âš ï¸ ExcÃ¨s de protÃ©ines")
                        else:
                            st.success("âœ… ProtÃ©ines Ã©quilibrÃ©es")
                else:
                    st.subheader("Optimisation de la ration (coÃ»t minimum)")
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
                    tolerance = st.slider("TolÃ©rance sur les besoins (%)", 0, 20, 10) / 100
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
                            df_opt["CoÃ»t (DA/jour)"] = df_opt["qte"] * df_opt["prix"]
                            st.dataframe(df_opt[["nom", "qte", "CoÃ»t (DA/jour)"]].round(2), use_container_width=True, hide_index=True)
                            total_opt = df_opt["CoÃ»t (DA/jour)"].sum()
                            st.metric("CoÃ»t optimal journalier", f"{total_opt:.2f} DA")
                            uem_tot = sum(q * aliments[i][3] for i, q in enumerate(quantites))
                            pdin_tot = sum(q * aliments[i][4] for i, q in enumerate(quantites))
                            ms_tot = sum(q * aliments[i][5]/100 for i, q in enumerate(quantites))
                            st.write(f"UEM apportÃ©e : {uem_tot:.2f} MJ (besoin {besoins['uem']})")
                            st.write(f"PDIN apportÃ©e : {pdin_tot:.2f} g (besoin {besoins['pdin']})")
                            st.write(f"MS apportÃ©e : {ms_tot:.2f} kg (max {besoins['ms']* (1+tolerance):.2f})")
                        else:
                            st.warning("Aucun aliment sÃ©lectionnÃ© par l'optimisation.")
                    else:
                        st.error("Impossible de trouver une solution optimale. VÃ©rifiez les contraintes ou ajoutez des aliments.")
        else:
            st.info("Aucune brebis disponible. Vous pouvez utiliser 'PersonnalisÃ©'.")

# ---- PAGE PRODUCTION LAITIÃˆRE ----
def page_production():
    st.title("ðŸ¥› Production laitiÃ¨re et analyses biochimiques")
    tab1, tab2 = st.tabs(["ðŸ“ˆ Suivi production", "ðŸ§ª Analyses biochimiques"])
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
        st.warning("Aucune brebis disponible pour cet Ã©leveur.")
        return
    with tab1:
        st.subheader("Saisie d'une production")
        with st.form("form_prod"):
            brebis_choice = st.selectbox("Brebis", list(brebis_dict.keys()))
            date_prod = st.date_input("Date", value=datetime.today().date())
            quantite = st.number_input("QuantitÃ© de lait (L)", min_value=0.0, step=0.1)
            if st.form_submit_button("Enregistrer production"):
                brebis_id = brebis_dict[brebis_choice]
                db.execute(
                    "INSERT INTO productions (brebis_id, date, quantite) VALUES (?, ?, ?)",
                    (brebis_id, date_prod.isoformat(), quantite)
                )
                st.success("Production enregistrÃ©e")
                st.rerun()
        st.subheader("Ã‰volution de la production")
        brebis_graph = st.selectbox("Choisir une brebis pour le graphique", list(brebis_dict.keys()), key="graph_brebis")
        bid = brebis_dict[brebis_graph]
        data = db.fetchall(
            "SELECT date, quantite FROM productions WHERE brebis_id=? ORDER BY date",
            (bid,)
        )
        if data:
            df = pd.DataFrame(data, columns=["Date", "QuantitÃ© (L)"])
            df["Date"] = pd.to_datetime(df["Date"])
            fig = px.line(df, x="Date", y="QuantitÃ© (L)", title=f"Production de {brebis_graph}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnÃ©e pour cette brebis.")
        st.subheader("Production par Ã©leveur")
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
            df_all = pd.DataFrame(data_all, columns=["Ã‰leveur", "Brebis", "Date", "QuantitÃ©"])
            df_all["Date"] = pd.to_datetime(df_all["Date"])
            fig2 = px.line(df_all, x="Date", y="QuantitÃ©", color="Brebis", line_group="Brebis",
                          title="Production par brebis")
            st.plotly_chart(fig2, use_container_width=True)
            total_par_eleveur = df_all.groupby("Ã‰leveur")["QuantitÃ©"].sum().reset_index()
            fig3 = px.bar(total_par_eleveur, x="Ã‰leveur", y="QuantitÃ©", title="Production totale par Ã©leveur")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Aucune donnÃ©e de production.")
    with tab2:
        st.subheader("Analyses biochimiques du lait")
        with st.form("form_biochimie"):
            brebis_choice2 = st.selectbox("Brebis", list(brebis_dict.keys()), key="bio_brebis")
            date_bio = st.date_input("Date de l'analyse", value=datetime.today().date())
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.7, step=0.1)
            mg = st.number_input("MatiÃ¨re grasse (g/L)", min_value=0.0, value=65.0, step=0.1)
            proteine = st.number_input("ProtÃ©ines (g/L)", min_value=0.0, value=55.0, step=0.1)
            ag_satures = st.number_input("Acides gras saturÃ©s (g/L)", min_value=0.0, value=35.0, step=0.1)
            densite = st.number_input("DensitÃ©", min_value=1.0, max_value=1.1, value=1.035, step=0.001, format="%.3f")
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
                st.success("Analyse enregistrÃ©e")
                st.rerun()
        st.subheader("DerniÃ¨res analyses enregistrÃ©es")
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
            df_bio = pd.DataFrame(data_bio, columns=["NumÃ©ro", "Nom", "Date", "pH", "MG", "ProtÃ©ines", "AGS", "DensitÃ©", "Extrait sec"])
            st.dataframe(df_bio, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune analyse biochimique.")

# ---- PAGE GÃ‰NOMIQUE AVANCÃ‰E ----
def page_genomique_avancee():
    st.title("ðŸ§¬ GÃ©nomique avancÃ©e")
    tab1, tab2, tab3 = st.tabs(["ðŸ” BLAST", "ðŸ§¬ SNPs d'intÃ©rÃªt", "ðŸ“Š GWAS"])
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
            blast_brebis = st.selectbox("SÃ©lectionner une brebis (pour utiliser sa sÃ©quence FASTA)", 
                                        ["Nouvelle sÃ©quence"] + list(brebis_dict.keys()))
            if blast_brebis != "Nouvelle sÃ©quence":
                bid = brebis_dict[blast_brebis]
                seq_result = db.fetchone("SELECT sequence_fasta FROM brebis WHERE id=?", (bid,))
                if seq_result and seq_result[0]:
                    default_seq = seq_result[0]
        seq_input = st.text_area("SÃ©quence FASTA", value=default_seq, height=150)
        database = st.selectbox("Base de donnÃ©es", ["nr", "nt", "refseq_rna", "refseq_protein"])
        if st.button("Lancer BLAST"):
            if not seq_input:
                st.error("Veuillez entrer une sÃ©quence.")
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
                        st.warning("Le BLAST en ligne est complexe Ã  intÃ©grer. Pour une dÃ©monstration, nous affichons un rÃ©sultat factice.")
                        time.sleep(2)
                        st.success("BLAST terminÃ© (simulation)")
                        mock_results = [
                            {"accession": "XM_004012345.1", "description": "Ovis aries BMP15 mRNA", "score": 1234, "evalue": 1e-150},
                            {"accession": "NM_001009345.1", "description": "Ovis aries MSTN mRNA", "score": 1100, "evalue": 1e-140},
                        ]
                        df_mock = pd.DataFrame(mock_results)
                        st.dataframe(df_mock)
                        if st.button("Enregistrer ce rÃ©sultat"):
                            st.info("FonctionnalitÃ© Ã  implÃ©menter (sauvegarde en base)")
                    except Exception as e:
                        st.error(f"Erreur BLAST: {e}")
    with tab2:
        st.subheader("SNPs d'intÃ©rÃªt Ã©conomique")
        st.markdown("**GÃ¨nes d'intÃ©rÃªt et SNPs associÃ©s**")
        df_genes = pd.DataFrame([
            {"GÃ¨ne": sym, "Nom": info["nom"], "Effet": info["effet"]}
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
                st.info("Aucun SNP enregistrÃ© pour cette brebis.")
            with st.expander("Ajouter / modifier les SNPs"):
                snps_json = st.text_area("SNPs au format JSON (ex: {'BMP15': 'AA', 'MSTN': 'GG'})", height=150)
                if st.button("Enregistrer"):
                    db.execute("UPDATE brebis SET variants_snps=? WHERE id=?", (snps_json, bid))
                    st.success("SNPs enregistrÃ©s")
                    st.rerun()
    with tab3:
        st.subheader("Analyse d'association GWAS")
        st.markdown("""
        Cette section permet de rÃ©aliser une Ã©tude d'association pangÃ©nomique simplifiÃ©e.
        Vous devez fournir deux fichiers CSV :
        - **GÃ©notypes** : avec une colonne `brebis_id` et une colonne par SNP (valeurs 0,1,2 pour le dosage allÃ©lique).
        - **PhÃ©notypes** : avec les colonnes `brebis_id` et un trait quantitatif (ex: production laitiÃ¨re, poids...).
        """)
        upload_geno = st.file_uploader("Fichier gÃ©notypes (CSV)", type="csv", key="geno")
        upload_pheno = st.file_uploader("Fichier phÃ©notypes (CSV)", type="csv", key="pheno")
        if upload_geno and upload_pheno:
            try:
                df_geno = pd.read_csv(upload_geno)
                df_pheno = pd.read_csv(upload_pheno)
                if 'brebis_id' not in df_geno.columns or 'brebis_id' not in df_pheno.columns:
                    st.error("Les fichiers doivent contenir une colonne 'brebis_id'.")
                else:
                    df_merged = pd.merge(df_geno, df_pheno, on='brebis_id')
                    trait_col = st.selectbox("SÃ©lectionner le trait phÃ©notypique", 
                                             [c for c in df_pheno.columns if c != 'brebis_id'])
                    snp_cols = [c for c in df_geno.columns if c != 'brebis_id' and df_geno[c].dtype in ['int64', 'float64']]
                    if len(snp_cols) == 0:
                        st.error("Aucune colonne SNP numÃ©rique trouvÃ©e.")
                    else:
                        st.write(f"Nombre de SNPs analysÃ©s : {len(snp_cols)}")
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

# ---- PAGE GESTION Ã‰LEVAGE (inchangÃ©e, mais reprise complÃ¨te) ----
def page_gestion_elevage():
    st.title("ðŸ‘ Gestion des Ã©levages")
    if st.session_state.eleveur_id is not None:
        eleveur = db.fetchone("SELECT nom, region FROM eleveurs WHERE id=?", (st.session_state.eleveur_id,))
        if eleveur:
            st.subheader(f"ðŸ“Š RÃ©sumÃ© de l'Ã©leveur : {eleveur[0]} ({eleveur[1]})")
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
            col1.metric("ðŸ¡ Ã‰levages", nb_elevages)
            col2.metric("ðŸ‘ Brebis", nb_brebis)
            col3.metric("ðŸ¥› Production moy. (L/j)", f"{prod_moy:.2f}" if prod_moy else "N/A")
            col4.metric("âš–ï¸ Poids moy. (kg)", f"{poids_moy:.1f}" if poids_moy else "N/A")
            races = db.fetchall("""
                SELECT b.race, COUNT(*) 
                FROM brebis b
                JOIN elevages e ON b.elevage_id = e.id
                WHERE e.eleveur_id=?
                GROUP BY b.race
            """, (st.session_state.eleveur_id,))
            if races:
                df_races = pd.DataFrame(races, columns=["Race", "Nombre"])
                fig = px.pie(df_races, values="Nombre", names="Race", title="RÃ©partition des races")
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.info("ðŸ‘ˆ SÃ©lectionnez un Ã©leveur dans la barre latÃ©rale pour voir un rÃ©sumÃ©.")
    tab1, tab2, tab3 = st.tabs(["ðŸ‘¨â€ðŸŒ¾ Ã‰leveurs", "ðŸ¡ Ã‰levages", "ðŸ‘ Brebis"])
    with tab1:
        st.subheader("Liste des Ã©leveurs")
        with st.expander("âž• Ajouter un Ã©leveur", expanded=True):
            with st.form("form_eleveur"):
                nom = st.text_input("Nom")
                region = st.text_input("RÃ©gion")
                telephone = st.text_input("TÃ©lÃ©phone")
                email = st.text_input("Email")
                submitted = st.form_submit_button("Ajouter")
                if submitted:
                    db.execute(
                        "INSERT INTO eleveurs (user_id, nom, region, telephone, email) VALUES (?, ?, ?, ?, ?)",
                        (st.session_state.user_id, nom, region, telephone, email)
                    )
                    st.success("Ã‰leveur ajoutÃ©")
                    st.rerun()
        eleveurs = db.fetchall(
            "SELECT id, nom, region, telephone, email FROM eleveurs WHERE user_id=?",
            (st.session_state.user_id,)
        )
        if eleveurs:
            df = pd.DataFrame(eleveurs, columns=["ID", "Nom", "RÃ©gion", "TÃ©lÃ©phone", "Email"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("ðŸ—‘ï¸ Supprimer un Ã©leveur"):
                del_id = st.selectbox("Choisir l'Ã©leveur", [f"{e[0]} - {e[1]}" for e in eleveurs], key="del_eleveur_select")
                if st.button("Supprimer", key="del_eleveur_btn"):
                    eid = int(del_id.split(" - ")[0])
                    count = db.fetchone("SELECT COUNT(*) FROM elevages WHERE eleveur_id=?", (eid,))[0]
                    if count > 0:
                        st.error("Cet Ã©leveur a encore des Ã©levages. Supprimez d'abord les Ã©levages.")
                    else:
                        db.execute("DELETE FROM eleveurs WHERE id=?", (eid,))
                        st.success("Ã‰leveur supprimÃ©")
                        st.rerun()
        else:
            st.info("Aucun Ã©leveur enregistrÃ©.")
    with tab2:
        st.subheader("Liste des Ã©levages")
        eleveurs_list = db.fetchall(
            "SELECT id, nom FROM eleveurs WHERE user_id=?", (st.session_state.user_id,)
        )
        eleveurs_dict = {f"{e[0]} - {e[1]}": e[0] for e in eleveurs_list}
        if not eleveurs_dict:
            st.warning("Vous devez d'abord ajouter un Ã©leveur.")
        else:
            with st.expander("âž• Ajouter un Ã©levage", expanded=True):
                with st.form("form_elevage"):
                    eleveur_choice = st.selectbox("Ã‰leveur", list(eleveurs_dict.keys()))
                    nom_elevage = st.text_input("Nom de l'Ã©levage")
                    localisation = st.text_input("Localisation")
                    superficie = st.number_input("Superficie (ha)", min_value=0.0, step=0.1)
                    submitted = st.form_submit_button("Ajouter")
                    if submitted:
                        eleveur_id = eleveurs_dict[eleveur_choice]
                        db.execute(
                            "INSERT INTO elevages (eleveur_id, nom, localisation, superficie) VALUES (?, ?, ?, ?)",
                            (eleveur_id, nom_elevage, localisation, superficie)
                        )
                        st.success("Ã‰levage ajoutÃ©")
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
                st.info("Aucun Ã©levage pour cet Ã©leveur.")
            else:
                df = pd.DataFrame(elevages, columns=["ID", "Nom", "Localisation", "Superficie", "Ã‰leveur"])
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
            st.warning("Aucun Ã©levage pour cet Ã©leveur. Veuillez d'abord ajouter un Ã©levage.")
        else:
            with st.expander("âž• Ajouter une brebis", expanded=False):
                with st.form("form_brebis"):
                    elevage_choice = st.selectbox("Ã‰levage", list(elevages_dict.keys()))
                    numero_id = st.text_input("NumÃ©ro d'identification (obligatoire)")
                    age_mode = st.radio("Mode de saisie de l'Ã¢ge", ["Ã‚ge en mois", "Dentition"])
                    date_naissance = None
                    if age_mode == "Ã‚ge en mois":
                        age_mois = st.number_input("Ã‚ge en mois", min_value=0, max_value=200, value=24, step=1)
                        date_naissance = datetime.today().date() - timedelta(days=age_mois * 30)
                        st.date_input("Date estimÃ©e (d'aprÃ¨s Ã¢ge)", value=date_naissance, disabled=True)
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
                        st.date_input("Date estimÃ©e (d'aprÃ¨s dentition)", value=date_naissance, disabled=True)
                    race = st.selectbox("Race", list(Config.RACES.keys()))
                    etat_physio = st.selectbox("Ã‰tat physiologique", Config.ETATS_PHYSIO)
                    photo_profil = st.file_uploader("Photo de profil (optionnelle)", type=['jpg','png','jpeg'])
                    photo_mamelle = st.file_uploader("Photo mamelle (optionnelle)", type=['jpg','png','jpeg'])
                    poids_vif = st.number_input("Poids vif (kg) (optionnel)", min_value=0.0, value=0.0, step=0.5)
                    submitted = st.form_submit_button("Ajouter")
                    if submitted and numero_id:
                        existing = db.fetchone("SELECT id FROM brebis WHERE numero_id=?", (numero_id,))
                        if existing:
                            st.error(f"Une brebis avec le numÃ©ro {numero_id} existe dÃ©jÃ .")
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
                            st.success("Brebis ajoutÃ©e")
                            st.rerun()
                    elif submitted and not numero_id:
                        st.error("Le numÃ©ro d'identification est obligatoire.")
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
                df_brebis = pd.DataFrame(brebis, columns=["ID", "NumÃ©ro", "Nom", "Race", "Naissance", "Ã‰tat", "Ã‰levage", "Poids vif (kg)", "Photo profil", "Photo mamelle"])
                st.dataframe(df_brebis[["NumÃ©ro", "Race", "Naissance", "Ã‰tat", "Ã‰levage", "Poids vif (kg)"]], use_container_width=True, hide_index=True)
                st.divider()
                st.subheader("ðŸ‘ Suivi individuel")
                selected_brebis = st.selectbox("Choisir une brebis", [f"{b[0]} - {b[1]}" for b in brebis], key="suivi_select")
                bid = int(selected_brebis.split(" - ")[0])
                brebis_info = db.fetchone("SELECT numero_id, nom, race, date_naissance, poids_vif, photo_profil, photo_mamelle FROM brebis WHERE id=?", (bid,))
                if brebis_info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("NumÃ©ro", brebis_info[0])
                    col2.metric("Nom", brebis_info[1] if brebis_info[1] else "Non renseignÃ©")
                    col3.metric("Race", brebis_info[2])
                    if brebis_info[3]:
                        naiss = datetime.strptime(brebis_info[3], "%Y-%m-%d").date()
                        age_jours = (datetime.today().date() - naiss).days
                        age_mois = age_jours // 30
                        age_ans = age_jours // 365
                        st.metric("Ã‚ge", f"{age_ans} ans ({age_mois} mois)")
                    else:
                        st.metric("Ã‚ge", "Non renseignÃ©")
                    st.metric("Dernier poids connu", f"{brebis_info[4]} kg" if brebis_info[4] else "Non renseignÃ©")
                tab_hist1, tab_hist2, tab_hist3, tab_hist4 = st.tabs(["ðŸ“ˆ Poids", "ðŸ¥› Production", "ðŸ“ MorphomÃ©trie", "ðŸ“ Notes"])
                with tab_hist1:
                    poids_data = db.fetchall("""
                        SELECT date_estimation, poids_vif FROM composition_corporelle 
                        WHERE brebis_id=? ORDER BY date_estimation
                    """, (bid,))
                    if poids_data:
                        df_poids = pd.DataFrame(poids_data, columns=["Date", "Poids (kg)"])
                        df_poids["Date"] = pd.to_datetime(df_poids["Date"])
                        fig_poids = px.line(df_poids, x="Date", y="Poids (kg)", title="Ã‰volution du poids")
                        st.plotly_chart(fig_poids, use_container_width=True)
                    else:
                        st.info("Aucune donnÃ©e de poids historique.")
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
                            st.success("Poids enregistrÃ© !")
                            st.rerun()
                with tab_hist2:
                    prod_data = db.fetchall("""
                        SELECT date, quantite FROM productions WHERE brebis_id=? ORDER BY date
                    """, (bid,))
                    if prod_data:
                        df_prod = pd.DataFrame(prod_data, columns=["Date", "Lait (L)"])
                        df_prod["Date"] = pd.to_datetime(df_prod["Date"])
                        fig_prod = px.line(df_prod, x="Date", y="Lait (L)", title="Production laitiÃ¨re")
                        st.plotly_chart(fig_prod, use_container_width=True)
                    else:
                        st.info("Aucune donnÃ©e de production.")
                    with st.form("form_prod_suivi"):
                        date_prod = st.date_input("Date", value=datetime.today().date())
                        quantite = st.number_input("QuantitÃ© (L)", min_value=0.0, step=0.1)
                        if st.form_submit_button("Enregistrer production"):
                            db.execute("INSERT INTO productions (brebis_id, date, quantite) VALUES (?, ?, ?)",
                                      (bid, date_prod.isoformat(), quantite))
                            st.success("Production enregistrÃ©e !")
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
                        fig_score = px.line(df_morpho, x="Date", y="Score", title="Ã‰volution du score morphologique")
                        st.plotly_chart(fig_score, use_container_width=True)
                    else:
                        st.info("Aucune mesure morphomÃ©trique.")
                    if st.button("ðŸ“¸ Aller Ã  la photogrammÃ©trie pour cette brebis"):
                        st.session_state.brebis_analyse_id = bid
                        st.session_state.current_page = "analyse_auto"
                        st.rerun()
                with tab_hist4:
                    diag_data = db.fetchall("""
                        SELECT date, maladie, symptomes, traitement FROM diagnostics WHERE brebis_id=? ORDER BY date DESC
                    """, (bid,))
                    if diag_data:
                        df_diag = pd.DataFrame(diag_data, columns=["Date", "Maladie", "SymptÃ´mes", "Traitement"])
                        st.dataframe(df_diag, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune note de diagnostic.")
                    with st.form("form_diag"):
                        date_diag = st.date_input("Date", value=datetime.today().date())
                        maladie = st.text_input("Maladie / Observation")
                        symptomes = st.text_area("SymptÃ´mes")
                        traitement = st.text_area("Traitement")
                        if st.form_submit_button("Enregistrer"):
                            db.execute("""
                                INSERT INTO diagnostics (brebis_id, date, maladie, symptomes, traitement)
                                VALUES (?, ?, ?, ?, ?)
                            """, (bid, date_diag.isoformat(), maladie, symptomes, traitement))
                            st.success("Note enregistrÃ©e !")
                            st.rerun()
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("ðŸ—‘ï¸ Supprimer cette brebis", key="del_brebis_suivi"):
                        photos = db.fetchone("SELECT photo_profil, photo_mamelle FROM brebis WHERE id=?", (bid,))
                        if photos:
                            for p in photos:
                                if p:
                                    try:
                                        os.remove(os.path.join(PHOTO_DIR, p))
                                    except:
                                        pass
                        db.execute("DELETE FROM brebis WHERE id=?", (bid,))
                        st.success("Brebis supprimÃ©e")
                        st.rerun()
                with col2:
                    if st.button("ðŸ“‹ Voir dÃ©tails complets", key="details_brebis_suivi"):
                        b = db.fetchone("SELECT * FROM brebis WHERE id=?", (bid,))
                        cols = [col[0] for col in db.conn.execute("PRAGMA table_info(brebis)").fetchall()]
                        data = dict(zip(cols, b))
                        with st.expander("DÃ©tails de la brebis", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("NumÃ©ro", data.get('numero_id', 'N/A'))
                                st.metric("Race", data.get('race', 'N/A'))
                                st.metric("Ã‰tat physiologique", data.get('etat_physio', 'N/A'))
                                if data.get('date_naissance'):
                                    naiss = datetime.strptime(data['date_naissance'], "%Y-%m-%d").date()
                                    age_jours = (datetime.today().date() - naiss).days
                                    age_mois = age_jours // 30
                                    st.metric("Ã‚ge", f"{age_mois} mois")
                            with col_b:
                                st.metric("Poids vif (kg)", data.get('poids_vif', 'Non renseignÃ©'))
                                if data.get('photo_profil'):
                                    st.image(os.path.join(PHOTO_DIR, data['photo_profil']), caption="Photo de profil", width=200)
                                if data.get('photo_mamelle'):
                                    st.image(os.path.join(PHOTO_DIR, data['photo_mamelle']), caption="Photo mamelle", width=200)
            else:
                st.info("Aucune brebis enregistrÃ©e.")

# ---- PAGE SANTÃ‰ ----
def page_sante():
    st.title("ðŸ¥ Suivi sanitaire et vaccinal")
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
        "ðŸ“œ Historique", 
        "â° Rappels", 
        "ðŸ“Š Statistiques", 
        "ðŸ¤– IA & PrÃ©dictions", 
        "ðŸ“¤ Export"
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
                "DÃ©tails": ""
            })
        for s in soins:
            historique.append({
                "Date": s[0],
                "Type": s[3],
                "Description": s[1],
                "DÃ©tails": s[2]
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
            fig = px.bar(df_count, x="Date", y="Nombre", color="Type", title="Ã‰vÃ©nements par mois")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun Ã©vÃ©nement enregistrÃ© pour cette brebis.")
        with st.expander("âž• Ajouter un Ã©vÃ©nement"):
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
                        st.success("Vaccin enregistrÃ©")
                        st.rerun()
            else:
                with st.form("form_soin_rapide"):
                    date_soin = st.date_input("Date du soin", value=datetime.today().date())
                    type_soin = st.selectbox("Type", ["Maladie", "Parasite", "Blessure", "Autre"])
                    diagnostic = st.text_area("Diagnostic / SymptÃ´mes")
                    traitement = st.text_area("Traitement administrÃ©")
                    if st.form_submit_button("Enregistrer"):
                        db.execute(
                            "INSERT INTO soins (brebis_id, date_soin, type, diagnostic, traitement) VALUES (?, ?, ?, ?, ?)",
                            (bid, date_soin.isoformat(), type_soin, diagnostic, traitement)
                        )
                        st.success("Soin enregistrÃ©")
                        st.rerun()
    with tab2:
        st.subheader("Rappels Ã  venir")
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
                st.warning("âš ï¸ Certains rappels sont imminents !")
                st.dataframe(imminents)
        else:
            st.info("Aucun rappel programmÃ©.")
        soins_recents = db.fetchall("""
            SELECT date_soin, type, diagnostic, traitement
            FROM soins
            WHERE brebis_id=? AND date_soin >= date('now', '-30 days')
            ORDER BY date_soin DESC
        """, (bid,))
        if soins_recents:
            st.subheader("Traitements rÃ©cents (mois en cours)")
            df_recents = pd.DataFrame(soins_recents, columns=["Date", "Type", "Diagnostic", "Traitement"])
            st.dataframe(df_recents, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Statistiques sanitaires")
        soins_stats = db.fetchall("""
            SELECT type, COUNT(*) FROM soins WHERE brebis_id=? GROUP BY type
        """, (bid,))
        if soins_stats:
            df_stats = pd.DataFrame(soins_stats, columns=["Type", "Nombre"])
            fig = px.pie(df_stats, values="Nombre", names="Type", title="RÃ©partition des soins par type")
            st.plotly_chart(fig, use_container_width=True)
        soins_temp = db.fetchall("""
            SELECT strftime('%Y-%m', date_soin) as mois, COUNT(*) 
            FROM soins WHERE brebis_id=?
            GROUP BY mois
            ORDER BY mois
        """, (bid,))
        if soins_temp:
            df_temp = pd.DataFrame(soins_temp, columns=["Mois", "Nombre"])
            fig2 = px.line(df_temp, x="Mois", y="Nombre", title="Ã‰volution du nombre de soins")
            st.plotly_chart(fig2, use_container_width=True)
        dernier_vaccin = db.fetchone("""
            SELECT MAX(date_vaccin) FROM vaccinations WHERE brebis_id=?
        """, (bid,))[0]
        if dernier_vaccin:
            jours_depuis = (datetime.now() - datetime.strptime(dernier_vaccin, "%Y-%m-%d")).days
            st.metric("Dernier vaccin", f"il y a {jours_depuis} jours")
        else:
            st.info("Aucun vaccin enregistrÃ©.")
    with tab4:
        st.subheader("Intelligence Artificielle â€“ Analyses prÃ©dictives")
        st.caption(
            "Score de risque sanitaire calculÃ© Ã  partir de l'historique rÃ©el de l'animal "
            "(frÃ©quence des soins rÃ©cents, retard vaccinal, diagnostics antÃ©rieurs)."
        )
        if st.button("Ã‰valuer le risque pour cette brebis"):
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
            score += min(nb_soins_90j, 5) * 10          # soins frÃ©quents rÃ©cents
            score += min(nb_diagnostics, 5) * 6          # historique de maladies
            if retard_vaccin_jours is None or retard_vaccin_jours > 365:
                score += 25                               # jamais vaccinÃ© ou rappel trÃ¨s en retard
            elif retard_vaccin_jours > 180:
                score += 10

            if score >= 45:
                risque = "Ã‰levÃ©"
            elif score >= 20:
                risque = "ModÃ©rÃ©"
            else:
                risque = "Faible"

            st.metric("Risque estimÃ©", risque, help=f"Score composite : {score}/100")
            with st.expander("DÃ©tail du calcul"):
                st.write(f"- Soins sur 90 jours : **{nb_soins_90j}**")
                st.write(f"- Diagnostics enregistrÃ©s : **{nb_diagnostics}**")
                st.write(
                    f"- Dernier vaccin : **{'il y a ' + str(retard_vaccin_jours) + ' jours' if retard_vaccin_jours is not None else 'aucun enregistrÃ©'}**"
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
                        f"âš ï¸ Anomalie dÃ©tectÃ©e : derniÃ¨re valeur ({derniere:.2f} L) "
                        f"s'Ã©carte fortement de la moyenne rÃ©cente ({moyenne:.2f} L, z-score={z_score:.1f})."
                    )
                else:
                    st.success("Production laitiÃ¨re normale (pas d'Ã©cart significatif).")
            else:
                st.info("Production stable, pas de variance pour Ã©valuer une anomalie.")
        else:
            st.info("Pas assez de donnÃ©es pour la dÃ©tection d'anomalies (minimum 5 mesures).")
        st.subheader("Recommandations vaccinales")
        dernier_vaccin_annuel = db.fetchone("""
            SELECT date_vaccin FROM vaccinations 
            WHERE brebis_id=? AND vaccin LIKE '%entÃ©ro%' OR vaccin LIKE '%annuel%'
            ORDER BY date_vaccin DESC LIMIT 1
        """, (bid,))
        if dernier_vaccin_annuel:
            date_dernier = datetime.strptime(dernier_vaccin_annuel[0], "%Y-%m-%d")
            if (datetime.now() - date_dernier).days > 365:
                st.warning("âš ï¸ Le vaccin annuel est Ã  renouveler (plus d'un an).")
            else:
                mois_restants = 12 - ((datetime.now() - date_dernier).days // 30)
                st.info(f"Prochain rappel annuel dans environ {mois_restants} mois.")
        else:
            st.info("Aucun vaccin annuel enregistrÃ©. Il est recommandÃ© de vacciner.")
        if age < 1:
            st.info("Les agneaux de moins d'un an doivent Ãªtre vaccinÃ©s contre la pasteurellose.")
    with tab5:
        st.subheader("Exporter l'historique")
        if st.button("GÃ©nÃ©rer le rapport CSV"):
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
                    "DÃ©tails": ""
                })
            for s in soins_all:
                data.append({
                    "Date": s[0],
                    "Type": s[1],
                    "Description": s[2],
                    "Rappel": "",
                    "DÃ©tails": s[3]
                })
            if data:
                df_export = pd.DataFrame(data)
                df_export = df_export.sort_values("Date", ascending=False)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="ðŸ“¥ TÃ©lÃ©charger CSV",
                    data=csv,
                    file_name=f"sante_{numero}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Aucune donnÃ©e Ã  exporter.")

# ---- PAGE REPRODUCTION ----
def page_reproduction():
    st.title("ðŸ¤° Gestion de la reproduction")
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
    tab1, tab2, tab3 = st.tabs(["ðŸ”¥ Chaleurs", "ðŸ Saillies", "ðŸ‘ Mises bas"])
    with tab1:
        st.subheader("Observations des chaleurs / synchronisation")
        with st.form("form_chaleur"):
            date_debut = st.date_input("Date de dÃ©but", value=datetime.today().date())
            date_fin = st.date_input("Date de fin (optionnelle)", value=None)
            methode = st.selectbox("MÃ©thode", ["Naturelle", "ProgestagÃ¨ne", "Autre"])
            obs = st.text_area("Observations")
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO chaleurs (brebis_id, date_debut, date_fin, methode_synchro, observation) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_debut.isoformat(), date_fin.isoformat() if date_fin else None, methode, obs)
                )
                st.success("Chaleurs enregistrÃ©es")
                st.rerun()
        chaleurs = db.fetchall(
            "SELECT date_debut, date_fin, methode_synchro, observation FROM chaleurs WHERE brebis_id=? ORDER BY date_debut DESC",
            (bid,)
        )
        if chaleurs:
            df = pd.DataFrame(chaleurs, columns=["DÃ©but", "Fin", "MÃ©thode", "Observations"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with tab2:
        st.subheader("Saillies / InsÃ©minations")
        with st.form("form_saillie"):
            date_saillie = st.date_input("Date de saillie", value=datetime.today().date())
            male_id = st.text_input("Identifiant du bÃ©lier")
            methode = st.selectbox("MÃ©thode", ["Naturelle", "InsÃ©mination artificielle"])
            resultat = st.selectbox("RÃ©sultat", ["En attente", "Gestante", "Non gestante"])
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO saillies (brebis_id, date_saillie, male_id, methode, resultat) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_saillie.isoformat(), male_id, methode, resultat)
                )
                st.success("Saillie enregistrÃ©e")
                st.rerun()
        saillies = db.fetchall(
            "SELECT date_saillie, male_id, methode, resultat FROM saillies WHERE brebis_id=? ORDER BY date_saillie DESC",
            (bid,)
        )
        if saillies:
            df = pd.DataFrame(saillies, columns=["Date", "BÃ©lier", "MÃ©thode", "RÃ©sultat"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            last_gest = db.fetchone(
                "SELECT date_saillie FROM saillies WHERE brebis_id=? AND resultat='Gestante' ORDER BY date_saillie DESC",
                (bid,)
            )
            if last_gest:
                date_saillie = datetime.strptime(last_gest[0], "%Y-%m-%d").date()
                date_mb = date_saillie + timedelta(days=150)
                st.success(f"ðŸ“… Mise bas prÃ©vue autour du : {date_mb.strftime('%d/%m/%Y')}")
    with tab3:
        st.subheader("Mises bas enregistrÃ©es")
        with st.form("form_mb"):
            date_mb = st.date_input("Date de mise bas", value=datetime.today().date())
            nb_agneaux = st.number_input("Nombre d'agneaux", min_value=1, step=1)
            poids_portee = st.number_input("Poids total de la portÃ©e (kg)", min_value=0.0, step=0.1)
            remarques = st.text_area("Remarques")
            if st.form_submit_button("Enregistrer"):
                db.execute(
                    "INSERT INTO mises_bas (brebis_id, date_mise_bas, nb_agneaux, poids_portee, remarques) VALUES (?, ?, ?, ?, ?)",
                    (bid, date_mb.isoformat(), nb_agneaux, poids_portee, remarques)
                )
                st.success("Mise bas enregistrÃ©e")
                st.rerun()
        mbas = db.fetchall(
            "SELECT date_mise_bas, nb_agneaux, poids_portee, remarques FROM mises_bas WHERE brebis_id=? ORDER BY date_mise_bas DESC",
            (bid,)
        )
        if mbas:
            df = pd.DataFrame(mbas, columns=["Date", "Agneaux", "Poids portÃ©e (kg)", "Remarques"])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ---- PAGE EXPORT ----
def page_export():
    st.title("ðŸ“¤ Export des donnÃ©es")
    st.markdown("TÃ©lÃ©chargez l'ensemble de vos donnÃ©es au format CSV ou Excel pour les partager avec votre professeur.")
    format_export = st.radio("Format", ["CSV (dossier compressÃ©)", "Excel (fichier unique)"])
    inclure_photos = st.checkbox("Inclure les photos dans l'archive (pour CSV uniquement)", value=True)
    if st.button("GÃ©nÃ©rer l'export"):
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
                st.warning(f"La table {table} n'existe pas. Elle sera ignorÃ©e.")
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
                label="ðŸ“¥ TÃ©lÃ©charger Excel",
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
                label="ðŸ“¥ TÃ©lÃ©charger ZIP (CSV + photos)",
                data=zip_buffer,
                file_name=f"ovin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

# ---- PAGE Ã‰LITE ----
def page_elite():
    st.title("ðŸ† Ã‰lite et comparaison")
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
        st.warning("Aucune brebis trouvÃ©e pour le contexte sÃ©lectionnÃ©.")
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
    st.subheader("ðŸ“Š Tableau des brebis")
    colonnes_affichees = ["numero", "nom", "eleveur", "elevage", "race", "poids", "prod_moy (L/j)", "score_morpho", "viande_estimee (kg)", "rendement (%)"]
    st.dataframe(df[colonnes_affichees].round(2))
    st.subheader("ðŸ† Classement")
    critere = st.selectbox("CritÃ¨re de classement", 
                           ["prod_moy (L/j)", "score_morpho", "viande_estimee (kg)", "poids", "rendement (%)"])
    top_n = st.slider("Nombre de brebis Ã  afficher", 5, 50, 10)
    ascending = st.checkbox("Ordre croissant", False)
    df[critere] = pd.to_numeric(df[critere], errors='coerce')
    df_class = df[df[critere].notna()].copy()
    if df_class.empty:
        st.warning(f"Aucune valeur numÃ©rique valide pour le critÃ¨re {critere}.")
    else:
        if ascending:
            top = df_class.nsmallest(top_n, critere)
        else:
            top = df_class.nlargest(top_n, critere)
        st.dataframe(top[["numero", "nom", "eleveur", "elevage", critere]].round(2))
        fig = px.bar(top, x="nom", y=critere, color="eleveur", title=f"Top {top_n} - {critere}")
        st.plotly_chart(fig, use_container_width=True)
    if st.session_state.eleveur_id is None and len(df["eleveur"].unique()) > 1:
        st.subheader("ðŸ“ˆ Comparaison par Ã©leveur")
        numeric_cols = ["prod_moy (L/j)", "score_morpho", "poids", "viande_estimee (kg)", "rendement (%)"]
        df_eleveur = df.groupby("eleveur")[numeric_cols].mean().reset_index()
        for col in numeric_cols:
            df_eleveur[col] = pd.to_numeric(df_eleveur[col], errors='coerce').fillna(0)
        st.dataframe(df_eleveur.round(2))
        fig2 = px.bar(df_eleveur, x="eleveur", y=["prod_moy (L/j)", "score_morpho", "rendement (%)"], 
                     barmode="group", title="Performances moyennes par Ã©leveur")
        st.plotly_chart(fig2, use_container_width=True)

# ---- PAGE IA ----
def page_ia():
    st.title("ðŸ§  Intelligence Artificielle & Data Mining")
    st.markdown("Analyses avancÃ©es basÃ©es sur les donnÃ©es de votre Ã©levage.")
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ“ˆ PrÃ©diction laitiÃ¨re avancÃ©e",
        "ðŸ” DÃ©tection d'anomalies",
        "ðŸ“Š Clustering des brebis",
        "ðŸ“‚ Analyse exploratoire (import)"
    ])
    with tab1:
        st.subheader("PrÃ©diction de production laitiÃ¨re par modÃ¨le ML")
        model_path = os.path.join(MODEL_DIR, 'lait_model.pkl')
        if os.path.exists(model_path):
            st.success("Un modÃ¨le ML est disponible.")
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
                if st.button("PrÃ©dire avec ML"):
                    pred = predict_lait_ml(bid)
                    if pred is not None:
                        st.metric("Production prÃ©dite (L/j)", f"{pred:.2f}")
                    else:
                        st.warning("Impossible de faire la prÃ©diction (donnÃ©es manquantes).")
            else:
                st.warning("Aucune brebis disponible.")
        else:
            st.info("Aucun modÃ¨le ML entraÃ®nÃ©. Vous pouvez en entraÃ®ner un si vous avez suffisamment de donnÃ©es de production.")
            if st.button("EntraÃ®ner un modÃ¨le ML"):
                with st.spinner("EntraÃ®nement en cours..."):
                    result = train_lait_model()
                    if result is None:
                        st.error("Pas assez de donnÃ©es (minimum 20 brebis avec productions).")
                    else:
                        model, score = result
                        st.success(f"ModÃ¨le entraÃ®nÃ© avec un score RÂ² de {score:.2f} sur le test.")
    with tab2:
        st.subheader("DÃ©tection d'anomalies (Isolation Forest)")
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
            st.warning("Aucune donnÃ©e disponible.")
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
            st.write(f"**{len(anomalies)}** brebis potentiellement anormales dÃ©tectÃ©es.")
            if not anomalies.empty:
                st.dataframe(anomalies[['numero_id', 'nom', 'prod_moy', 'score_morpho', 'poids_vif']])
            else:
                st.success("Aucune anomalie dÃ©tectÃ©e.")
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
            st.warning("Aucune donnÃ©e disponible pour le clustering.")
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
                st.success("Fichier chargÃ© avec succÃ¨s.")
                st.dataframe(df.head())
                if profiling_available:
                    analyse_mode = st.radio("Type d'analyse", ["Statistiques descriptives", "Rapport complet (ydata-profiling)"])
                else:
                    st.info("Module ydata-profiling non installÃ©. Utilisation des statistiques descriptives.")
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
                        if st.button("GÃ©nÃ©rer le rapport d'analyse"):
                            with st.spinner("GÃ©nÃ©ration du rapport..."):
                                profile = ProfileReport(df, title="Rapport d'analyse", explorative=True)
                                st_profile_report(profile)
                    else:
                        st.warning("Le module ydata-profiling n'est pas disponible.")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

# ---- PAGE APPRENTISSAGE ----
def page_apprentissage():
    st.title("ðŸ§  Apprentissage automatique")
    st.markdown("Cette page permet d'entraÃ®ner un modÃ¨le de deep learning pour la dÃ©tection automatique des points anatomiques.")
    if not tensorflow_available:
        st.warning(
            "TensorFlow n'est pas disponible sur cet environnement (pas de "
            "wheel compatible avec la version de Python actuellement utilisÃ©e "
            "par la plateforme). L'entraÃ®nement d'un modÃ¨le custom est "
            "dÃ©sactivÃ© pour le moment, mais la dÃ©tection automatique des "
            "points anatomiques continue de fonctionner via MediaPipe."
        )
        return
    nb_images = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.npz')])
    st.write(f"Images collectÃ©es pour l'entraÃ®nement : **{nb_images}**")
    if nb_images < 10:
        st.warning("Il faut au moins 10 images pour un premier entraÃ®nement significatif. Continuez Ã  utiliser la photogrammÃ©trie et Ã  contribuer.")
    else:
        if st.button("ðŸš€ Lancer l'entraÃ®nement"):
            with st.spinner("EntraÃ®nement en cours... (cela peut prendre plusieurs minutes)"):
                model, history = entrainer_modele()
                if model is None:
                    st.error(history)
                else:
                    st.success("EntraÃ®nement terminÃ© ! ModÃ¨le sauvegardÃ© dans models/keypoints_model_custom.h5")
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
    st.subheader("Tester le modÃ¨le")
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
            st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), caption="Points prÃ©dits")
    else:
        st.info("Aucun modÃ¨le personnalisÃ© entraÃ®nÃ©. Utilisez la photogrammÃ©trie pour collecter des donnÃ©es et entraÃ®ner le modÃ¨le.")

# ---- PAGE IoT ----
def page_iot():
    st.title("ðŸ“¡ Import de donnÃ©es IoT (capteurs)")
    st.markdown("Chargez un fichier CSV contenant des donnÃ©es de capteurs (tempÃ©rature, activitÃ©, rythme cardiaque).")
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
            st.success(f"{inserted} lignes importÃ©es.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur: {e}")

# ---- PAGE VALIDATION ----
def page_validation():
    st.title("ðŸ“Š Validation des modÃ¨les")
    st.markdown("Compare les prÃ©dictions IA avec les mesures rÃ©elles.")
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
        st.info("Aucune donnÃ©e de validation disponible.")
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
                                  mode='lines+markers', name='PrÃ©dit'))
        fig2.add_trace(go.Scatter(x=df2['date_prediction'], y=df2['valeur_reelle'],
                                  mode='lines+markers', name='RÃ©el'))
        fig2.update_layout(title=f"Ã‰volution {var} â€“ PrÃ©dit vs RÃ©el")
        st.plotly_chart(fig2, use_container_width=True)

# ---- FONCTION ENTRAINEMENT MODÃˆLE CUSTOM (pour clÃ©s points) ----
def entrainer_modele():
    # Fonction similaire Ã  celle dÃ©jÃ  prÃ©sente, mais sauvegarde sous keypoints_model_custom.h5
    X, y, _ = charger_dataset()
    if X is None or len(X) < 10:
        return None, "Pas assez de donnÃ©es (minimum 10 images)."
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
        st.title(f"ðŸ‘ {Config.APP_NAME}")
        st.caption(f"**{Config.LABORATOIRE}** v{Config.VERSION}")
        st.divider()
        if st.session_state.user_id:
            eleveurs = db.fetchall(
                "SELECT id, nom FROM eleveurs WHERE user_id=? ORDER BY nom",
                (st.session_state.user_id,)
            )
            eleveurs_options = {"Tous les Ã©leveurs": None}
            eleveurs_options.update({f"{e[1]} (ID {e[0]})": e[0] for e in eleveurs})
            current = st.session_state.get("eleveur_id", None)
            default_index = 0
            for i, (label, eid) in enumerate(eleveurs_options.items()):
                if eid == current:
                    default_index = i
                    break
            selected_label = st.selectbox(
                "ðŸ‘¨â€ðŸŒ¾ Ã‰leveur actif",
                options=list(eleveurs_options.keys()),
                index=default_index,
                key="eleveur_selector"
            )
            st.session_state.eleveur_id = eleveurs_options[selected_label]
            st.divider()
            menu = st.radio(
                "Navigation",
                ["ðŸ“Š Tableau de bord", 
                 "ðŸ‘ Gestion Ã©levage",
                 "ðŸ§¬ GÃ©nomique NCBI", 
                 "ðŸ¥© Composition", 
                 "ðŸ“¸ PhotogrammÃ©trie auto", 
                 "ðŸ”® PrÃ©dictions", 
                 "ðŸŒ¾ Nutrition avancÃ©e",
                 "ðŸ¥› Production laitiÃ¨re",
                 "ðŸ§¬ GÃ©nomique avancÃ©e",
                 "ðŸ¥ SantÃ©",
                 "ðŸ¤° Reproduction",
                 "ðŸ“¤ Export donnÃ©es",
                 "ðŸ† Ã‰lite et comparaison",
                 "ðŸ§  IA & Data Mining",
                 "ðŸ§  Apprentissage automatique",
                 "ðŸ“¡ IoT",
                 "ðŸ“Š Validation",
                 "ðŸšª DÃ©connexion"],
                label_visibility="collapsed"
            )
            st.divider()
            if st.button("ðŸ’¾ Sauvegarde rapide", use_container_width=True):
                st.download_button(
                    label="TÃ©lÃ©charger JSON",
                    data=json.dumps({"user_id": st.session_state.user_id, "date": datetime.now().isoformat()}),
                    file_name=f"ovin_backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            page_map = {
                "ðŸ“Š Tableau de bord": "dashboard",
                "ðŸ‘ Gestion Ã©levage": "gestion_elevage",
                "ðŸ§¬ GÃ©nomique NCBI": "genomique",
                "ðŸ¥© Composition": "composition",
                "ðŸ“¸ PhotogrammÃ©trie auto": "analyse_auto",
                "ðŸ”® PrÃ©dictions": "prediction",
                "ðŸŒ¾ Nutrition avancÃ©e": "nutrition_avancee",
                "ðŸ¥› Production laitiÃ¨re": "production",
                "ðŸ§¬ GÃ©nomique avancÃ©e": "genomique_avancee",
                "ðŸ¥ SantÃ©": "sante",
                "ðŸ¤° Reproduction": "reproduction",
                "ðŸ“¤ Export donnÃ©es": "export",
                "ðŸ† Ã‰lite et comparaison": "elite",
                "ðŸ§  IA & Data Mining": "ia",
                "ðŸ§  Apprentissage automatique": "apprentissage",
                "ðŸ“¡ IoT": "iot",
                "ðŸ“Š Validation": "validation",
                "ðŸšª DÃ©connexion": "logout"
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
# POINT D'ENTRÃ‰E
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
        page_icon="ðŸ‘",
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
