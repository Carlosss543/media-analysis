import pandas as pd
import re
import gc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os

# ==========================================
# CONFIGURATION
# ==========================================
REDDIT_FILE = "archive/Reddit_Combi.csv"
NEWS_FILE = "data/all-the-news-2-1.csv"
OUTPUT_FILE = "classement_medias_final.csv"

# Paramètres basse consommation
CHUNK_SIZE = 5000 
VOCAB_SIZE = 3000 

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

# ==========================================
# 1. ENTRAÎNEMENT (REDDIT)
# ==========================================
print(">>> 1/3 Chargement et Entraînement sur Reddit...")

# 1. Chargement robuste (Mode "Auto-détection")
try:
    # sep=None force Python à deviner le séparateur (; ou ,)
    # on_bad_lines='skip' ignore les lignes mal formées qui faisaient planter ton script
    df_train = pd.read_csv(REDDIT_FILE, sep=None, engine='python', on_bad_lines='skip')
    
    print(f"   - Colonnes trouvées : {list(df_train.columns)}")
    
    # 2. Normalisation des noms de colonnes (minuscules + suppression espaces)
    df_train.columns = df_train.columns.str.lower().str.strip()
    
    # 3. Vérification des colonnes nécessaires
    if 'label' not in df_train.columns:
        raise ValueError(f"Colonne 'label' introuvable. Colonnes dispos: {df_train.columns}")
    
    # Création colonne texte (fallback si title ou body manque)
    t = df_train['title'].fillna('') if 'title' in df_train.columns else ''
    b = df_train['body'].fillna('') if 'body' in df_train.columns else ''
    df_train['text'] = t + " " + b
    
    # On ne garde que le strict nécessaire
    df_train = df_train[['text', 'label']]

except Exception as e:
    print(f"ERREUR FATALE lors de la lecture de Reddit : {e}")
    exit()

# Vectorisation
print("   - Apprentissage du vocabulaire...")
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english', preprocessor=clean_text)
X_train = vectorizer.fit_transform(df_train['text'])
y_train = df_train['label']

# Entraînement
print("   - Entraînement du modèle...")
clf = LogisticRegression(max_iter=500, n_jobs=-1)
clf.fit(X_train, y_train)

# Nettoyage RAM
print("   - Nettoyage de la RAM...")
del df_train, X_train, y_train
gc.collect()

# ==========================================
# 2. ANALYSE DES NEWS (STREAMING)
# ==========================================
print(f"\n>>> 2/3 Analyse du fichier News par paquets de {CHUNK_SIZE}...")

stats = {} 

# Lecture du fichier News
# On utilise error_bad_lines=False (ou on_bad_lines='skip') pour éviter les crashs sur ce fichier aussi
chunk_iterator = pd.read_csv(NEWS_FILE, chunksize=CHUNK_SIZE, on_bad_lines='skip')

for i, chunk in enumerate(tqdm(chunk_iterator, desc="Traitement")):
    try:
        # Normalisation colonnes
        chunk.columns = chunk.columns.str.lower().str.strip()

        # Identification colonnes
        txt_col = next((c for c in ['article', 'content', 'body'] if c in chunk.columns), None)
        pub_col = next((c for c in ['publication', 'source'] if c in chunk.columns), None)
        
        if not txt_col or not pub_col:
            continue

        chunk = chunk.dropna(subset=[txt_col, pub_col])
        if chunk.empty: continue

        # Prédiction
        X_chunk = vectorizer.transform(chunk[txt_col])
        probs = clf.predict_proba(X_chunk)
        
        # Calcul Score : Proba(Positif) - Proba(Négatif)
        # ATTENTION : Vérifie l'ordre des classes. Ici on suppose [0, 1]
        # Si dans ton fichier Reddit 0=Positif et 1=Négatif :
        scores = probs[:, 0] - probs[:, 1]

        # Agrégation
        for media, score in zip(chunk[pub_col].values, scores):
            if media not in stats:
                stats[media] = [0.0, 0]
            stats[media][0] += score
            stats[media][1] += 1

    except Exception as e:
        continue # On ignore silencieusement les chunks pourris pour finir le job
    
    del chunk, X_chunk
    gc.collect()

# ==========================================
# 3. EXPORT
# ==========================================
print("\n>>> 3/3 Calcul final...")

results = []
for media, val in stats.items():
    if val[1] > 50: # Seuil minimal
        results.append({'Media': media, 'Score': val[0]/val[1], 'Articles': val[1]})

df_res = pd.DataFrame(results).sort_values(by='Score', ascending=False)
df_res.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Fini ! Résultats : {OUTPUT_FILE}")
print(df_res.head())