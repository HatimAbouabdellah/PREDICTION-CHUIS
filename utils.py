"""
Utilitaires pour le traitement des données de trésorerie
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(excel_file):
    """
    Charge et nettoie les données à partir d'un fichier Excel.
    
    Args:
        excel_file: Fichier Excel uploadé via Streamlit
        
    Returns:
        tuple: (df_enc, df_dec, df_tgr) - DataFrames pour encaissements, décaissements et TGR
    """
    try:
        # Validation du fichier
        if not excel_file:
            raise ValueError("Aucun fichier Excel n'a été chargé.")
        
        # Chargement multi-fichiers avec gestion des erreurs
        try:
            df_flux = pd.read_excel(excel_file, sheet_name="31-12-24")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de la feuille '31-12-24': {e}")
        
        try:
            df_tgr = pd.read_excel(excel_file, sheet_name="TGR 31-12-2024", header=None)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de la feuille 'TGR 31-12-2024': {e}")
        
        # Nettoyage des données de flux
        try:
            dates = pd.to_datetime(df_flux.columns[1:-1])
            encaissements = df_flux.iloc[0, 1:-1].astype(float)
            decaissements = df_flux.iloc[1, 1:-1].astype(float)
        except Exception as e:
            raise ValueError(f"Erreur lors du traitement des données de flux : {e}")
        
        # Gestion des valeurs nulles et doublons
        encaissements = pd.Series(encaissements).fillna(method='ffill').fillna(0).values
        decaissements = pd.Series(decaissements).fillna(method='ffill').fillna(0).values
        
        df_enc = pd.DataFrame({'ds': dates, 'y_enc': encaissements})
        df_dec = pd.DataFrame({'ds': dates, 'y_dec': decaissements})
        
        # Nettoyage et préparation des données TGR
        try:
            # Gérer les colonnes dynamiquement
            df_tgr.columns = [str(col).strip() for col in df_tgr.iloc[1]]
            df_tgr = df_tgr[1:].reset_index(drop=True)
            
            # Convertir et nettoyer les dates - utiliser le nom de colonne correct
            date_col = "Dated'opération"
            if date_col not in df_tgr.columns:
                # Essayer de trouver la colonne de date
                date_candidates = [col for col in df_tgr.columns if 'date' in col.lower() or 'opération' in col.lower()]
                if date_candidates:
                    date_col = date_candidates[0]
                else:
                    raise ValueError("Colonne de date non trouvée dans les données TGR")
            
            df_tgr[date_col] = pd.to_datetime(df_tgr[date_col], errors='coerce')
            df_tgr = df_tgr.dropna(subset=[date_col])
            
            # Identifier les colonnes de débit et crédit
            debit_col = "Débit(MAD)" if "Débit(MAD)" in df_tgr.columns else "Débit"
            credit_col = "Crédit(MAD)" if "Crédit(MAD)" in df_tgr.columns else "Crédit"
            
            # Convertir les colonnes numériques
            numeric_columns = [debit_col, credit_col]
            for col in numeric_columns:
                if col in df_tgr.columns:
                    df_tgr[col] = pd.to_numeric(df_tgr[col], errors='coerce').fillna(0)
                else:
                    df_tgr[col] = 0
        except Exception as e:
            raise ValueError(f"Erreur lors du traitement des données TGR : {e}")
        
        # Fusion des données
        try:
            # Agréger les données TGR par mois
            df_tgr_monthly = df_tgr.groupby(pd.Grouper(key=date_col, freq='MS')).agg({
                debit_col: 'sum',
                credit_col: 'sum'
            }).reset_index()
            
            # Aligner les dates
            df_tgr_monthly['ds'] = df_tgr_monthly[date_col]
            
            # Fusionner avec les données de flux
            df_enc_merged = pd.merge(df_enc, df_tgr_monthly[['ds', credit_col]], on='ds', how='left')
            df_dec_merged = pd.merge(df_dec, df_tgr_monthly[['ds', debit_col]], on='ds', how='left')
            
            # Remplacer les valeurs nulles
            df_enc_merged[credit_col] = df_enc_merged[credit_col].fillna(df_enc_merged['y_enc'])
            df_dec_merged[debit_col] = df_dec_merged[debit_col].fillna(df_dec_merged['y_dec'])
            
            # Mettre à jour les colonnes de valeurs
            df_enc_merged['y_enc'] = (df_enc_merged['y_enc'] + df_enc_merged[credit_col]) / 2
            df_dec_merged['y_dec'] = (df_dec_merged['y_dec'] + df_dec_merged[debit_col]) / 2
            
            # Sélectionner les colonnes finales
            df_enc_final = df_enc_merged[['ds', 'y_enc']]
            df_dec_final = df_dec_merged[['ds', 'y_dec']]
        except Exception as e:
            raise ValueError(f"Erreur lors de la fusion des données : {e}")
        
        return df_enc_final, df_dec_final, df_tgr
    
    except Exception as e:
        raise ValueError(f"Erreur générale lors du chargement des données : {e}")

def prepare_lstm_data(series, n_steps=6):
    """
    Prépare les données pour l'entraînement du modèle LSTM avec gestion améliorée des données.
    
    Args:
        series: Série temporelle à préparer
        n_steps: Nombre de pas de temps pour la séquence d'entrée
        
    Returns:
        tuple: (X, y, scaler, scaled_data) - Données préparées pour LSTM
    """
    print(f"Préparation des données LSTM avec n_steps={n_steps}...")
    
    # Vérification des données d'entrée
    if len(series) < n_steps + 1:
        print(f"Attention: La série est trop courte ({len(series)} points) pour n_steps={n_steps}")
        # Ajuster n_steps si nécessaire
        n_steps = max(1, len(series) - 1)
        print(f"Ajustement de n_steps à {n_steps}")
    
    # Vérification des valeurs manquantes ou infinies
    if series.isna().any() or np.isinf(series).any():
        print("Attention: La série contient des valeurs NaN ou Inf. Nettoyage en cours...")
        # Remplacer les valeurs manquantes par interpolation
        clean_series = series.copy()
        clean_series = clean_series.interpolate(method='linear').ffill().bfill()
        # Remplacer les valeurs infinies par la moyenne
        if np.isinf(clean_series).any():
            mean_val = clean_series[~np.isinf(clean_series)].mean()
            clean_series[np.isinf(clean_series)] = mean_val
        series = clean_series
        print("Nettoyage terminé.")
    
    # Normalisation avec gestion des erreurs
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        print(f"Normalisation effectuée. Plage des données: [{np.min(scaled_data):.4f}, {np.max(scaled_data):.4f}]")
    except Exception as e:
        print(f"Erreur lors de la normalisation: {e}")
        print("Utilisation d'une normalisation alternative...")
        # Normalisation alternative en cas d'erreur
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:  # Éviter la division par zéro
            scaled_data = np.zeros_like(series.values.reshape(-1, 1))
        else:
            scaled_data = (series.values.reshape(-1, 1) - min_val) / (max_val - min_val)
        
        # Créer un scaler manuel pour pouvoir inverser la transformation plus tard
        class ManualScaler:
            def __init__(self, min_val, max_val):
                self.min_val = min_val
                self.max_val = max_val
            
            def inverse_transform(self, data):
                return data * (self.max_val - self.min_val) + self.min_val
        
        scaler = ManualScaler(min_val, max_val)
    
    # Création des séquences avec augmentation de données
    X, y = [], []
    
    # Séquences standard
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i+n_steps, 0])
        y.append(scaled_data[i+n_steps, 0])
    
    # Augmentation des données si nous avons peu de points
    if len(X) < 10 and len(scaled_data) >= n_steps + 1:
        print("Peu de données disponibles. Augmentation des données en cours...")
        # Ajouter des séquences avec un léger bruit pour augmenter le jeu de données
        for _ in range(min(20, 100 - len(X))):
            idx = np.random.randint(0, len(scaled_data) - n_steps)
            noise = np.random.normal(0, 0.01, n_steps)  # Bruit gaussien faible
            seq = scaled_data[idx:idx+n_steps, 0] + noise
            seq = np.clip(seq, 0, 1)  # Garder les valeurs dans [0, 1]
            X.append(seq)
            y.append(scaled_data[idx+n_steps, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Vérification finale des données
    if len(X) == 0:
        print("Attention: Impossible de créer des séquences. Création de données synthétiques...")
        # Créer des données synthétiques si nous n'avons pas pu créer de séquences
        X = np.random.rand(10, n_steps, 1)
        y = np.random.rand(10)
        print("Données synthétiques créées pour permettre l'entraînement du modèle.")
    else:
        # Reshape pour le format LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"Préparation terminée. Données d'entraînement: X={X.shape}, y={y.shape}")
    return X, y, scaler, scaled_data.flatten()

def calculate_financial_metrics(df_enc, df_dec):
    """
    Calcule les métriques financières à partir des données d'encaissements et de décaissements.
    
    Args:
        df_enc: DataFrame des encaissements
        df_dec: DataFrame des décaissements
        
    Returns:
        dict: Dictionnaire des métriques financières
    """
    # Calcul des statistiques de base
    enc_mean = np.mean(df_enc['y_enc'])
    dec_mean = np.mean(df_dec['y_dec'])
    solde_mean = enc_mean - dec_mean
    
    # Calcul des tendances (avec vérification pour éviter les divisions par zéro)
    try:
        enc_trend = (np.mean(df_enc['y_enc'][-3:]) / np.mean(df_enc['y_enc'][:3]) - 1) * 100 if len(df_enc) >= 6 else 0
        dec_trend = (np.mean(df_dec['y_dec'][-3:]) / np.mean(df_dec['y_dec'][:3]) - 1) * 100 if len(df_dec) >= 6 else 0
    except Exception:
        enc_trend = 0
        dec_trend = 0
    
    # Calcul des ratios financiers
    try:
        ratio_couverture = enc_mean / dec_mean if dec_mean > 0 else 0
        
        # Calcul des taux de croissance
        enc_growth = (df_enc['y_enc'].iloc[-1] / df_enc['y_enc'].iloc[0] - 1) * 100 if df_enc['y_enc'].iloc[0] > 0 else 0
        dec_growth = (df_dec['y_dec'].iloc[-1] / df_dec['y_dec'].iloc[0] - 1) * 100 if df_dec['y_dec'].iloc[0] > 0 else 0
        
        # Calcul de la volatilité (écart-type normalisé)
        enc_volatility = np.std(df_enc['y_enc']) / enc_mean * 100 if enc_mean > 0 else 0
        dec_volatility = np.std(df_dec['y_dec']) / dec_mean * 100 if dec_mean > 0 else 0
        
        # Indice de stabilité (inverse de la volatilité, normalisé entre 0 et 1)
        stability_index = 1 / (1 + (enc_volatility + dec_volatility) / 200)
        
        # Marge de sécurité (en %)
        safety_margin = (enc_mean - dec_mean) / dec_mean * 100 if dec_mean > 0 else 0
        
    except Exception:
        ratio_couverture = 0
        enc_growth = 0
        dec_growth = 0
        enc_volatility = 0
        dec_volatility = 0
        stability_index = 0
        safety_margin = 0
    
    # Création du dictionnaire de métriques
    metrics = {
        'enc_mean': enc_mean,
        'dec_mean': dec_mean,
        'solde_mean': solde_mean,
        'enc_trend': enc_trend,
        'dec_trend': dec_trend,
        'Ratio de Couverture': ratio_couverture,
        'Taux de Croissance Encaissements': enc_growth,
        'Taux de Croissance Décaissements': dec_growth,
        'Volatilité Encaissements (%)': enc_volatility,
        'Volatilité Décaissements (%)': dec_volatility,
        'Indice de Stabilité': stability_index,
        'Marge de Sécurité (%)': safety_margin
    }
    
    return metrics
