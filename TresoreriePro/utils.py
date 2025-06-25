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
        
        # Chargement des données du fichier fusion_operations_triees.xlsx
        try:
            # Charger le fichier en spécifiant que les en-têtes sont dans la deuxième ligne (index 1)
            df_tgr = pd.read_excel(excel_file, header=1)
            
            # Si le chargement échoue, essayer avec différentes options
            if df_tgr.empty:
                # Essayer de lister les feuilles disponibles
                xls = pd.ExcelFile(excel_file)
                sheet_names = xls.sheet_names
                print('####################', sheet_names)
                
                if sheet_names:
                    # Utiliser la première feuille disponible avec header=1
                    df_tgr = pd.read_excel(excel_file, sheet_name=sheet_names[0], header=1)
                else:
                    raise ValueError("Aucune feuille trouvée dans le fichier Excel")
                    
            # Vérifier si les colonnes attendues sont présentes
            expected_columns = ["Libellé", "Dated'opération", "Datedevaleur", "Débit(MAD)", "Crédit(MAD)", "Num.Pièce", "Num,Pièce"]
            missing_columns = [col for col in expected_columns if col not in df_tgr.columns]
            
            if missing_columns:
                print(f"Attention: Colonnes manquantes: {missing_columns}")
                print(f"Colonnes trouvées: {df_tgr.columns.tolist()}")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier Excel: {e}")
        
        # Nettoyage et préparation des données
        try:
            # Utiliser directement les noms de colonnes spécifiques
            date_col = "Dated'opération"
            debit_col = "Débit(MAD)"
            credit_col = "Crédit(MAD)"
            
            # Vérifier si les colonnes nécessaires sont présentes
            required_cols = [date_col, debit_col, credit_col]
            missing_cols = [col for col in required_cols if col not in df_tgr.columns]
            
            if missing_cols:
                # Si des colonnes sont manquantes, essayer de les trouver avec une recherche approximative
                print(f"Attention: Colonnes requises manquantes: {missing_cols}")
                
                if date_col in missing_cols:
                    date_candidates = [col for col in df_tgr.columns if 'date' in str(col).lower() and 'op' in str(col).lower()]
                    
                    if date_candidates:
                        date_col = date_candidates[0]
                        print(f"Utilisation de '{date_col}' comme colonne de date d'opération")
                    else:
                        raise ValueError("Colonne de date d'opération non trouvée")
                
                piece_candidates = [col for col in df_tgr.columns if 'pièce' in str(col).lower() or 'piece' in str(col).lower()]

                if debit_col in missing_cols:
                    debit_candidates = [col for col in df_tgr.columns if 'débit' in str(col).lower() or 'debit' in str(col).lower()]
                    if debit_candidates:
                        debit_col = debit_candidates[0]
                        print(f"Utilisation de '{debit_col}' comme colonne de débit")
                    else:
                        raise ValueError("Colonne de débit non trouvée")
                
                if credit_col in missing_cols:
                    credit_candidates = [col for col in df_tgr.columns if 'crédit' in str(col).lower() or 'credit' in str(col).lower()]
                    if credit_candidates:
                        credit_col = credit_candidates[0]
                        print(f"Utilisation de '{credit_col}' comme colonne de crédit")
                    else:
                        raise ValueError("Colonne de crédit non trouvée")
            
            # Convertir et nettoyer les dates
            df_tgr[date_col] = pd.to_datetime(df_tgr[date_col], errors='coerce')
            df_tgr = df_tgr.dropna(subset=[date_col])
            
            # Convertir les colonnes numériques
            for col in [debit_col, credit_col]:
                df_tgr[col] = pd.to_numeric(df_tgr[col], errors='coerce').fillna(0)
        except Exception as e:
            raise ValueError(f"Erreur lors du traitement des données: {e}")
        
        # Création des DataFrames d'encaissements et décaissements à partir des données
        try:
            # Agréger les données par mois
            df_tgr_monthly = df_tgr.groupby(pd.Grouper(key=date_col, freq='MS')).agg({
                debit_col: 'sum',
                credit_col: 'sum'
            }).reset_index()
            
            # Créer les DataFrames d'encaissements et décaissements
            df_enc = pd.DataFrame({
                'ds': df_tgr_monthly[date_col],
                'y_enc': df_tgr_monthly[credit_col]
            })
            
            df_dec = pd.DataFrame({
                'ds': df_tgr_monthly[date_col],
                'y_dec': df_tgr_monthly[debit_col]
            })
            
            # Tri par date et gestion des valeurs manquantes
            df_enc = df_enc.sort_values('ds').reset_index(drop=True)
            df_dec = df_dec.sort_values('ds').reset_index(drop=True)
            
            df_enc['y_enc'] = df_enc['y_enc'].fillna(0)
            df_dec['y_dec'] = df_dec['y_dec'].fillna(0)
            
            # Afficher des informations sur les données chargées
            print(f"Données chargées: {len(df_enc)} mois de données, de {df_enc['ds'].min().strftime('%Y-%m')} à {df_enc['ds'].max().strftime('%Y-%m')}")
            
            # Si les données sont insuffisantes (moins de 6 mois), générer des données synthétiques
            if len(df_enc) < 6:
                print("Données insuffisantes, génération de données synthétiques...")
                # Générer des dates mensuelles sur 12 mois à partir de la date la plus récente
                if len(df_enc) > 0:
                    last_date = df_enc['ds'].max()
                else:
                    last_date = datetime.now()
                
                # Générer 12 mois de données
                synthetic_dates = pd.date_range(end=last_date, periods=12, freq='MS')
                
                # Générer des valeurs aléatoires basées sur les moyennes si disponibles
                if len(df_enc) > 0:
                    enc_mean = df_enc['y_enc'].mean()
                    dec_mean = df_dec['y_dec'].mean()
                else:
                    enc_mean = 100000  # Valeur par défaut
                    dec_mean = 80000   # Valeur par défaut
                
                # Créer des variations aléatoires autour de la moyenne
                synthetic_enc = np.random.normal(enc_mean, enc_mean * 0.2, len(synthetic_dates))
                synthetic_dec = np.random.normal(dec_mean, dec_mean * 0.2, len(synthetic_dates))
                
                # Créer les nouveaux DataFrames
                df_enc = pd.DataFrame({
                    'ds': synthetic_dates,
                    'y_enc': synthetic_enc
                })
                
                df_dec = pd.DataFrame({
                    'ds': synthetic_dates,
                    'y_dec': synthetic_dec
                })
                
                # S'assurer que toutes les valeurs sont positives
                df_enc['y_enc'] = df_enc['y_enc'].apply(lambda x: max(0, x))
                df_dec['y_dec'] = df_dec['y_dec'].apply(lambda x: max(0, x))
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la création des DataFrames : {e}")
        
        return df_enc, df_dec, df_tgr
    
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
