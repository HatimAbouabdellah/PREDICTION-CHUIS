"""
Modèles de prévision pour l'analyse de trésorerie
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

class ForecastingModels:
    """Classe pour gérer les modèles de prévision de trésorerie"""
    
    def __init__(self, config=None):
        """
        Initialise les modèles de prévision avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration des modèles à utiliser et leurs paramètres
        """
        self.config = config or {}
        self.models = {}
        self.seasonal_patterns = {}
        self.anomalies = {}
        
    def analyze_seasonality(self, df):
        """
        Détecte automatiquement les tendances saisonnières dans les données.
        
        Args:
            df (DataFrame): Données à analyser avec colonnes 'ds' et 'y'
            
        Returns:
            dict: Dictionnaire contenant les composantes saisonnières
        """
        try:
            # Vérifier si nous avons assez de données pour l'analyse saisonnière
            if len(df) < 12:
                return {'has_seasonality': False, 'message': 'Pas assez de données pour détecter la saisonnalité'}
                
            # Convertir en série temporelle indexée par date
            ts = df.set_index('ds')['y']
            
            # Décomposition saisonnière
            result = seasonal_decompose(ts, model='additive', period=12)  # Période de 12 mois par défaut
            
            # Tester la significativité de la saisonnalité
            seasonal_strength = np.std(result.seasonal) / np.std(result.resid)
            has_seasonality = seasonal_strength > 0.3  # Seuil arbitraire pour déterminer si la saisonnalité est significative
            
            # Déterminer la période dominante
            if has_seasonality:
                # Calculer l'autocorrélation pour identifier la période
                acf = np.correlate(result.seasonal, result.seasonal, mode='full')
                acf = acf[len(acf)//2:]
                # Trouver les pics dans l'autocorrélation
                peaks = np.where((acf[1:-1] > acf[0:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
                if len(peaks) > 0:
                    dominant_period = peaks[0]
                else:
                    dominant_period = 12  # Par défaut
            else:
                dominant_period = None
                
            return {
                'has_seasonality': has_seasonality,
                'seasonal_strength': seasonal_strength,
                'dominant_period': dominant_period,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'resid': result.resid
            }
        except Exception as e:
            print(f"Erreur lors de l'analyse de saisonnalité: {e}")
            return {'has_seasonality': False, 'error': str(e)}
    
    def detect_anomalies(self, df, column='y', method='zscore', threshold=3.0):
        """
        Détecte les valeurs aberrantes dans les données.
        
        Args:
            df (DataFrame): Données à analyser
            column (str): Colonne à analyser
            method (str): Méthode de détection ('zscore', 'iqr')
            threshold (float): Seuil pour la détection
            
        Returns:
            DataFrame: DataFrame avec une colonne supplémentaire indiquant les anomalies
        """
        try:
            # Copie du DataFrame pour ne pas modifier l'original
            result_df = df.copy()
            
            # Méthode Z-score
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(result_df[column]))
                result_df['is_anomaly'] = z_scores > threshold
                result_df['anomaly_score'] = z_scores
            
            # Méthode IQR (Interquartile Range)
            elif method == 'iqr':
                Q1 = result_df[column].quantile(0.25)
                Q3 = result_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result_df['is_anomaly'] = (result_df[column] < lower_bound) | (result_df[column] > upper_bound)
                # Calculer un score d'anomalie basé sur la distance aux limites
                result_df['anomaly_score'] = result_df[column].apply(
                    lambda x: max(0, abs(x - lower_bound) / IQR) if x < lower_bound else 
                              max(0, abs(x - upper_bound) / IQR) if x > upper_bound else 0
                )
            
            # Stocker les anomalies détectées
            anomalies = result_df[result_df['is_anomaly']].copy()
            if len(anomalies) > 0:
                return {
                    'anomalies_detected': True,
                    'anomaly_count': len(anomalies),
                    'anomaly_percent': (len(anomalies) / len(result_df)) * 100,
                    'anomaly_data': anomalies,
                    'all_data': result_df
                }
            else:
                return {
                    'anomalies_detected': False,
                    'anomaly_count': 0,
                    'all_data': result_df
                }
        except Exception as e:
            print(f"Erreur lors de la détection d'anomalies: {e}")
            return {'anomalies_detected': False, 'error': str(e)}
    
    def train_models(self, df_enc, df_dec, n_mois):
        """
        Entraîne les modèles sélectionnés sur les données fournies.
        
        Args:
            df_enc (DataFrame): Données d'encaissements
            df_dec (DataFrame): Données de décaissements
            n_mois (int): Nombre de mois à prévoir
            
        Returns:
            dict: Dictionnaire des modèles entraînés
        """
        # Analyser la saisonnalité des données
        self.seasonal_patterns['enc'] = self.analyze_seasonality(df_enc.rename(columns={'y_enc': 'y'}))
        self.seasonal_patterns['dec'] = self.analyze_seasonality(df_dec.rename(columns={'y_dec': 'y'}))
        
        # Détecter les anomalies
        self.anomalies['enc'] = self.detect_anomalies(df_enc, column='y_enc')
        self.anomalies['dec'] = self.detect_anomalies(df_dec, column='y_dec')
        models = {}
        
        # Récupérer les paramètres de configuration
        use_prophet = self.config.get('use_prophet', True)
        use_arima = self.config.get('use_arima', True)
        use_lstm = self.config.get('use_lstm', True)
        use_xgboost = self.config.get('use_xgboost', True)
        use_rf = self.config.get('use_rf', True)
        use_hybrid = self.config.get('use_hybrid', True)
        confidence_interval = self.config.get('confidence_interval', 95)
        use_cross_validation = self.config.get('use_cross_validation', False)
        
        # Prepare Prophet dataframes
        df_enc_prophet = df_enc.rename(columns={'y_enc': 'y'})
        df_dec_prophet = df_dec.rename(columns={'y_dec': 'y'})
        
        # Prophet (si sélectionné)
        if use_prophet:
            try:
                # Configurer Prophet avec l'intervalle de confiance
                models['prophet_enc'] = Prophet(interval_width=confidence_interval/100)
                
                # Ajouter des régresseurs saisonniers si la saisonnalité est détectée
                if self.seasonal_patterns['enc'].get('has_seasonality', False):
                    models['prophet_enc'].add_seasonality(
                        name='custom_seasonal', 
                        period=self.seasonal_patterns['enc'].get('dominant_period', 12),
                        fourier_order=5
                    )
                
                models['prophet_enc'].fit(df_enc_prophet)
                
                models['prophet_dec'] = Prophet(interval_width=confidence_interval/100)
                
                # Ajouter des régresseurs saisonniers si la saisonnalité est détectée
                if self.seasonal_patterns['dec'].get('has_seasonality', False):
                    models['prophet_dec'].add_seasonality(
                        name='custom_seasonal', 
                        period=self.seasonal_patterns['dec'].get('dominant_period', 12),
                        fourier_order=5
                    )
                    
                models['prophet_dec'].fit(df_dec_prophet)
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle Prophet: {e}")
        
        # ARIMA/SARIMA (si sélectionné)
        if use_arima:
            try:
                # Si la saisonnalité est détectée, utiliser SARIMA au lieu d'ARIMA
                if self.seasonal_patterns['enc'].get('has_seasonality', False):
                    # Utiliser un modèle SARIMA avec composante saisonnière
                    period = self.seasonal_patterns['enc'].get('dominant_period', 12)
                    models['arima_enc'] = ARIMA(df_enc['y_enc'], 
                                               order=(5,1,0), 
                                               seasonal_order=(1,1,1,period))
                else:
                    models['arima_enc'] = ARIMA(df_enc['y_enc'], order=(5,1,0))
                models['arima_enc'] = models['arima_enc'].fit()
                
                if self.seasonal_patterns['dec'].get('has_seasonality', False):
                    # Utiliser un modèle SARIMA avec composante saisonnière
                    period = self.seasonal_patterns['dec'].get('dominant_period', 12)
                    models['arima_dec'] = ARIMA(df_dec['y_dec'], 
                                               order=(5,1,0), 
                                               seasonal_order=(1,1,1,period))
                else:
                    models['arima_dec'] = ARIMA(df_dec['y_dec'], order=(5,1,0))
                models['arima_dec'] = models['arima_dec'].fit()
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle ARIMA: {e}")
        
        # LSTM (si sélectionné)
        if use_lstm:
            try:
                print("Début de l'entraînement du modèle LSTM...")
                # Vérifier si nous avons assez de données
                min_data_points = 12  # Augmentation du minimum de points de données pour un meilleur apprentissage
                
                # Vérifier si nous avons assez de données, sinon générer des données synthétiques
                if len(df_enc) < min_data_points:
                    print(f"Attention: Pas assez de données pour un modèle LSTM optimal pour les encaissements. {len(df_enc)} points disponibles, minimum recommandé: {min_data_points}")
                    
                    if len(df_enc) < 5:
                        print("Génération de données synthétiques pour les encaissements...")
                        # Générer des données synthétiques pour compléter
                        if len(df_enc) > 0:
                            # Utiliser la moyenne et l'écart-type des données existantes
                            enc_mean = df_enc['y_enc'].mean()
                            enc_std = max(df_enc['y_enc'].std(), enc_mean * 0.1)  # Éviter un écart-type trop faible
                        else:
                            # Valeurs par défaut si aucune donnée n'est disponible
                            enc_mean, enc_std = 1000, 100
                        
                        # Générer des données synthétiques
                        synthetic_count = max(12, 5 - len(df_enc))
                        synthetic_enc = np.random.normal(enc_mean, enc_std, synthetic_count)
                        
                        # Créer un DataFrame synthétique
                        last_date = df_enc['ds'].iloc[-1] if len(df_enc) > 0 else pd.Timestamp.now().replace(day=1)
                        synthetic_dates = pd.date_range(end=last_date, periods=synthetic_count+1, freq='MS')[:-1]
                        synthetic_df_enc = pd.DataFrame({
                            'ds': synthetic_dates,
                            'y_enc': synthetic_enc
                        })
                        
                        # Concaténer avec les données existantes
                        df_enc = pd.concat([synthetic_df_enc, df_enc]).reset_index(drop=True)
                        print(f"Données synthétiques ajoutées. Nombre total de points pour les encaissements: {len(df_enc)}")
                
                if len(df_dec) < min_data_points:
                    print(f"Attention: Pas assez de données pour un modèle LSTM optimal pour les décaissements. {len(df_dec)} points disponibles, minimum recommandé: {min_data_points}")
                    
                    if len(df_dec) < 5:
                        print("Génération de données synthétiques pour les décaissements...")
                        # Générer des données synthétiques pour compléter
                        if len(df_dec) > 0:
                            # Utiliser la moyenne et l'écart-type des données existantes
                            dec_mean = df_dec['y_dec'].mean()
                            dec_std = max(df_dec['y_dec'].std(), dec_mean * 0.1)  # Éviter un écart-type trop faible
                        else:
                            # Valeurs par défaut si aucune donnée n'est disponible
                            dec_mean, dec_std = 800, 80
                        
                        # Générer des données synthétiques
                        synthetic_count = max(12, 5 - len(df_dec))
                        synthetic_dec = np.random.normal(dec_mean, dec_std, synthetic_count)
                        
                        # Créer un DataFrame synthétique
                        last_date = df_dec['ds'].iloc[-1] if len(df_dec) > 0 else pd.Timestamp.now().replace(day=1)
                        synthetic_dates = pd.date_range(end=last_date, periods=synthetic_count+1, freq='MS')[:-1]
                        synthetic_df_dec = pd.DataFrame({
                            'ds': synthetic_dates,
                            'y_dec': synthetic_dec
                        })
                        
                        # Concaténer avec les données existantes
                        df_dec = pd.concat([synthetic_df_dec, df_dec]).reset_index(drop=True)
                        print(f"Données synthétiques ajoutées. Nombre total de points pour les décaissements: {len(df_dec)}")
                
                # Vérifier à nouveau si nous avons assez de données après l'ajout des données synthétiques
                if len(df_enc) < 5 or len(df_dec) < 5:
                    raise ValueError("Pas assez de données pour entraîner le modèle LSTM (minimum 5 points)")
                    
                print(f"Entraînement du modèle LSTM avec {len(df_enc)} points pour les encaissements et {len(df_dec)} points pour les décaissements.")
                
                # Importation de prepare_lstm_data
                try:
                    from utils import prepare_lstm_data  # Essai d'importation directe
                except ImportError:
                    try:
                        from .utils import prepare_lstm_data  # Essai d'importation relative
                    except ImportError:
                        import sys
                        print(f"Chemins de recherche Python: {sys.path}")
                        raise ImportError("Impossible d'importer prepare_lstm_data. Vérifiez la structure du projet.")
                
                # Importation des modules nécessaires pour LSTM
                from tensorflow.keras.optimizers import Adam
                from tensorflow.keras.callbacks import EarlyStopping
                
                # Création d'un early stopping commun
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                
                # LSTM pour les encaissements (dans un bloc try séparé)
                enc_lstm_success = False
                try:
                    print("\n--- Entraînement du modèle LSTM pour les encaissements ---")
                    print("Préparation des données pour LSTM encaissements...")
                    X_enc, y_enc, scaler_enc, scaled_enc = prepare_lstm_data(df_enc['y_enc'], n_steps=6)
                    print(f"Forme des données LSTM encaissements: X={X_enc.shape}, y={y_enc.shape}")
                    
                    # Configuration du modèle avec régularisation pour éviter le surapprentissage
                    model_enc = Sequential()
                    model_enc.add(LSTM(64, return_sequences=True, input_shape=(X_enc.shape[1], 1), 
                                     dropout=0.2, recurrent_dropout=0.2))  # Ajout de dropout
                    model_enc.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # Couche plus petite avec dropout
                    model_enc.add(Dense(16, activation='relu'))  # Couche dense intermédiaire
                    model_enc.add(Dense(1))
                    
                    # Utilisation d'un optimiseur avec taux d'apprentissage réduit
                    optimizer = Adam(learning_rate=0.001)
                    model_enc.compile(optimizer=optimizer, loss='mse')
                    
                    print("Entraînement du modèle LSTM encaissements...")
                    history_enc = model_enc.fit(X_enc, y_enc, epochs=100, batch_size=16, 
                                             callbacks=[early_stopping], verbose=0)
                    print(f"Entraînement terminé après {len(history_enc.history['loss'])} époques")
                    print(f"Perte finale: {history_enc.history['loss'][-1]:.4f}")
                    
                    models['lstm_enc_model'] = model_enc
                    models['lstm_enc_scaler'] = scaler_enc
                    models['scaled_enc'] = scaled_enc
                    enc_lstm_success = True
                    print("Modèle LSTM pour les encaissements entraîné avec succès.")
                except Exception as e:
                    import traceback
                    print(f"Erreur lors de l'entraînement du modèle LSTM pour les encaissements: {e}")
                    print(traceback.format_exc())
                    print("Le modèle LSTM pour les encaissements ne sera pas disponible.")
                
                # LSTM pour les décaissements (dans un bloc try séparé)
                dec_lstm_success = False
                try:
                    print("\n--- Entraînement du modèle LSTM pour les décaissements ---")
                    print("Préparation des données pour LSTM décaissements...")
                    X_dec, y_dec, scaler_dec, scaled_dec = prepare_lstm_data(df_dec['y_dec'], n_steps=6)
                    print(f"Forme des données LSTM décaissements: X={X_dec.shape}, y={y_dec.shape}")
                    
                    # Configuration du modèle avec architecture identique à celle des encaissements
                    model_dec = Sequential()
                    model_dec.add(LSTM(64, return_sequences=True, input_shape=(X_dec.shape[1], 1), 
                                     dropout=0.2, recurrent_dropout=0.2))
                    model_dec.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
                    model_dec.add(Dense(16, activation='relu'))
                    model_dec.add(Dense(1))
                    
                    optimizer = Adam(learning_rate=0.001)
                    model_dec.compile(optimizer=optimizer, loss='mse')
                    
                    print("Entraînement du modèle LSTM décaissements...")
                    history_dec = model_dec.fit(X_dec, y_dec, epochs=100, batch_size=16, 
                                             callbacks=[early_stopping], verbose=0)
                    print(f"Entraînement terminé après {len(history_dec.history['loss'])} époques")
                    print(f"Perte finale: {history_dec.history['loss'][-1]:.4f}")
                    
                    models['lstm_dec_model'] = model_dec
                    models['lstm_dec_scaler'] = scaler_dec
                    models['scaled_dec'] = scaled_dec
                    dec_lstm_success = True
                    print("Modèle LSTM pour les décaissements entraîné avec succès.")
                except Exception as e:
                    import traceback
                    print(f"Erreur lors de l'entraînement du modèle LSTM pour les décaissements: {e}")
                    print(traceback.format_exc())
                    print("Le modèle LSTM pour les décaissements ne sera pas disponible.")
                
                # Si l'un des modèles a échoué mais pas l'autre, créer un modèle de remplacement
                if enc_lstm_success and not dec_lstm_success:
                    print("\nCréation d'un modèle LSTM de remplacement pour les décaissements basé sur le modèle des encaissements...")
                    try:
                        # Copier le modèle des encaissements pour les décaissements
                        models['lstm_dec_model'] = models['lstm_enc_model']
                        models['lstm_dec_scaler'] = models['lstm_enc_scaler']
                        models['scaled_dec'] = models['scaled_enc']
                        print("Modèle de remplacement créé avec succès.")
                    except Exception as e:
                        print(f"Erreur lors de la création du modèle de remplacement: {e}")
                
                elif not enc_lstm_success and dec_lstm_success:
                    print("\nCréation d'un modèle LSTM de remplacement pour les encaissements basé sur le modèle des décaissements...")
                    try:
                        # Copier le modèle des décaissements pour les encaissements
                        models['lstm_enc_model'] = models['lstm_dec_model']
                        models['lstm_enc_scaler'] = models['lstm_dec_scaler']
                        models['scaled_enc'] = models['scaled_dec']
                        print("Modèle de remplacement créé avec succès.")
                    except Exception as e:
                        print(f"Erreur lors de la création du modèle de remplacement: {e}")
                
                # Vérifier si au moins un modèle a été entraîné avec succès
                if enc_lstm_success or dec_lstm_success:
                    print("\nEntraînement des modèles LSTM terminé avec succès (au moins un modèle disponible).")
                else:
                    print("\nAucun modèle LSTM n'a pu être entraîné. Le modèle LSTM ne sera pas disponible pour les prévisions.")
            except Exception as e:
                import traceback
                print(f"Erreur lors de l'entraînement du modèle LSTM: {e}")
                print(f"Détails de l'erreur: {traceback.format_exc()}")
                print("Le modèle LSTM ne sera pas utilisé pour les prévisions.")
        
        # XGBoost (si sélectionné)
        if use_xgboost:
            try:
                # Préparation des données pour XGBoost
                X_enc = np.array(range(len(df_enc))).reshape(-1, 1)
                y_enc = df_enc['y_enc'].values
                
                X_dec = np.array(range(len(df_dec))).reshape(-1, 1)
                y_dec = df_dec['y_dec'].values
                
                # Entraînement des modèles
                models['xgboost_enc'] = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                
                # Validation croisée si activée
                if use_cross_validation:
                    # Utiliser TimeSeriesSplit pour la validation croisée temporelle
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = []
                    
                    for train_index, test_index in tscv.split(X_enc):
                        X_train, X_test = X_enc[train_index], X_enc[test_index]
                        y_train, y_test = y_enc[train_index], y_enc[test_index]
                        
                        # Entraîner sur les données d'entraînement
                        models['xgboost_enc'].fit(X_train, y_train)
                        # Prédire sur les données de test
                        y_pred = models['xgboost_enc'].predict(X_test)
                        # Calculer l'erreur
                        mae = mean_absolute_error(y_test, y_pred)
                        cv_scores.append(mae)
                    
                    # Stocker les scores de validation croisée
                    self.cv_scores = {'xgboost_enc': np.mean(cv_scores)}
                    
                    # Réentraîner sur toutes les données
                    models['xgboost_enc'].fit(X_enc, y_enc)
                else:
                    models['xgboost_enc'].fit(X_enc, y_enc)
                    
                # Entraîner le modèle XGBoost pour les décaissements
                models['xgboost_dec'] = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                models['xgboost_dec'].fit(X_dec, y_dec)
                
                print("Entraînement du modèle XGBoost terminé.")
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle XGBoost: {e}")
        
        # Entraînement du modèle Random Forest
        if self.config.get('use_rf', True):
            try:
                print("Début de l'entraînement du modèle Random Forest...")
                # Préparer les données pour Random Forest
                X_enc = np.array(range(len(df_enc))).reshape(-1, 1)
                y_enc = df_enc['y_enc'].values
                X_dec = np.array(range(len(df_dec))).reshape(-1, 1)
                y_dec = df_dec['y_dec'].values
                
                # Entraîner le modèle Random Forest pour les encaissements
                models['rf_enc'] = RandomForestRegressor(n_estimators=100, random_state=42)
                models['rf_enc'].fit(X_enc, y_enc)
                
                # Entraîner le modèle Random Forest pour les décaissements
                models['rf_dec'] = RandomForestRegressor(n_estimators=100, random_state=42)
                models['rf_dec'].fit(X_dec, y_dec)
                
                print("Entraînement du modèle Random Forest terminé.")
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle Random Forest: {e}")
        
        # Entraînement du modèle Hybride
        if self.config.get('use_hybrid', False):
            try:
                print("Début de l'entraînement du modèle Hybride...")
                # Vérifier que les modèles nécessaires sont disponibles
                if 'xgboost_enc' in models and 'rf_enc' in models:
                    # Préparer les données pour le modèle Hybride
                    X_enc = np.array(range(len(df_enc))).reshape(-1, 1)
                    y_enc = df_enc['y_enc'].values
                    X_dec = np.array(range(len(df_dec))).reshape(-1, 1)
                    y_dec = df_dec['y_dec'].values
                    
                    # Créer un modèle hybride (VotingRegressor) pour les encaissements
                    models['hybrid_enc'] = VotingRegressor([
                        ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100)),
                        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
                    ])
                    models['hybrid_enc'].fit(X_enc, y_enc)
                    
                    # Créer un modèle hybride (VotingRegressor) pour les décaissements
                    models['hybrid_dec'] = VotingRegressor([
                        ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100)),
                        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
                    ])
                    models['hybrid_dec'].fit(X_dec, y_dec)
                    
                    print("Entraînement du modèle Hybride terminé.")
                else:
                    print("Impossible de créer le modèle hybride : XGBoost et Random Forest doivent être activés.")
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle Hybride: {e}")
        
        self.models = models
        return models
    
    def cross_validate_models(self, df, column, models_list, n_splits=5):
        """
        Effectue une validation croisée sur les modèles spécifiés.
        
        Args:
            df (DataFrame): Données à utiliser pour la validation
            column (str): Colonne cible ('y_enc' ou 'y_dec')
            models_list (list): Liste des noms de modèles à valider
            n_splits (int): Nombre de divisions pour la validation croisée
            
        Returns:
            dict: Dictionnaire des scores de validation croisée par modèle
        """
        cv_scores = {}
        
        try:
            # Préparation des données
            X = np.array(range(len(df))).reshape(-1, 1)
            y = df[column].values
            
            # Utiliser TimeSeriesSplit pour la validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for model_name in models_list:
                if 'prophet' in model_name:
                    # Prophet a une interface différente, traitement spécial
                    scores = []
                    for train_index, test_index in tscv.split(X):
                        # Créer un nouveau modèle Prophet pour chaque fold
                        prophet_model = Prophet(interval_width=0.95)
                        
                        # Préparer les données d'entraînement
                        train_df = df.iloc[train_index].copy()
                        train_df = train_df.rename(columns={column: 'y'})
                        
                        # Entraîner le modèle
                        prophet_model.fit(train_df[['ds', 'y']])
                        
                        # Préparer les données de test
                        test_df = df.iloc[test_index].copy()
                        future = prophet_model.make_future_dataframe(periods=0, freq='MS')
                        future = future[future['ds'].isin(test_df['ds'])]
                        
                        # Prédire
                        forecast = prophet_model.predict(future)
                        
                        # Calculer l'erreur
                        y_true = test_df[column].values
                        y_pred = forecast['yhat'].values
                        
                        if len(y_true) > 0 and len(y_pred) > 0:
                            mae = mean_absolute_error(y_true, y_pred)
                            scores.append(mae)
                    
                    if scores:
                        cv_scores[model_name] = np.mean(scores)
                
                elif 'arima' in model_name:
                    # ARIMA a une interface différente
                    scores = []
                    for train_index, test_index in tscv.split(X):
                        try:
                            # Préparer les données d'entraînement
                            train_y = y[train_index]
                            test_y = y[test_index]
                            
                            # Entraîner le modèle
                            arima_model = ARIMA(train_y, order=(5,1,0))
                            arima_model = arima_model.fit()
                            
                            # Prédire
                            predictions = arima_model.forecast(steps=len(test_y))
                            
                            # Calculer l'erreur
                            if len(test_y) > 0 and len(predictions) > 0:
                                mae = mean_absolute_error(test_y, predictions)
                                scores.append(mae)
                        except Exception as e:
                            print(f"Erreur lors de la validation croisée d'ARIMA: {e}")
                    
                    if scores:
                        cv_scores[model_name] = np.mean(scores)
                
                elif any(m in model_name for m in ['xgboost', 'rf', 'hybrid']):
                    # Modèles scikit-learn compatibles
                    scores = []
                    for train_index, test_index in tscv.split(X):
                        # Préparer les données
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # Créer et entraîner le modèle
                        if 'xgboost' in model_name:
                            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                        elif 'rf' in model_name:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif 'hybrid' in model_name and not 'advanced' in model_name:
                            # Modèle hybride simple (VotingRegressor)
                            xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            model = VotingRegressor([('xgb', xgb), ('rf', rf)])
                        else:
                            # Cas non géré, passer au modèle suivant
                            continue
                        
                        model.fit(X_train, y_train)
                    
                    # Prédire
                    y_pred = model.predict(X_test)
                    
                    # Calculer l'erreur
                    if len(y_test) > 0 and len(y_pred) > 0:
                        mae = mean_absolute_error(y_test, y_pred)
                        scores.append(mae)
                
                if scores:
                    cv_scores[model_name] = np.mean(scores)
            
            return cv_scores
        
        except Exception as e:
            print(f"Erreur lors de la validation croisée: {e}")
            return cv_scores
    
    def generate_forecasts(self, df_enc, df_dec, n_mois):
        """
        Génère des prévisions avec tous les modèles entraînés.
        
        Args:
            df_enc (DataFrame): Données d'encaissements
            df_dec (DataFrame): Données de décaissements
            n_mois (int): Nombre de mois à prévoir
            
        Returns:
            dict: Dictionnaire des prévisions par modèle
        """
        forecasts = {}
        
        # Prepare Prophet dataframes
        df_enc_prophet = df_enc.rename(columns={'y_enc': 'y', 'ds': 'ds'})
        df_dec_prophet = df_dec.rename(columns={'y_dec': 'y', 'ds': 'ds'})
        
        # Prophet
        if 'prophet_enc' in self.models and 'prophet_dec' in self.models:
            try:
                future_enc = self.models['prophet_enc'].make_future_dataframe(periods=n_mois, freq='MS')
                future_dec = self.models['prophet_dec'].make_future_dataframe(periods=n_mois, freq='MS')
                
                prophet_enc_forecast = self.models['prophet_enc'].predict(future_enc)['yhat'][-n_mois:].values
                prophet_dec_forecast = self.models['prophet_dec'].predict(future_dec)['yhat'][-n_mois:].values
                
                forecasts['prophet_enc'] = prophet_enc_forecast
                forecasts['prophet_dec'] = prophet_dec_forecast
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions Prophet: {e}")
        
        # ARIMA/SARIMA
        if 'arima_enc' in self.models and 'arima_dec' in self.models:
            try:
                arima_enc_forecast = self.models['arima_enc'].forecast(steps=n_mois)
                arima_dec_forecast = self.models['arima_dec'].forecast(steps=n_mois)
                forecasts['arima_enc'] = arima_enc_forecast
                forecasts['arima_dec'] = arima_dec_forecast
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions ARIMA: {e}")
        
        # LSTM forecasting with enhanced error handling and diagnostics
    def lstm_forecast(self, model, scaler, scaled_data, n_mois=12, n_steps=6):
        """
        Génère des prévisions avec un modèle LSTM.
        
        Args:
            model: Modèle LSTM entraîné
            scaler: Scaler utilisé pour normaliser les données
            scaled_data: Données historiques normalisées
            n_mois: Nombre de mois à prédire
            n_steps: Taille de la fenêtre temporelle utilisée pour l'entraînement
            
        Returns:
            array: Prévisions pour les n_mois suivants
        """
        # Préparation des données pour la prévision
        predictions = []
        curr_batch = scaled_data[-n_steps:].reshape((1, n_steps, 1))
        
        # Générer les prévisions pour chaque mois
        for i in range(n_mois):
            # Prédire la valeur suivante
            curr_pred = model.predict(curr_batch, verbose=0)[0][0]
            predictions.append(curr_pred)
            
            # Mettre à jour le batch pour la prédiction suivante
            # Reshape curr_pred pour avoir la même dimension que curr_batch
            curr_pred_reshaped = np.array([[[curr_pred]]])  # Forme (1, 1, 1)
            curr_batch = np.concatenate((curr_batch[:, 1:, :], curr_pred_reshaped), axis=1)
        
        # Inverse la normalisation pour obtenir les valeurs réelles
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()
        
        return predictions
        
    def generate_forecasts(self, df_enc, df_dec, n_mois):
        """
        Génère des prévisions pour les n_mois suivants en utilisant les modèles entraînés.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            n_mois (int): Nombre de mois à prédire
            
        Returns:
            dict: Dictionnaire des prévisions par modèle
        """
        forecasts = {}
        
        # Vérifier si les composants LSTM pour les encaissements sont disponibles
        enc_lstm_keys = ['lstm_enc_model', 'lstm_enc_scaler', 'scaled_enc']
        enc_lstm_available = all(k in self.models for k in enc_lstm_keys)
        
        # Vérifier si les composants LSTM pour les décaissements sont disponibles
        dec_lstm_keys = ['lstm_dec_model', 'lstm_dec_scaler', 'scaled_dec']
        dec_lstm_available = all(k in self.models for k in dec_lstm_keys)
        
        # Afficher les informations sur les composants disponibles
        if not (enc_lstm_available or dec_lstm_available):
            print("LSTM: Aucun composant LSTM n'est disponible.")
            print(f"Composants disponibles: {[k for k in self.models.keys() if 'lstm' in k]}")
            print("Le modèle LSTM ne sera pas utilisé pour les prévisions.")
        else:
            # Si au moins un des modèles est disponible, générer les prévisions LSTM
            print("\n=== Génération des prévisions LSTM ===\n")
            try:
                lstm_enc_forecast = None
                lstm_dec_forecast = None
                enc_valid = False
                dec_valid = False
                
                # Génération des prévisions d'encaissements si disponible
                if enc_lstm_available:
                    print("Génération des prévisions LSTM pour les encaissements...")
                    try:
                        lstm_enc_forecast = self.lstm_forecast(
                            self.models['lstm_enc_model'], 
                            self.models['lstm_enc_scaler'], 
                            self.models['scaled_enc'], 
                            n_mois=n_mois
                        )
                        enc_valid = np.all(np.isfinite(lstm_enc_forecast))
                        if not enc_valid:
                            print("Les prévisions LSTM d'encaissements contiennent des valeurs non valides (NaN ou infini).")
                    except Exception as e:
                        print(f"Erreur lors de la génération des prévisions LSTM pour les encaissements: {e}")
                else:
                    print("Modèle LSTM pour les encaissements non disponible.")
                
                # Génération des prévisions de décaissements si disponible
                if dec_lstm_available:
                    print("Génération des prévisions LSTM pour les décaissements...")
                    try:
                        lstm_dec_forecast = self.lstm_forecast(
                            self.models['lstm_dec_model'], 
                            self.models['lstm_dec_scaler'], 
                            self.models['scaled_dec'], 
                            n_mois=n_mois
                        )
                        dec_valid = np.all(np.isfinite(lstm_dec_forecast))
                        if not dec_valid:
                            print("Les prévisions LSTM de décaissements contiennent des valeurs non valides (NaN ou infini).")
                    except Exception as e:
                        print(f"Erreur lors de la génération des prévisions LSTM pour les décaissements: {e}")
                else:
                    print("Modèle LSTM pour les décaissements non disponible.")
                
                # Si un modèle est disponible mais pas l'autre, générer des prévisions synthétiques
                if enc_valid and not dec_valid:
                    print("Génération de prévisions LSTM synthétiques pour les décaissements basées sur les encaissements...")
                    # Générer des prévisions de décaissements basées sur les encaissements
                    lstm_dec_forecast = lstm_enc_forecast * 0.8  # Supposer que les décaissements sont environ 80% des encaissements
                    dec_valid = True
                elif dec_valid and not enc_valid:
                    print("Génération de prévisions LSTM synthétiques pour les encaissements basées sur les décaissements...")
                    # Générer des prévisions d'encaissements basées sur les décaissements
                    lstm_enc_forecast = lstm_dec_forecast * 1.25  # Supposer que les encaissements sont environ 125% des décaissements
                    enc_valid = True
                
                # Vérifier si au moins un modèle est valide
                if enc_valid or dec_valid:
                    # Vérification des valeurs négatives
                    if np.any(lstm_enc_forecast < 0):
                        print("Attention: Certaines prévisions d'encaissements sont négatives. Correction en cours...")
                        lstm_enc_forecast = np.maximum(lstm_enc_forecast, 0)  # Remplacer les valeurs négatives par 0
                    
                    if np.any(lstm_dec_forecast < 0):
                        print("Attention: Certaines prévisions de décaissements sont négatives. Correction en cours...")
                        lstm_dec_forecast = np.maximum(lstm_dec_forecast, 0)  # Remplacer les valeurs négatives par 0
                    
                    # Vérification des variations extrêmes par rapport aux données historiques
                    if len(df_enc) > 0:
                        enc_mean_hist = np.mean(df_enc['y_enc'])
                        enc_mean_pred = np.mean(lstm_enc_forecast)
                        enc_ratio = enc_mean_pred / enc_mean_hist if enc_mean_hist > 0 else 1
                        
                        if enc_ratio > 3 or enc_ratio < 0.3:
                            print(f"Attention: Les prévisions d'encaissements semblent anormales (ratio: {enc_ratio:.2f})")
                            print(f"Moyenne historique: {enc_mean_hist:.2f}, Moyenne prédite: {enc_mean_pred:.2f}")
                            print("Ajustement des prévisions pour qu'elles soient plus réalistes...")
                            # Ajustement des prévisions pour qu'elles soient plus réalistes
                            adjustment_factor = min(max(enc_ratio, 0.5), 2.0) / enc_ratio
                            lstm_enc_forecast = lstm_enc_forecast * adjustment_factor
                    
                    if len(df_dec) > 0:
                        dec_mean_hist = np.mean(df_dec['y_dec'])
                        dec_mean_pred = np.mean(lstm_dec_forecast)
                        dec_ratio = dec_mean_pred / dec_mean_hist if dec_mean_hist > 0 else 1
                        
                        if dec_ratio > 3 or dec_ratio < 0.3:
                            print(f"Attention: Les prévisions de décaissements semblent anormales (ratio: {dec_ratio:.2f})")
                            print(f"Moyenne historique: {dec_mean_hist:.2f}, Moyenne prédite: {dec_mean_pred:.2f}")
                            print("Ajustement des prévisions pour qu'elles soient plus réalistes...")
                            # Ajustement des prévisions pour qu'elles soient plus réalistes
                            adjustment_factor = min(max(dec_ratio, 0.5), 2.0) / dec_ratio
                            lstm_dec_forecast = lstm_dec_forecast * adjustment_factor
                    
                    # Enregistrement des prévisions
                    forecasts['lstm_enc'] = lstm_enc_forecast
                    forecasts['lstm_dec'] = lstm_dec_forecast
                    print("Prévisions LSTM ajoutées avec succès aux prévisions globales.")
                    print(f"Résumé des prévisions LSTM - Encaissements: min={np.min(lstm_enc_forecast):.2f}, max={np.max(lstm_enc_forecast):.2f}, mean={np.mean(lstm_enc_forecast):.2f}")
                    print(f"Résumé des prévisions LSTM - Décaissements: min={np.min(lstm_dec_forecast):.2f}, max={np.max(lstm_dec_forecast):.2f}, mean={np.mean(lstm_dec_forecast):.2f}")
                else:
                    if not enc_valid:
                        print("Les prévisions LSTM d'encaissements contiennent des valeurs non valides (NaN ou infini).")
                    if not dec_valid:
                        print("Les prévisions LSTM de décaissements contiennent des valeurs non valides (NaN ou infini).")
                    print("Le modèle LSTM ne sera pas utilisé pour les prévisions.")
            except Exception as e:
                import traceback
                print(f"Erreur lors de la génération des prévisions LSTM: {e}")
                print(f"Détails de l'erreur: {traceback.format_exc()}")
                print("Le modèle LSTM ne sera pas utilisé pour les prévisions.")
            print("\n=== Fin de la génération des prévisions LSTM ===\n")
        
        # XGBoost with error handling
        if 'xgboost_enc' in self.models:
            try:
                # Préparation des données pour la prédiction
                X_enc_future = np.array(range(len(df_enc), len(df_enc) + n_mois)).reshape(-1, 1)
                forecasts['xgb_enc'] = self.models['xgboost_enc'].predict(X_enc_future)
                print("Prévisions XGBoost pour les encaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions XGBoost pour les encaissements: {e}")
            
        if 'xgboost_dec' in self.models:
            try:
                X_dec_future = np.array(range(len(df_dec), len(df_dec) + n_mois)).reshape(-1, 1)
                forecasts['xgb_dec'] = self.models['xgboost_dec'].predict(X_dec_future)
                print("Prévisions XGBoost pour les décaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions XGBoost pour les décaissements: {e}")
                
        # Random Forest
        if 'rf_enc' in self.models:
            try:
                X_enc_future = np.array(range(len(df_enc), len(df_enc) + n_mois)).reshape(-1, 1)
                forecasts['rf_enc'] = self.models['rf_enc'].predict(X_enc_future)
                print("Prévisions Random Forest pour les encaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions Random Forest pour les encaissements: {e}")
                
        if 'rf_dec' in self.models:
            try:
                X_dec_future = np.array(range(len(df_dec), len(df_dec) + n_mois)).reshape(-1, 1)
                forecasts['rf_dec'] = self.models['rf_dec'].predict(X_dec_future)
                print("Prévisions Random Forest pour les décaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions Random Forest pour les décaissements: {e}")
            
        # Modèle hybride
        if 'hybrid_enc' in self.models:
            try:
                X_enc_future = np.array(range(len(df_enc), len(df_enc) + n_mois)).reshape(-1, 1)
                forecasts['hybrid_enc'] = self.models['hybrid_enc'].predict(X_enc_future)
                print("Prévisions du modèle Hybride pour les encaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions du modèle Hybride pour les encaissements: {e}")
            
        if 'hybrid_dec' in self.models:
            try:
                X_dec_future = np.array(range(len(df_dec), len(df_dec) + n_mois)).reshape(-1, 1)
                forecasts['hybrid_dec'] = self.models['hybrid_dec'].predict(X_dec_future)
                print("Prévisions du modèle Hybride pour les décaissements générées avec succès.")
            except Exception as e:
                print(f"Erreur lors de la génération des prévisions du modèle Hybride pour les décaissements: {e}")
            
        # Modèle hybride avancé (combinaison pondérée)
        if 'hybrid_advanced_enc' in self.models:
            # Récupérer les prévisions individuelles
            prophet_pred = None
            xgb_pred = None
            rf_pred = None
            
            # Prophet
            if 'prophet' in self.models['hybrid_advanced_enc']['models']:
                if 'prophet_enc' in forecasts:
                    prophet_pred = forecasts['prophet_enc']
            
            # XGBoost
            if 'xgboost' in self.models['hybrid_advanced_enc']['models']:
                X_enc_future = np.array(range(len(df_enc), len(df_enc) + n_mois)).reshape(-1, 1)
                xgb_pred = self.models['hybrid_advanced_enc']['models']['xgboost'].predict(X_enc_future)
            
            # Random Forest
            if 'rf' in self.models['hybrid_advanced_enc']['models']:
                X_enc_future = np.array(range(len(df_enc), len(df_enc) + n_mois)).reshape(-1, 1)
                rf_pred = self.models['hybrid_advanced_enc']['models']['rf'].predict(X_enc_future)
            
            # Combiner les prévisions si toutes sont disponibles
            if prophet_pred is not None and xgb_pred is not None and rf_pred is not None:
                weights = self.models['hybrid_advanced_enc']['weights']
                forecasts['hybrid_advanced_enc'] = (
                    weights[0] * prophet_pred + 
                    weights[1] * xgb_pred + 
                    weights[2] * rf_pred
                )
                
        # Même chose pour les décaissements
        if 'hybrid_advanced_dec' in self.models:
            prophet_pred = None
            xgb_pred = None
            rf_pred = None
            
            if 'prophet' in self.models['hybrid_advanced_dec']['models']:
                if 'prophet_dec' in forecasts:
                    prophet_pred = forecasts['prophet_dec']
            
            if 'xgboost' in self.models['hybrid_advanced_dec']['models']:
                X_dec_future = np.array(range(len(df_dec), len(df_dec) + n_mois)).reshape(-1, 1)
                xgb_pred = self.models['hybrid_advanced_dec']['models']['xgboost'].predict(X_dec_future)
            
            if 'rf' in self.models['hybrid_advanced_dec']['models']:
                X_dec_future = np.array(range(len(df_dec), len(df_dec) + n_mois)).reshape(-1, 1)
                rf_pred = self.models['hybrid_advanced_dec']['models']['rf'].predict(X_dec_future)
            
            if prophet_pred is not None and xgb_pred is not None and rf_pred is not None:
                weights = self.models['hybrid_advanced_dec']['weights']
                forecasts['hybrid_advanced_dec'] = (
                    weights[0] * prophet_pred + 
                    weights[1] * xgb_pred + 
                    weights[2] * rf_pred
                )
        
        # Vérifier si nous avons les modèles LSTM
        lstm_components_available = ['lstm_enc_model', 'lstm_dec_model', 'lstm_enc_scaler', 'lstm_dec_scaler'] 
        if all(k in self.models for k in lstm_components_available):
            try:
                # Génération des prévisions LSTM
                lstm_enc_forecast = self.lstm_forecast(
                    self.models['lstm_enc_model'],
                    self.models['lstm_enc_scaler'],
                    self.models['scaled_enc'],
                    n_mois=n_mois
                )
                
                lstm_dec_forecast = self.lstm_forecast(
                    self.models['lstm_dec_model'],
                    self.models['lstm_dec_scaler'],
                    self.models['scaled_dec'],
                    n_mois=n_mois
                )
                
                # Ajout des prévisions au dictionnaire
                forecasts['lstm_enc'] = lstm_enc_forecast
                forecasts['lstm_dec'] = lstm_dec_forecast
                print("Prévisions LSTM générées avec succès.")
            except Exception as e:
                import traceback
                print(f"Erreur lors de la génération des prévisions LSTM: {e}")
                print(traceback.format_exc())
            
            # Si toujours pas de prévisions, générer des prévisions simples basées sur les moyennes historiques
            if not forecasts:
                print("\nGénération de prévisions de secours basées sur les moyennes historiques...")
                try:
                    # Calcul des moyennes historiques
                    enc_mean = np.mean(df_enc['y_enc']) if len(df_enc) > 0 else 1000
                    dec_mean = np.mean(df_dec['y_dec']) if len(df_dec) > 0 else 800
                    
                    # Génération de prévisions simples avec un peu de bruit
                    np.random.seed(42)  # Pour la reproductibilité
                    enc_noise = np.random.normal(0, enc_mean * 0.05, n_mois)
                    dec_noise = np.random.normal(0, dec_mean * 0.05, n_mois)
                    
                    # Création des prévisions
                    fallback_enc = np.ones(n_mois) * enc_mean + enc_noise
                    fallback_dec = np.ones(n_mois) * dec_mean + dec_noise
                    
                    # Ajout des prévisions au dictionnaire
                    forecasts['fallback_enc'] = fallback_enc
                    forecasts['fallback_dec'] = fallback_dec
                    print("Prévisions de secours générées avec succès.")
                except Exception as e:
                    print(f"Erreur lors de la génération des prévisions de secours: {e}")
                    raise ValueError("Aucun modèle n'a pu générer de prévisions. Vérifiez les paramètres et les données.")
            
        # Si nous avons des prévisions maintenant, tout va bien
        if not forecasts:
            raise ValueError("Aucun modèle n'a pu générer de prévisions. Vérifiez les paramètres et les données.")
        
        # Ensure all forecasts have the same length
        forecast_lengths = [len(f) for f in forecasts.values()]
        if forecast_lengths:
            min_forecast_length = min(forecast_lengths)
            
            # Trim all forecasts to the same length
            for model in list(forecasts.keys()):
                forecasts[model] = forecasts[model][:min_forecast_length]
        
        return forecasts
    
    def select_best_model(self, df_enc, forecasts, metric='MAE'):
        """
        Sélectionne le meilleur modèle en fonction des métriques d'erreur.
        
        Args:
            df_enc (DataFrame): Données d'encaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            metric (str): Métrique à utiliser pour la sélection ('MAE' ou 'MAPE')
            
        Returns:
            tuple: (best_model, metrics) - Nom du meilleur modèle et dictionnaire des métriques
        """
        if not forecasts:
            return '', {}
            
        metrics = {}
        for model_name, forecast in forecasts.items():
            if 'enc' in model_name:
                try:
                    # Ensure we have enough historical data to compare
                    if len(forecast) > 0 and len(df_enc) >= len(forecast):
                        # Calculate metrics
                        y_true = df_enc['y_enc'].values[-len(forecast):]
                        y_pred = forecast
                        
                        mae = mean_absolute_error(y_true, y_pred)
                        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
                        
                        metrics[model_name] = {'MAE': mae, 'MAPE': mape}
                except Exception as e:
                    print(f"Erreur lors du calcul des métriques pour {model_name}: {e}")
        
        # Handle case where no metrics could be computed
        if not metrics:
            print("Impossible de calculer les métriques des modèles.")
            return None, {}
        
        # Sélectionner le meilleur modèle selon la métrique choisie par l'utilisateur
        if metric == "MAPE":
            best_model = min(metrics, key=lambda k: metrics[k]['MAPE'])
        else:  # Par défaut, utiliser MAE
            best_model = min(metrics, key=lambda k: metrics[k]['MAE'])
        
        return best_model, metrics
    
    def create_scenarios(self, forecasts, n_mois, confidence_interval=0.95):
        """
        Crée différents scénarios de prévision basés sur les modèles entraînés.
        
        Args:
            forecasts (dict): Dictionnaire des prévisions par modèle
            n_mois (int): Nombre de mois à prévoir
            confidence_interval (float): Intervalle de confiance (entre 0 et 1)
            
        Returns:
            dict: Dictionnaire des scénarios
        """
        scenarios = {}
        
        try:
            # Vérifier si nous avons des prévisions
            if not forecasts:
                return {}
                
            # Calculer plusieurs niveaux d'intervalles de confiance si demandé
            confidence_levels = self.config.get('confidence_levels', [0.80, 0.90, 0.95])
            if not isinstance(confidence_levels, list):
                confidence_levels = [0.95]  # Valeur par défaut
                
            # Identifier les modèles d'encaissement et de décaissement
            enc_models = [m for m in forecasts.keys() if 'enc' in m]
            dec_models = [m for m in forecasts.keys() if 'dec' in m]
            
            if not enc_models or not dec_models:
                return {}
            
            # Utiliser le premier modèle disponible pour chaque type
            base_enc_model = enc_models[0]
            base_dec_model = dec_models[0]
            
            # Vérifier que les prévisions ne sont pas vides
            if len(forecasts[base_enc_model]) == 0 or len(forecasts[base_dec_model]) == 0:
                return {}
                
            # Convertir en numpy arrays si nécessaire pour éviter les problèmes d'indexation
            base_enc_forecast = np.array(forecasts[base_enc_model]) if not isinstance(forecasts[base_enc_model], np.ndarray) else forecasts[base_enc_model]
            base_dec_forecast = np.array(forecasts[base_dec_model]) if not isinstance(forecasts[base_dec_model], np.ndarray) else forecasts[base_dec_model]
            
            # Scénario optimiste: +10% pour les encaissements, -5% pour les décaissements
            optimistic_enc = base_enc_forecast.copy() * 1.10
            optimistic_dec = base_dec_forecast.copy() * 0.95
            
            scenarios['optimiste'] = {
                'encaissement': optimistic_enc,
                'decaissement': optimistic_dec,
                'solde': optimistic_enc - optimistic_dec
            }
            
            # Scénario pessimiste: -5% pour les encaissements, +10% pour les décaissements
            pessimistic_enc = base_enc_forecast.copy() * 0.95
            pessimistic_dec = base_dec_forecast.copy() * 1.10
            
            scenarios['pessimiste'] = {
                'encaissement': pessimistic_enc,
                'decaissement': pessimistic_dec,
                'solde': pessimistic_enc - pessimistic_dec
            }
            
            # Scénario neutre: prévisions de base
            scenarios['neutre'] = {
                'encaissement': base_enc_forecast.copy(),
                'decaissement': base_dec_forecast.copy(),
                'solde': base_enc_forecast.copy() - base_dec_forecast.copy()
            }
            
            # Scénario de croissance: +5% par mois pour les encaissements
            growth_enc = base_enc_forecast.copy()
            for i in range(len(growth_enc)):
                growth_enc[i] = growth_enc[i] * (1 + 0.05 * (i+1))
            
            scenarios['croissance'] = {
                'encaissement': growth_enc,
                'decaissement': base_dec_forecast.copy(),
                'solde': growth_enc - base_dec_forecast.copy()
            }
            
            # Intervalles de confiance personnalisables
            for level in confidence_levels:
                level_str = str(int(level * 100))
                
                # Calculer l'écart-type des prévisions de tous les modèles
                enc_forecasts = []
                dec_forecasts = []
                
                for model_name, forecast in forecasts.items():
                    if 'enc' in model_name:
                        enc_forecasts.append(forecast)
                    elif 'dec' in model_name:
                        dec_forecasts.append(forecast)
                
                if len(enc_forecasts) > 1 and len(dec_forecasts) > 1:
                    # Convertir en tableau numpy pour faciliter les calculs
                    enc_array = np.array(enc_forecasts)
                    dec_array = np.array(dec_forecasts)
                    
                    # Calculer l'écart-type pour chaque mois
                    enc_std = np.std(enc_array, axis=0)
                    dec_std = np.std(dec_array, axis=0)
                    
                    # Calculer le z-score pour l'intervalle de confiance
                    z_score = stats.norm.ppf((1 + level) / 2)
                    
                    # Calculer les bornes supérieures et inférieures
                    enc_upper = base_enc_forecast + z_score * enc_std
                    enc_lower = base_enc_forecast - z_score * enc_std
                    dec_upper = base_dec_forecast + z_score * dec_std
                    dec_lower = base_dec_forecast - z_score * dec_std
                    
                    # S'assurer que les valeurs sont positives
                    enc_lower = np.maximum(enc_lower, 0)
                    dec_lower = np.maximum(dec_lower, 0)
                    
                    # Scénario optimiste avec intervalle de confiance
                    scenarios[f'optimiste_{level_str}'] = {
                        'encaissement': enc_upper,
                        'decaissement': dec_lower,
                        'solde': enc_upper - dec_lower,
                        'confidence_level': level
                    }
                    
                    # Scénario pessimiste avec intervalle de confiance
                    scenarios[f'pessimiste_{level_str}'] = {
                        'encaissement': enc_lower,
                        'decaissement': dec_upper,
                        'solde': enc_lower - dec_upper,
                        'confidence_level': level
                    }
            
            return scenarios
        except Exception as e:
            # En cas d'erreur, retourner un dictionnaire vide
            print(f"Erreur lors de la création des scénarios: {e}")
            return {}
    
    def simulate_monte_carlo(self, forecasts, n_mois, n_simulations=1000):
        """
        Effectue une simulation Monte Carlo pour évaluer les risques.
        
        Args:
            forecasts (dict): Dictionnaire des prévisions par modèle
            n_mois (int): Nombre de mois à prévoir
            n_simulations (int): Nombre de simulations à effectuer
            
        Returns:
            dict: Résultats de la simulation Monte Carlo
        """
        try:
            # Vérifier si nous avons des prévisions
            if not forecasts:
                return {}
            
            # Identifier les modèles d'encaissement et de décaissement
            enc_models = [m for m in forecasts.keys() if 'enc' in m]
            dec_models = [m for m in forecasts.keys() if 'dec' in m]
            
            if not enc_models or not dec_models:
                return {}
            
            # Utiliser le premier modèle disponible pour chaque type
            base_enc_model = enc_models[0]
            base_dec_model = dec_models[0]
            
            # Récupérer les prévisions de base
            base_enc = forecasts[base_enc_model]
            base_dec = forecasts[base_dec_model]
            
            # Calculer l'écart-type des prévisions de tous les modèles
            enc_forecasts = []
            dec_forecasts = []
            
            for model_name, forecast in forecasts.items():
                if 'enc' in model_name:
                    enc_forecasts.append(forecast)
                elif 'dec' in model_name:
                    dec_forecasts.append(forecast)
            
            # Convertir en tableau numpy pour faciliter les calculs
            enc_array = np.array(enc_forecasts) if len(enc_forecasts) > 1 else np.array([base_enc, base_enc])
            dec_array = np.array(dec_forecasts) if len(dec_forecasts) > 1 else np.array([base_dec, base_dec])
            
            # Calculer l'écart-type pour chaque mois
            enc_std = np.std(enc_array, axis=0)
            dec_std = np.std(dec_array, axis=0)
            
            # Générer des simulations
            np.random.seed(42)  # Pour la reproductibilité
            
            # Initialiser les tableaux pour stocker les résultats
            simulations_enc = np.zeros((n_simulations, len(base_enc)))
            simulations_dec = np.zeros((n_simulations, len(base_dec)))
            simulations_solde = np.zeros((n_simulations, len(base_enc)))
            
            # Générer des simulations avec distribution normale
            for i in range(n_simulations):
                # Générer des variations aléatoires
                enc_variation = np.random.normal(0, enc_std)
                dec_variation = np.random.normal(0, dec_std)
                
                # Appliquer les variations aux prévisions de base
                sim_enc = base_enc + enc_variation
                sim_dec = base_dec + dec_variation
                
                # S'assurer que les valeurs sont positives
                sim_enc = np.maximum(sim_enc, 0)
                sim_dec = np.maximum(sim_dec, 0)
                
                # Calculer le solde
                sim_solde = sim_enc - sim_dec
                
                # Stocker les résultats
                simulations_enc[i] = sim_enc
                simulations_dec[i] = sim_dec
                simulations_solde[i] = sim_solde
            
            # Calculer les statistiques des simulations
            enc_mean = np.mean(simulations_enc, axis=0)
            dec_mean = np.mean(simulations_dec, axis=0)
            solde_mean = np.mean(simulations_solde, axis=0)
            
            # Calculer les percentiles pour les intervalles de confiance
            enc_lower_95 = np.percentile(simulations_enc, 2.5, axis=0)
            enc_upper_95 = np.percentile(simulations_enc, 97.5, axis=0)
            dec_lower_95 = np.percentile(simulations_dec, 2.5, axis=0)
            dec_upper_95 = np.percentile(simulations_dec, 97.5, axis=0)
            solde_lower_95 = np.percentile(simulations_solde, 2.5, axis=0)
            solde_upper_95 = np.percentile(simulations_solde, 97.5, axis=0)
            
            # Calculer la probabilité de solde négatif pour chaque mois
            prob_negative_solde = np.mean(simulations_solde < 0, axis=0) * 100
            
            return {
                'encaissement_mean': enc_mean,
                'encaissement_lower_95': enc_lower_95,
                'encaissement_upper_95': enc_upper_95,
                'decaissement_mean': dec_mean,
                'decaissement_lower_95': dec_lower_95,
                'decaissement_upper_95': dec_upper_95,
                'solde_mean': solde_mean,
                'solde_lower_95': solde_lower_95,
                'solde_upper_95': solde_upper_95,
                'prob_negative_solde': prob_negative_solde,
                'n_simulations': n_simulations
            }
        
        except Exception as e:
            print(f"Erreur lors de la simulation Monte Carlo: {e}")
            return {}
    
    def analyze_sensitivity(self, forecasts, n_mois, factors=None):
        """
        Effectue une analyse de sensibilité pour identifier les facteurs les plus influents.
        
        Args:
            forecasts (dict): Dictionnaire des prévisions par modèle
            n_mois (int): Nombre de mois à prévoir
            factors (dict): Dictionnaire des facteurs à analyser et leurs plages de variation
            
        Returns:
            dict: Résultats de l'analyse de sensibilité
        """
        try:
            # Vérifier si nous avons des prévisions
            if not forecasts:
                return {}
            
            # Facteurs par défaut si non spécifiés
            if factors is None:
                factors = {
                    'enc_growth': {'values': [-10, -5, 0, 5, 10], 'unit': '%'},
                    'dec_growth': {'values': [-10, -5, 0, 5, 10], 'unit': '%'},
                    'enc_volatility': {'values': [0, 5, 10, 15], 'unit': '%'},
                    'dec_volatility': {'values': [0, 5, 10, 15], 'unit': '%'}
                }
            
            # Identifier les modèles d'encaissement et de décaissement
            enc_models = [m for m in forecasts.keys() if 'enc' in m]
            dec_models = [m for m in forecasts.keys() if 'dec' in m]
            
            if not enc_models or not dec_models:
                return {}
            
            # Utiliser le premier modèle disponible pour chaque type
            base_enc_model = enc_models[0]
            base_dec_model = dec_models[0]
            
            # Récupérer les prévisions de base
            base_enc = forecasts[base_enc_model]
            base_dec = forecasts[base_dec_model]
            base_solde = base_enc - base_dec
            
            results = {}
            
            # Analyser chaque facteur
            for factor_name, factor_info in factors.items():
                factor_results = []
                
                for value in factor_info['values']:
                    # Créer un scénario personnalisé avec ce facteur
                    params = {k: 0 for k in factors.keys()}  # Initialiser tous les facteurs à 0
                    params[factor_name] = value  # Définir le facteur à analyser
                    
                    # Générer le scénario
                    scenario = self.create_custom_scenario(forecasts, n_mois, params)
                    
                    if scenario:
                        # Calculer l'impact sur le solde final
                        solde_impact = np.sum(scenario['solde'] - base_solde)
                        solde_impact_percent = (solde_impact / np.sum(np.abs(base_solde))) * 100 if np.sum(np.abs(base_solde)) > 0 else 0
                        
                        factor_results.append({
                            'value': value,
                            'unit': factor_info['unit'],
                            'solde_impact': solde_impact,
                            'solde_impact_percent': solde_impact_percent
                        })
                
                # Trier les résultats par impact absolu
                factor_results.sort(key=lambda x: abs(x['solde_impact']), reverse=True)
                results[factor_name] = factor_results
            
            # Calculer l'importance relative de chaque facteur
            factor_importance = {}
            max_impact = 0
            
            for factor_name, factor_results in results.items():
                if factor_results:
                    # Prendre l'impact maximal pour ce facteur
                    max_factor_impact = max([abs(r['solde_impact']) for r in factor_results])
                    factor_importance[factor_name] = max_factor_impact
                    max_impact = max(max_impact, max_factor_impact)
            
            # Normaliser l'importance relative
            if max_impact > 0:
                for factor_name in factor_importance:
                    factor_importance[factor_name] = (factor_importance[factor_name] / max_impact) * 100
            
            return {
                'factor_results': results,
                'factor_importance': factor_importance
            }
        
        except Exception as e:
            print(f"Erreur lors de l'analyse de sensibilité: {e}")
            return {}
    
    def create_custom_scenario(self, forecasts, n_mois, params):
        """
        Crée un scénario personnalisé basé sur les paramètres fournis.
        
        Args:
            forecasts (dict): Dictionnaire des prévisions par modèle
            n_mois (int): Nombre de mois à prévoir
            params (dict): Paramètres du scénario personnalisé
            
        Returns:
            dict: Dictionnaire du scénario personnalisé
        """
        try:
            # Vérifier si nous avons des prévisions
            if not forecasts:
                return {}
            
            # Récupérer les paramètres
            enc_growth = params.get('enc_growth', 0)
            enc_volatility = params.get('enc_volatility', 0)
            enc_seasonality = params.get('enc_seasonality', 'Aucune')
            dec_growth = params.get('dec_growth', 0)
            dec_volatility = params.get('dec_volatility', 0)
            dec_seasonality = params.get('dec_seasonality', 'Aucune')
            
            # Identifier les modèles d'encaissement et de décaissement
            enc_models = [m for m in forecasts.keys() if 'enc' in m]
            dec_models = [m for m in forecasts.keys() if 'dec' in m]
            
            if not enc_models or not dec_models:
                return {}
            
            # Utiliser le premier modèle disponible pour chaque type
            base_enc_model = enc_models[0]
            base_dec_model = dec_models[0]
            
            # Récupérer les prévisions de base
            base_enc = forecasts[base_enc_model]
            base_dec = forecasts[base_dec_model]
            
            # Appliquer les modifications selon les paramètres
            custom_enc = base_enc * (1 + enc_growth/100)
            custom_dec = base_dec * (1 + dec_growth/100)
            
            # Appliquer la saisonnalité
            if enc_seasonality == "Mensuelle":
                custom_enc = custom_enc * (1 + 0.1 * np.sin(np.arange(len(custom_enc)) * (2*np.pi/12)))
            elif enc_seasonality == "Trimestrielle":
                custom_enc = custom_enc * (1 + 0.15 * np.sin(np.arange(len(custom_enc)) * (2*np.pi/3)))
            
            if dec_seasonality == "Mensuelle":
                custom_dec = custom_dec * (1 + 0.1 * np.sin(np.arange(len(custom_dec)) * (2*np.pi/12)))
            elif dec_seasonality == "Trimestrielle":
                custom_dec = custom_dec * (1 + 0.15 * np.sin(np.arange(len(custom_dec)) * (2*np.pi/3)))
            
            # Appliquer la volatilité
            np.random.seed(42)  # Pour la reproductibilité des résultats
            custom_enc = custom_enc * (1 + np.random.normal(0, enc_volatility/100, size=len(custom_enc)))
            custom_dec = custom_dec * (1 + np.random.normal(0, dec_volatility/100, size=len(custom_dec)))
            
            # Calcul du solde prévisionnel
            custom_solde = custom_enc - custom_dec
            
            return {
                'encaissement': custom_enc,
                'decaissement': custom_dec,
                'solde': custom_solde
            }
            
        except Exception as e:
            print(f"Erreur lors de la création du scénario personnalisé: {e}")
            return {}
