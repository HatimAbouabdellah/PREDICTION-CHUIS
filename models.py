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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from cache import CacheManager, cached, get_cached_data
from parallel import ParallelManager
from validation import DataValidator, ErrorHandler, DataValidationError, ConfigurationError

class ForecastingModels:
    """Classe pour gérer les modèles de prévision de trésorerie"""
    
    def __init__(self, config=None, parallel=True):
        """
        Initialise les modèles de prévision avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration des modèles à utiliser et leurs paramètres
            parallel (bool): Si True, utilise la parallélisation pour les calculs
        """
        try:
            # Valider la configuration
            validation_report = DataValidator.validate_config(config or {})
            if not validation_report['valid']:
                raise ConfigurationError(
                    "Configuration invalide",
                    {'errors': validation_report['errors']}
                )
            
            # Utiliser la configuration validée
            self.config = validation_report.get('validated_config', config or {})
            self.models = {}
            self.seasonal_patterns = {}
            self.anomalies = {}
            self.parallel = parallel
            self.parallel_manager = ParallelManager() if parallel else None
            
        except Exception as e:
            error_report = ErrorHandler.handle_error(e, {'config': config})
            raise type(e)(error_report['message']) from e
        
    def _forecast_single_model(self, df, n_mois, model_type):
        """
        Prévision pour un seul modèle
        
        Args:
            df (pd.DataFrame): Données historiques
            n_mois (int): Nombre de mois à prévoir
            model_type (str): Type de modèle
            
        Returns:
            dict: Résultats de la prévision
        """
        if model_type == 'prophet':
            return self._forecast_prophet(df, n_mois)
        elif model_type == 'arima':
            return self._forecast_arima(df, n_mois)
        elif model_type == 'xgboost':
            return self._forecast_xgboost(df, n_mois)
        else:
            raise ValueError(f"Modèle non supporté: {model_type}")
    
    def _analyze_single_seasonality(self, df, min_periods):
        """
        Analyse saisonnière pour un seul DataFrame
        
        Args:
            df (pd.DataFrame): Données à analyser
            min_periods (int): Nombre minimum de périodes
            
        Returns:
            dict: Résultats de l'analyse
        """
        try:
            # Convertir en série temporelle indexée par date
            ts = df.set_index('ds')['y']
            
            # Décomposition saisonnière
            result = seasonal_decompose(
                ts,
                model='additive',
                period=min_periods,
                extrapolate_trend='freq'
            )
            
            return {
                'has_seasonality': True,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'validation': validation_report
            }
        except Exception as e:
            error_report = ErrorHandler.handle_error(
                e,
                {'df_shape': df.shape if isinstance(df, pd.DataFrame) else None}
            )
            return {
                'has_seasonality': False,
                'error': error_report
            }
        
    @cached()
    def analyze_seasonality(self, df, min_periods=12, parallel=False):
        """
        Détecte automatiquement les tendances saisonnières dans les données.
        
        Args:
            df (DataFrame): Données à analyser avec colonnes 'ds' et 'y'
            min_periods (int): Nombre minimum de périodes pour l'analyse
            parallel (bool): Si True, utilise la parallélisation
            
        Returns:
            dict: Dictionnaire contenant les composantes saisonnières
            
        Raises:
            DataValidationError: Si les données ne sont pas valides
            ConfigurationError: Si la configuration est invalide
        """
        try:
            # Valider les données
            validation_report = DataValidator.validate_dataframe(df)
            if not validation_report['valid']:
                raise DataValidationError(
                    "Données invalides",
                    validation_report
                )
                
            # Valider la configuration
            if min_periods < 1:
                raise ConfigurationError(
                    "min_periods doit être supérieur à 0",
                    {'min_periods': min_periods}
                )
                
            # Convertir les données en tuple pour la clé de cache
            df_tuple = tuple(zip(df['ds'].astype(str), df['y']))
            
            # Vérifier si nous avons assez de données
            if len(df) < min_periods:
                return {
                    'has_seasonality': False,
                    'message': f'Pas assez de données pour détecter la saisonnalité (minimum {min_periods} périodes)'
                }
                
            # Vérifier la présence des colonnes requises
            required_cols = ['ds', 'y']
            if not all(col in df.columns for col in required_cols):
                return {
                    'has_seasonality': False,
                    'message': f'Colonnes manquantes: {set(required_cols) - set(df.columns)}'
                }
                
            # Convertir en série temporelle indexée par date
            ts = df.set_index('ds')['y']
            
            try:
                # Si parallélisation activée pour l'analyse
                if self.parallel and parallel:
                    # Créer une liste de DataFrames pour l'analyse parallèle
                    dfs = [df]  # Pour le moment, on analyse un seul DataFrame
                    if self.parallel_manager:
                        results = self.parallel_manager.parallel_analyze(
                            self._analyze_single_seasonality,
                            dfs,
                            min_periods=min_periods
                        )
                        return results.get(f"df_{id(df)}", {})
            
                # Décomposition saisonnière séquentielle
                result = seasonal_decompose(
                    ts,
                    model='additive',
                    period=min_periods,
                    extrapolate_trend='freq'
                )
            except Exception as decomp_error:
                return {
                    'has_seasonality': False,
                    'error': f'Erreur lors de la décomposition: {str(decomp_error)}'
                }
            
            # Tester la significativité de la saisonnalité
            try:
                seasonal_strength = np.std(result.seasonal) / np.std(result.resid)
                has_seasonality = seasonal_strength > 0.3
            except Exception as strength_error:
                return {
                    'has_seasonality': False,
                    'error': f'Erreur lors du calcul de la force saisonnière: {str(strength_error)}'
                }
            
            # Déterminer la période dominante
            dominant_period = None
            if has_seasonality:
                try:
                    # Calculer l'autocorrélation pour identifier la période
                    acf = np.correlate(result.seasonal.dropna(), result.seasonal.dropna(), mode='full')
                    acf = acf[len(acf)//2:]
                    # Trouver les pics dans l'autocorrélation
                    peaks = np.where((acf[1:-1] > acf[0:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
                    if len(peaks) > 0:
                        dominant_period = peaks[0]
                except Exception as acf_error:
                    print(f"Avertissement: Erreur lors du calcul de l'autocorrélation: {acf_error}")
                    
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
            return {
                'has_seasonality': False,
                'error': str(e)
            }
    
    def detect_anomalies(self, df, column='y', method='zscore', threshold=3.0, min_points=5):
        """
        Détecte les valeurs aberrantes dans les données.
        
        Args:
            df (DataFrame): Données à analyser
            column (str): Colonne à analyser
            method (str): Méthode de détection ('zscore', 'iqr')
            threshold (float): Seuil pour la détection
            min_points (int): Nombre minimum de points pour la détection
            
        Returns:
            DataFrame: DataFrame avec une colonne supplémentaire indiquant les anomalies
        """
        try:
            # Vérifier si nous avons assez de données
            if len(df) < min_points:
                return df.assign(anomaly=False)
                
            # Vérifier la présence de la colonne
            if column not in df.columns:
                raise ValueError(f"Colonne '{column}' non trouvée dans les données")
                
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
        
        # Initialiser le dictionnaire des modèles
        self.models = {}
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
                    
                    # Configuration du modèle avec une architecture plus simple
                    model_enc = Sequential()
                    model_enc.add(LSTM(32, input_shape=(X_enc.shape[1], 1), return_sequences=False))
                    model_enc.add(Dense(16, activation='relu'))
                    model_enc.add(Dense(1))
                    
                    # Utilisation d'un optimiseur avec taux d'apprentissage réduit et décroissance
                    optimizer = Adam(learning_rate=0.001, decay=1e-6)
                    model_enc.compile(optimizer=optimizer, loss='mse')
                    
                    print("Entraînement du modèle LSTM encaissements...")
                    history_enc = model_enc.fit(X_enc, y_enc, epochs=100, batch_size=16, 
                                             callbacks=[early_stopping], verbose=0)
                    print(f"Entraînement terminé après {len(history_enc.history['loss'])} époques")
                    print(f"Perte finale: {history_enc.history['loss'][-1]:.4f}")
                    
                    # Sauvegarder le modèle et les scalers
                    models['lstm_enc_model'] = model_enc
                    models['lstm_enc_scaler'] = scaler_enc
                    models['scaled_enc'] = scaled_enc
                    
                    # Vérifier les prédictions pour éviter les valeurs anormales
                    predictions = model_enc.predict(X_enc)
                    predictions = scaler_enc.inverse_transform(predictions)
                    
                    # Calculer les statistiques historiques
                    enc_mean = df_enc['y_enc'].mean()
                    enc_std = df_enc['y_enc'].std()
                    
                    # Identifier les prédictions anormales (plus de 3 écart-types de la moyenne)
                    abnormal_mask = np.abs(predictions - enc_mean) > 3 * enc_std
                    if np.any(abnormal_mask):
                        print(f"Avertissement: Prédictions anormales détectées pour les encaissements")
                        print(f"Moyenne historique: {enc_mean:.2f}, Moyenne prédite: {predictions.mean():.2f}")
                        
                        # Ajuster les prédictions anormales
                        predictions[abnormal_mask] = np.clip(predictions[abnormal_mask], 
                                                         enc_mean - 3 * enc_std, 
                                                         enc_mean + 3 * enc_std)
                        print("Prédictions ajustées avec succès")
                    
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
                    # Récupérer le nombre de folds depuis la configuration
                    n_folds = self.config.get('n_folds', 5)
                    
                    # Liste des modèles à valider par validation croisée
                    models_to_validate = [
                        'prophet_enc', 'prophet_dec',
                        'arima_enc', 'arima_dec',
                        'xgboost_enc', 'xgboost_dec',
                        'rf_enc', 'rf_dec',
                        'hybrid_enc', 'hybrid_dec'
                    ]
                    
                    # Filtrer pour ne garder que les modèles qui existent
                    models_to_validate = [m for m in models_to_validate if m in models]
                    
                    # Afficher des informations sur la validation croisée
                    print(f"Exécution de la validation croisée avec {n_folds} folds pour {len(models_to_validate)} modèles")
                    
                    # Effectuer la validation croisée pour les encaissements
                    enc_models = [m for m in models_to_validate if 'enc' in m]
                    if enc_models:
                        # Vérifier si le DataFrame des encaissements est valide
                        if df_enc is not None and not df_enc.empty and 'y_enc' in df_enc.columns and len(df_enc) > n_folds:
                            print("Validation croisée des modèles d'encaissement...")
                            try:
                                enc_cv_results = self.cross_validate_models(df_enc, 'y_enc', enc_models, n_splits=n_folds)
                                
                                # Stocker les résultats dans un attribut pour utilisation ultérieure
                                if not hasattr(self, 'cv_results'):
                                    self.cv_results = {}
                                self.cv_results.update(enc_cv_results)
                                
                                # Afficher les résultats
                                if enc_cv_results:
                                    print("Résultats de la validation croisée pour les encaissements:")
                                    for model_name, score in enc_cv_results.items():
                                        print(f"  - {model_name}: MAE = {score:.2f}")
                                else:
                                    print("Aucun résultat de validation croisée pour les encaissements.")
                            except Exception as e:
                                print(f"Erreur lors de la validation croisée des encaissements: {e}")
                        else:
                            print("Impossible d'effectuer la validation croisée pour les encaissements: données insuffisantes.")
                    
                    # Effectuer la validation croisée pour les décaissements
                    dec_models = [m for m in models_to_validate if 'dec' in m]
                    if dec_models:
                        # Vérifier si le DataFrame des décaissements est valide
                        if df_dec is not None and not df_dec.empty and 'y_dec' in df_dec.columns and len(df_dec) > n_folds:
                            print("Validation croisée des modèles de décaissement...")
                            try:
                                dec_cv_results = self.cross_validate_models(df_dec, 'y_dec', dec_models, n_splits=n_folds)
                                
                                # Stocker les résultats
                                if not hasattr(self, 'cv_results'):
                                    self.cv_results = {}
                                self.cv_results.update(dec_cv_results)
                                
                                # Afficher les résultats
                                if dec_cv_results:
                                    print("Résultats de la validation croisée pour les décaissements:")
                                    for model_name, score in dec_cv_results.items():
                                        print(f"  - {model_name}: MAE = {score:.2f}")
                                else:
                                    print("Aucun résultat de validation croisée pour les décaissements.")
                            except Exception as e:
                                print(f"Erreur lors de la validation croisée des décaissements: {e}")
                        else:
                            print("Impossible d'effectuer la validation croisée pour les décaissements: données insuffisantes.")
                    
                    # Utiliser les résultats de la validation croisée pour améliorer la sélection du modèle
                    # (Cette partie sera utilisée par la méthode select_best_model)
                    
                    # Réentraîner sur toutes les données
                    models['xgboost_enc'].fit(X_enc, y_enc)
                else:
                    models['xgboost_enc'].fit(X_enc, y_enc)

# ... (code après la modification)
                    
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
        
        # Vérifier si le DataFrame est valide
        if df is None or df.empty or column not in df.columns:
            print(f"Erreur lors de la validation croisée: DataFrame vide ou colonne '{column}' manquante.")
            return cv_scores
        
        # Vérifier les valeurs manquantes
        if df[column].isnull().any():
            print("Attention: Des valeurs manquantes détectées dans la colonne cible. Remplissage avec la méthode ffill.")
            df = df.fillna(method='ffill')
            
            # Si des valeurs manquantes restent, utiliser la moyenne
            if df[column].isnull().any():
                df[column] = df[column].fillna(df[column].mean())
        
        # Vérifier si nous avons assez de données pour la validation croisée
        if len(df) < n_splits + 1:
            print(f"Avertissement: Nombre insuffisant de données ({len(df)} lignes) pour {n_splits} folds.")
            # Ajuster le nombre de folds si possible
            if len(df) > 2:
                n_splits = min(n_splits, len(df) - 1)
                print(f"Ajustement du nombre de folds à {n_splits}.")
            else:
                print("Impossible d'effectuer la validation croisée avec moins de 2 points de données.")
                return cv_scores
        
        try:
            # Préparation des données
            X = np.array(range(len(df))).reshape(-1, 1)
            y = df[column].values
            
            # Vérifier que nous avons des données valides
            if len(y) == 0 or np.isnan(y).all():
                print("Erreur: Aucune donnée valide pour la validation croisée.")
                return cv_scores
            
            # Remplacer les valeurs NaN restantes par la moyenne
            y = np.nan_to_num(y, nan=np.nanmean(y) if not np.isnan(y).all() else 0)
            
            # Utiliser TimeSeriesSplit pour la validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for model_name in models_list:
                print(f"\nValidation croisée du modèle: {model_name}")
                scores = []
                
                if 'prophet' in model_name:
                    # Vérifier que la colonne 'ds' existe
                    if 'ds' not in df.columns:
                        print("  Colonne 'ds' manquante pour le modèle Prophet")
                        continue
                        
                    # Initialiser la liste des scores pour ce modèle
                    scores = []
                    
                    # Itérer sur les folds de validation croisée
                    for train_index, test_index in tscv.split(X):
                        try:
                            # Vérifier si les indices sont valides
                            if len(train_index) == 0 or len(test_index) == 0:
                                print("  Indices de validation croisée invalides (vides)")
                                continue
                            
                            # Créer des copies des données d'entraînement et de test
                            train_df = df.iloc[train_index].copy()
                            test_df = df.iloc[test_index].copy()
                            
                            # Vérifier que nous avons suffisamment de données
                            if len(train_df) < 2 or len(test_df) < 1:
                                print(f"  Données insuffisantes: {len(train_df)} points d'entraînement, {len(test_df)} points de test")
                                continue
                            
                            # Renommer les colonnes pour Prophet
                            train_data = train_df.rename(columns={column: 'y', 'ds': 'ds'})
                            test_data = test_df.rename(columns={column: 'y', 'ds': 'ds'})
                            
                            # Créer et entraîner le modèle Prophet
                            prophet_model = Prophet()
                            prophet_model.fit(train_data[['ds', 'y']])
                            
                            # Préparer les données futures pour la prédiction
                            future_dates = pd.DataFrame({'ds': test_data['ds']})
                            
                            # Vérifier si nous avons des dates de test valides
                            if future_dates.empty:
                                print("  Aucune date de test valide pour la prédiction")
                                continue
                            
                            # Effectuer la prédiction
                            forecast = prophet_model.predict(future_dates)
                            
                            # Extraire les vraies valeurs et les prédictions
                            y_true = test_data['y'].values
                            y_pred = forecast['yhat'].values
                            
                            # Vérifier les dimensions
                            if len(y_true) != len(y_pred):
                                print(f"  Dimensions incohérentes: y_true={len(y_true)}, y_pred={len(y_pred)}")
                                continue
                            
                            # Gérer les valeurs manquantes
                            if np.isnan(y_true).any() or np.isnan(y_pred).any():
                                print("  Valeurs NaN détectées, remplacement par la moyenne")
                                y_true = np.nan_to_num(y_true, nan=np.nanmean(y_true) if not np.isnan(y_true).all() else 0)
                                y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred) if not np.isnan(y_pred).all() else 0)
                            
                            # Calculer et stocker le score MAE
                            mae = mean_absolute_error(y_true, y_pred)
                            scores.append(mae)
                            print(f"  Fold {len(scores)}: MAE = {mae:.2f}")
                            
                        except Exception as e:
                            print(f"  Erreur lors de la validation croisée de Prophet: {str(e)}")
                            continue
                    
                    if scores:
                        cv_scores[model_name] = np.mean(scores)
                        print(f"  Score MAE moyen: {cv_scores[model_name]:.2f}")
                    else:
                        print(f"  Aucun score valide pour le modèle {model_name}")
                
                elif 'arima' in model_name:
                    scores = []
                    for train_idx, test_idx in tscv.split(X):
                        try:
                            # Préparer les données d'entraînement et de test
                            train_y = y[train_idx]
                            test_y = y[test_idx]
                            
                            # Vérifier que nous avons assez de données
                            if len(train_y) < 2 or len(test_y) < 1:
                                print(f"  Données insuffisantes: {len(train_y)} points d'entraînement, {len(test_y)} points de test")
                                continue
                            
                            # Remplacer les valeurs NaN si nécessaire
                            train_y = np.nan_to_num(train_y, nan=np.nanmean(train_y) if not np.isnan(train_y).all() else 0)
                            
                            # Définir l'ordre ARIMA (peut être ajusté selon les besoins)
                            order = (1, 1, 0)  # Ordre plus simple pour plus de stabilité
                            
                            # Entraîner le modèle
                            arima_model = ARIMA(train_y, order=order)
                            arima_model_fit = arima_model.fit(disp=0)
                            
                            # Faire des prédictions
                            predictions = arima_model_fit.forecast(steps=len(test_y))[0]
                            
                            # Vérifier les dimensions
                            if len(test_y) != len(predictions):
                                print(f"  Dimensions incohérentes: y_true={len(test_y)}, y_pred={len(predictions)}")
                                continue
                            
                            # Calculer et stocker le score MAE
                            mae = mean_absolute_error(test_y, predictions)
                            scores.append(mae)
                            print(f"  Fold {len(scores)}: MAE = {mae:.2f}")
                            
                        except Exception as e:
                            print(f"  Erreur lors de la validation croisée d'ARIMA: {str(e)}")
                            continue
                    
                    if scores:
                        cv_scores[model_name] = np.mean(scores)
                        print(f"  Score MAE moyen: {cv_scores[model_name]:.2f}")
                    else:
                        print(f"  Aucun score valide pour le modèle {model_name}")
                
                elif any(m in model_name for m in ['xgboost', 'rf', 'hybrid']):
                    scores = []
                    
                    for train_idx, test_idx in tscv.split(X):
                        try:
                            # Préparer les données
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            # Vérifier que nous avons assez de données
                            if len(X_train) < 2 or len(X_test) < 1:
                                print(f"  Données insuffisantes: {len(X_train)} points d'entraînement, {len(X_test)} points de test")
                                continue
                            
                            # Créer et entraîner le modèle approprié
                            model = None
                            if 'xgboost' in model_name:
                                model = XGBRegressor(
                                    objective='reg:squarederror',
                                    n_estimators=100,
                                    random_state=42
                                )
                            elif 'rf' in model_name:
                                model = RandomForestRegressor(
                                    n_estimators=100,
                                    random_state=42,
                                    n_jobs=-1
                                )
                            elif 'hybrid' in model_name and 'advanced' not in model_name:
                                # Modèle hybride simple (VotingRegressor)
                                xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                model = VotingRegressor([('xgb', xgb), ('rf', rf)], n_jobs=-1)
                            
                            if model is None:
                                print(f"  Type de modèle non pris en charge: {model_name}")
                                break
                            
                            # Entraîner le modèle
                            model.fit(X_train, y_train)
                            
                            # Faire des prédictions
                            y_pred = model.predict(X_test)
                            
                            # Vérifier les dimensions
                            if len(y_test) != len(y_pred):
                                print(f"  Dimensions incohérentes: y_test={len(y_test)}, y_pred={len(y_pred)}")
                                continue
                            
                            # Calculer et stocker le score MAE
                            mae = mean_absolute_error(y_test, y_pred)
                            scores.append(mae)
                            print(f"  Fold {len(scores)}: MAE = {mae:.2f}")
                            
                        except Exception as e:
                            print(f"  Erreur lors de la validation croisée de {model_name}: {str(e)}")
                            continue
                    
                    if scores:
                        cv_scores[model_name] = np.mean(scores)
                        print(f"  Score MAE moyen: {cv_scores[model_name]:.2f}")
                    else:
                        print(f"  Aucun score valide pour le modèle {model_name}")
            
            return cv_scores
        
        except Exception as e:
            print(f"Erreur lors de la validation croisée: {e}")
            return cv_scores
    
    @cached()
    def forecast(self, df, n_mois, model_type='ensemble', config=None):
        """
        Génère des prévisions pour les n_mois suivants.
        
        Args:
            df (pd.DataFrame): Données historiques avec colonnes 'ds' et 'y'
            n_mois (int): Nombre de mois à prévoir
            model_type (str): Type de modèle à utiliser ('prophet', 'arima', 'ensemble')
            config (dict): Configuration spécifique au modèle
            
        Returns:
            dict: Dictionnaire contenant les prédictions et les métriques
            
        Raises:
            DataValidationError: Si les données ne sont pas valides
            ConfigurationError: Si la configuration est invalide
        """
        try:
            # Valider les données
            validation_report = DataValidator.validate_dataframe(df)
            if not validation_report['valid']:
                raise DataValidationError(
                    "Données invalides",
                    validation_report
                )
                
            # Valider la configuration
            forecast_config = {
                'n_mois': n_mois,
                'model_type': model_type,
                'config': config or {}
            }
            validation_report = DataValidator.validate_config(forecast_config)
            if not validation_report['valid']:
                raise ConfigurationError(
                    "Configuration de prévision invalide",
                    validation_report
                )
                
            # Convertir les données en tuple pour la clé de cache
            df_tuple = tuple(zip(df['ds'].astype(str), df['y']))
            
            # Si parallélisation activée et plusieurs modèles à prévoir
            if self.parallel and model_type == 'ensemble':
                # Préparer les tâches
                model_types = ['prophet', 'arima', 'xgboost']
                if self.parallel_manager:
                    # Exécuter les prédictions en parallèle
                    results = self.parallel_manager.parallel_forecast(
                        self._forecast_single_model,
                        [df] * len(model_types),
                        n_mois,
                        model_types=model_types
                    )
                    
                    # Aggréger les résultats
                    final_forecast = {}
                    for key, result in results.items():
                        if result:
                            model_type = key.split('_')[-1]
                            final_forecast[model_type] = result
                    
                    return final_forecast
            
            # Sinon, exécuter en séquentiel
            return self._forecast_single_model(df, n_mois, model_type)
            
        except Exception as e:
            error_report = ErrorHandler.handle_error(
                e,
                {
                    'df_shape': df.shape if isinstance(df, pd.DataFrame) else None,
                    'model_type': model_type,
                    'n_mois': n_mois
                }
            )
            return {
                'error': error_report,
                'validation': validation_report
            }
        
        # Si parallélisation activée et plusieurs modèles à prévoir
        if self.parallel and model_type == 'ensemble':
            # Préparer les tâches
            model_types = ['prophet', 'arima', 'xgboost']
            if self.parallel_manager:
                # Exécuter les prédictions en parallèle
                results = self.parallel_manager.parallel_forecast(
                    self._forecast_single_model,
                    [df] * len(model_types),
                    n_mois,
                    model_types=model_types
                )
                
                # Aggréger les résultats
                final_forecast = {}
                for key, result in results.items():
                    if result:
                        model_type = key.split('_')[-1]
                        final_forecast[model_type] = result
                
                return final_forecast
            
        # Sinon, exécuter en séquentiel
        return self._forecast_single_model(df, n_mois, model_type)
        forecasts = {}
        
        # Prepare Prophet dataframes
        df_enc_prophet = df_enc.rename(columns={'y_enc': 'y', 'ds': 'ds'})
        df_dec_prophet = df_dec.rename(columns={'y_dec': 'y', 'ds': 'ds'})
        
        # Prophet - Vérifier si Prophet est activé dans la configuration
        if self.config.get('use_prophet', True):
            if 'prophet_enc' in self.models and 'prophet_dec' in self.models:
                try:
                    future_enc = self.models['prophet_enc'].make_future_dataframe(periods=n_mois, freq='MS')
                    future_dec = self.models['prophet_dec'].make_future_dataframe(periods=n_mois, freq='MS')
                    
                    prophet_enc_forecast = self.models['prophet_enc'].predict(future_enc)['yhat'][-n_mois:].values
                    prophet_dec_forecast = self.models['prophet_dec'].predict(future_dec)['yhat'][-n_mois:].values
                    
                    forecasts['prophet_enc'] = prophet_enc_forecast
                    forecasts['prophet_dec'] = prophet_dec_forecast
                    print("Prévisions Prophet générées avec succès.")
                except Exception as e:
                    print(f"Erreur lors de la génération des prévisions Prophet: {e}")
            else:
                print("Les modèles Prophet ne sont pas disponibles dans self.models. Vérifiez l'entraînement des modèles.")
                print(f"Modèles disponibles: {list(self.models.keys())}")
        
        # ARIMA/SARIMA - Vérifier si ARIMA est activé dans la configuration
        if self.config.get('use_arima', True):
            if 'arima_enc' in self.models and 'arima_dec' in self.models:
                try:
                    arima_enc_forecast = self.models['arima_enc'].forecast(steps=n_mois)
                    arima_dec_forecast = self.models['arima_dec'].forecast(steps=n_mois)
                    forecasts['arima_enc'] = arima_enc_forecast
                    forecasts['arima_dec'] = arima_dec_forecast
                    print("Prévisions ARIMA générées avec succès.")
                except Exception as e:
                    print(f"Erreur lors de la génération des prévisions ARIMA: {e}")
            else:
                print("Les modèles ARIMA ne sont pas disponibles dans self.models. Vérifiez l'entraînement des modèles.")
                print(f"Modèles disponibles: {list(self.models.keys())}")
        
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
        
    @cached()
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
        # Générer une clé de cache basée sur les données
        cache_key = f"forecasts_{df_enc['y_enc'].sum():.2f}_{df_dec['y_dec'].sum():.2f}_{n_mois}"
        print(f"Clé de cache générée : {cache_key}")
        
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
        Si la validation croisée est activée, les résultats de celle-ci sont également pris en compte.
        
        Args:
            df_enc (DataFrame): Données d'encaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            metric (str): Métrique à utiliser pour la sélection ('MAE', 'RMSE' ou 'MAPE')
            
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
                        rmse = sqrt(mean_squared_error(y_true, y_pred))  # Calcul du RMSE
                        
                        metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                except Exception as e:
                    print(f"Erreur lors du calcul des métriques pour {model_name}: {e}")
        
        # Handle case where no metrics could be computed
        if not metrics:
            print("Impossible de calculer les métriques des modèles.")
            return None, {}
        
        # Vérifier si nous avons des résultats de validation croisée
        use_cross_validation = self.config.get('use_cross_validation', False)
        
        if use_cross_validation and hasattr(self, 'cv_results') and self.cv_results:
            print("Utilisation des résultats de validation croisée pour la sélection du modèle")
            
            # Combiner les métriques d'évaluation et de validation croisée
            combined_metrics = {}
            for model_name, model_metrics in metrics.items():
                if model_name in self.cv_results:
                    # Pondération : 40% validation croisée, 60% évaluation sur les données historiques
                    cv_score = self.cv_results[model_name]  # Score MAE de la validation croisée
                    if metric == "MAPE":
                        combined_metrics[model_name] = 0.6 * model_metrics['MAPE'] + 0.4 * cv_score
                    elif metric == "RMSE":
                        combined_metrics[model_name] = 0.6 * model_metrics['RMSE'] + 0.4 * cv_score
                    else:  # MAE
                        combined_metrics[model_name] = 0.6 * model_metrics['MAE'] + 0.4 * cv_score
                    
                    print(f"  - {model_name}: Score combiné = {combined_metrics[model_name]:.2f}")
                else:
                    # Si pas de résultat de validation croisée, utiliser uniquement l'évaluation
                    if metric == "MAPE":
                        combined_metrics[model_name] = model_metrics['MAPE']
                    elif metric == "RMSE":
                        combined_metrics[model_name] = model_metrics['RMSE']
                    else:  # MAE
                        combined_metrics[model_name] = model_metrics['MAE']
            
            # Sélectionner le modèle avec le meilleur score combiné
            if combined_metrics:
                best_model = min(combined_metrics, key=lambda k: combined_metrics[k])
                print(f"Meilleur modèle selon les scores combinés: {best_model}")
            else:
                # Fallback sur la méthode standard si pas de scores combinés
                if metric == "MAPE":
                    best_model = min(metrics, key=lambda k: metrics[k]['MAPE'])
                elif metric == "RMSE":
                    best_model = min(metrics, key=lambda k: metrics[k]['RMSE'])
                else:  # Par défaut, utiliser MAE
                    best_model = min(metrics, key=lambda k: metrics[k]['MAE'])
        else:
            # Méthode standard sans validation croisée
            if metric == "MAPE":
                best_model = min(metrics, key=lambda k: metrics[k]['MAPE'])
            elif metric == "RMSE":
                best_model = min(metrics, key=lambda k: metrics[k]['RMSE'])
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
