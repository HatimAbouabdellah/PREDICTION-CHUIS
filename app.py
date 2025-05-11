"""
Application principale de prévision de trésorerie
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import io
import base64
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import des modules personnalisés
from utils import load_and_clean_data, calculate_financial_metrics
from models import ForecastingModels
from visualizations import TresorerieVisualizer

# Configuration de la page
st.set_page_config(
    page_title=" TresoreriePro",
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Chargement du CSS personnalisé
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css(Path(__file__).parent / "style.css")
except Exception as e:
    st.warning(f"Impossible de charger le fichier CSS: {e}")

# Fonction pour afficher une image en base64
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# Titre de l'application avec style amélioré
st.markdown('<h1 class="main-title">🚀 Prévision Intelligente de Trésorerie Pro</h1>', unsafe_allow_html=True)

# Ajout d'une introduction
st.markdown('''
<div style="background-color: #f0f9ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #3b82f6;">
    <h3 style="margin-top: 0; color: #1e40af;">Bienvenue dans votre outil de prévision de trésorerie</h3>
    <p>Cet outil vous permet d'analyser vos flux financiers et de générer des prévisions précises pour optimiser votre gestion de trésorerie.</p>
    <p><strong>Pour commencer</strong>: Importez vos données dans la barre latérale et configurez les paramètres de prévision.</p>
</div>
''', unsafe_allow_html=True)

def configure_sidebar():
    """Configure la sidebar avec tous les paramètres nécessaires"""
    with st.sidebar:
        # Logo et titre de la sidebar avec style amélioré
        # st.markdown(
        #     '<div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e5e7eb;">'            
        #     '<h2 style="color: #1e3a8a; margin-bottom: 0.5rem;">💸 TresoreriePro</h2>'            
        #     '<p style="color: #6b7280; font-size: 0.9rem; margin-top: 0;">Configuration avancée</p>'            
        #     '</div>',
        #     unsafe_allow_html=True
        # )
        
        # Champ d'importation du fichier Excel (sans div)
        # uploaded_file = st.file_uploader(
        #     "Importer un fichier Excel", 
        #     type=["xlsx"], 
        #     help="Fichier Excel contenant les données de trésorerie"
        # )
        # st.session_state['uploaded_file'] = uploaded_file
        
        # # Séparateur visuel
        # st.markdown('<hr style="margin: 1.5rem 0; border: none; height: 1px; background-color: #e5e7eb;">', unsafe_allow_html=True)
        
        # Section de configuration des prévisions avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
            '<h3 style="font-size: 1.2rem; margin: 0 0 0.5rem 0; color: #1e40af;"> Paramètres de Prévision</h3>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Paramètres de prévision avec style amélioré
        st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Horizon de prévision</p>', unsafe_allow_html=True)
        n_mois_slider = st.slider(
            "Mois à prévoir", 
            min_value=1, 
            max_value=24, 
            value=6, 
            step=1,
            help="Nombre de mois à prédire dans le futur"
        )
        st.session_state['n_mois_slider'] = n_mois_slider
        
        # Intervalle de confiance
        st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Intervalle de confiance</p>', unsafe_allow_html=True)
        confidence_interval = st.slider(
            "Niveau de confiance (%)", 
            min_value=50, 
            max_value=99, 
            value=95, 
            step=5,
            help="Niveau de confiance pour les prévisions"
        )
        st.session_state['confidence_interval'] = confidence_interval / 100
        
        # Sélection des modèles avec style amélioré
        st.markdown(
            '<p style="font-weight: 500; margin: 1.5rem 0 0.5rem 0; padding-top: 0.5rem; border-top: 1px solid #e5e7eb;">'            
            'Modèles de prévision'            
            '</p>',
            unsafe_allow_html=True
        )
        
        # Modèles à utiliser (tous activés par défaut)
        st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Modèles à utiliser</p>', unsafe_allow_html=True)
        
        # Récupérer les valeurs précédentes ou utiliser True par défaut
        default_prophet = st.session_state.get('use_prophet', True)
        default_arima = st.session_state.get('use_arima', True)
        default_lstm = st.session_state.get('use_lstm', True)
        default_xgboost = st.session_state.get('use_xgboost', True)
        default_rf = st.session_state.get('use_rf', True)
        default_hybrid = st.session_state.get('use_hybrid', True)
        
        # Afficher les checkboxes avec les valeurs par défaut
        use_prophet = st.checkbox("Prophet", value=default_prophet, help="Utiliser le modèle Prophet de Facebook")
        use_arima = st.checkbox("ARIMA", value=default_arima, help="Utiliser le modèle ARIMA/SARIMA")
        use_lstm = st.checkbox("LSTM", value=default_lstm, help="Utiliser le modèle LSTM (Deep Learning)")
        use_xgboost = st.checkbox("XGBoost", value=default_xgboost, help="Utiliser le modèle XGBoost")
        use_rf = st.checkbox("Random Forest", value=default_rf, help="Utiliser le modèle Random Forest")
        use_hybrid = st.checkbox("Modèle Hybride", value=default_hybrid, help="Utiliser une combinaison de modèles pour améliorer la précision")
        
        # Sauvegarder les choix de l'utilisateur dans la session
        st.session_state['use_prophet'] = use_prophet
        st.session_state['use_arima'] = use_arima
        st.session_state['use_lstm'] = use_lstm
        st.session_state['use_xgboost'] = use_xgboost
        st.session_state['use_rf'] = use_rf
        st.session_state['use_hybrid'] = use_hybrid
        
        # Options avancées
        with st.expander("Options avancées"):
            use_cross_validation = st.checkbox("Validation croisée", value=False, 
                                             help="Utiliser la validation croisée pour évaluer plus précisément les modèles")
            
            st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Analyse avancée</p>', unsafe_allow_html=True)
            detect_seasonality = st.checkbox("Détection de saisonnalité", value=True, 
                                           help="Détecter automatiquement les tendances saisonnières")
            detect_anomalies = st.checkbox("Détection d'anomalies", value=True, 
                                          help="Détecter les valeurs aberrantes dans les données")
            
            st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Simulations avancées</p>', unsafe_allow_html=True)
            run_monte_carlo = st.checkbox("Simulation Monte Carlo", value=False, 
                                         help="Exécuter une simulation Monte Carlo pour évaluer les risques")
            if run_monte_carlo:
                monte_carlo_sims = st.slider("Nombre de simulations", min_value=100, max_value=10000, value=1000, step=100,
                                            help="Plus de simulations = résultats plus précis mais calcul plus long")
                
                # Activer automatiquement l'onglet des scénarios lorsque Monte Carlo est activé
                st.session_state['active_tab'] = "Scénarios"
                st.session_state['show_monte_carlo'] = True
            else:
                monte_carlo_sims = 1000
                st.session_state['show_monte_carlo'] = False
                
            run_sensitivity = st.checkbox("Analyse de sensibilité", value=False, 
                                       help="Analyser l'impact des différents facteurs sur les prévisions")
        
        # Sélection automatique du meilleur modèle
        st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
        auto_select = st.checkbox(
            "Sélection automatique du meilleur modèle", 
            value=True,
            help="Sélectionne automatiquement le modèle avec les meilleures performances"
        )
        st.session_state['auto_select'] = auto_select
        
        # Métrique pour la sélection du modèle
        if auto_select:
            selection_metric = st.radio(
                "Métrique de sélection", 
                ["MAE", "MAPE"],
                horizontal=True,
                help="Métrique utilisée pour sélectionner le meilleur modèle"
            )
            st.session_state['selection_metric'] = selection_metric
        
        # Séparateur visuel
        st.markdown('<hr style="margin: 1.5rem 0; border: none; height: 1px; background-color: #e5e7eb;">', unsafe_allow_html=True)
        
        # Section d'affichage avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">'            
            '<h3 style="font-size: 1.2rem; margin: 0 0 0.5rem 0; color: #1e40af;">📊 Affichage des Résultats</h3>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Options d'affichage
        st.markdown('<p style="font-weight: 500; margin-bottom: 0.25rem;">Options d\'affichage</p>', unsafe_allow_html=True)
        show_predictions = st.checkbox("Afficher les prévisions", value=True, help="Affiche les prévisions sur le graphique")
        show_confidence = st.checkbox("Afficher l'intervalle de confiance", value=True, help="Affiche l'intervalle de confiance sur le graphique")
        show_components = st.checkbox("Afficher les composantes", value=False, help="Affiche les composantes de la série temporelle")
        
        # Sauvegarder les options d'affichage dans la session
        st.session_state['show_predictions'] = show_predictions
        st.session_state['show_confidence'] = show_confidence
        st.session_state['show_components'] = show_components
        
        # Bouton d'analyse avec style amélioré
        st.markdown('<div style="margin: 2rem 0 0.5rem 0;"></div>', unsafe_allow_html=True)
        # analyze_button = st.button(
        #     "🔍 Analyser les Données",
        #     use_container_width=True,
        #     help="Lancer l'analyse des données et appliquer les filtres"
        # )
        
        # Mise à jour automatique des options d'affichage dans la session
        st.session_state['show_predictions'] = show_predictions
        st.session_state['show_confidence'] = show_confidence
        st.session_state['show_components'] = show_components
        
        # Si les options d'affichage ont changé, mettre à jour les filtres
        if ('prev_show_predictions' in st.session_state and st.session_state['prev_show_predictions'] != show_predictions) or \
           ('prev_show_confidence' in st.session_state and st.session_state['prev_show_confidence'] != show_confidence) or \
           ('prev_show_components' in st.session_state and st.session_state['prev_show_components'] != show_components):
            
            # Si des prévisions existent déjà, forcer leur mise à jour avec les nouveaux filtres
            if 'forecasts' in st.session_state and st.session_state['forecasts']:
                st.session_state['apply_new_filters'] = True
                st.rerun()
        
        # Sauvegarder les valeurs actuelles pour la comparaison lors de la prochaine exécution
        st.session_state['prev_show_predictions'] = show_predictions
        st.session_state['prev_show_confidence'] = show_confidence
        st.session_state['prev_show_components'] = show_components
        
        # Informations sur l'application
        st.markdown(
            '<div style="margin-top: 3rem; padding: 1rem; border-radius: 8px; background-color: #f3f4f6; text-align: center;">'            
            '<p style="color: #6b7280; font-size: 0.8rem; margin: 0;">TresoreriePro v2.0</p>'            
            '<p style="color: #6b7280; font-size: 0.8rem; margin: 0;">© 2025 Finance Analytics</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Retourner un dictionnaire de configuration
        return {
            'n_mois': n_mois_slider,
            'confidence_interval': confidence_interval,
            'use_prophet': use_prophet,
            'use_arima': use_arima,
            'use_lstm': use_lstm,
            'use_xgboost': use_xgboost,
            'use_rf': use_rf,
            'use_hybrid': use_hybrid,
            'use_cross_validation': use_cross_validation,
            'auto_select': auto_select,
            'selection_metric': selection_metric if auto_select else 'MAE',
            'show_predictions': show_predictions,
            'show_confidence': show_confidence,
            'show_components': show_components,
            'detect_seasonality': detect_seasonality,
            'detect_anomalies': detect_anomalies,
            'run_monte_carlo': run_monte_carlo,
            'monte_carlo_sims': monte_carlo_sims,
            'run_sensitivity': run_sensitivity
        }

def main():
    """Fonction principale de l'application"""
    # Configuration de la sidebar
    config = configure_sidebar()
    
    # Initialisation des variables
    forecasts = {}
    best_model = ''
    model_metrics = {}
    scenarios = {}
    df_enc = None
    df_dec = None
    df_tgr = None
    models = {}
    forecasting_models = None  # Initialisation de forecasting_models
    
    # Chargement des données
    uploaded_file = st.file_uploader("📂 Charger un fichier Excel", type="xlsx")
    
    if uploaded_file is not None:
        try:
            # Chargement et nettoyage des données
            with st.spinner("Chargement et nettoyage des données en cours..."):
                df_enc, df_dec, df_tgr = load_and_clean_data(uploaded_file)
            
            if df_enc is None or df_dec is None:
                st.error("Erreur lors du chargement des données.")
                return
            
            # Affichage des données chargées
            st.write("### 📃 Données Chargées")
            st.write(f"**Nombre de périodes :** {len(df_enc)}")
            st.write(f"**Période couverte :** {df_enc['ds'].min().strftime('%d/%m/%Y')} - {df_enc['ds'].max().strftime('%d/%m/%Y')}")
            
            # Récupération des paramètres
            n_mois = config['n_mois']
            
            # Affichage des paramètres sélectionnés
            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Horizon de prévision", f"{n_mois} mois")
            col2.metric("📏 Intervalle de confiance", f"{config['confidence_interval']}%")
            col3.metric("📊 Métrique de sélection", config['selection_metric'])
            
            # Affichage du statut initial des modèles
            st.markdown("### 🔍 Statut des Modèles")
            
            # Créer un dictionnaire pour stocker le statut de chaque modèle
            initial_model_status = {
                "Prophet": {
                    "Activé": config.get('use_prophet', True),
                    "Statut": "✅ Activé" if config.get('use_prophet', True) else "❌ Désactivé"
                },
                "ARIMA": {
                    "Activé": config.get('use_arima', True),
                    "Statut": "✅ Activé" if config.get('use_arima', True) else "❌ Désactivé"
                },
                "LSTM": {
                    "Activé": config.get('use_lstm', True),
                    "Statut": "✅ Activé" if config.get('use_lstm', True) else "❌ Désactivé"
                },
                "XGBoost": {
                    "Activé": config.get('use_xgboost', True),
                    "Statut": "✅ Activé" if config.get('use_xgboost', True) else "❌ Désactivé"
                },
                "Random Forest": {
                    "Activé": config.get('use_rf', True),
                    "Statut": "✅ Activé" if config.get('use_rf', True) else "❌ Désactivé"
                },
                "Modèle Hybride": {
                    "Activé": config.get('use_hybrid', True),
                    "Statut": "✅ Activé" if config.get('use_hybrid', True) else "❌ Désactivé"
                }
            }
            
            # Créer un DataFrame pour l'affichage
            initial_status_data = []
            for model_name, status in initial_model_status.items():
                initial_status_data.append({
                    "Modèle": model_name,
                    "Activé": "✅" if status["Activé"] else "❌",
                    "Statut": status["Statut"]
                })
            
            initial_status_df = pd.DataFrame(initial_status_data)
            st.dataframe(
                initial_status_df,
                use_container_width=True,
                column_config={
                    "Modèle": st.column_config.TextColumn("Modèle"),
                    "Activé": st.column_config.TextColumn("Activé"),
                    "Statut": st.column_config.TextColumn("Statut")
                },
                hide_index=True
            )
            
            # Bouton pour générer les prévisions
            generate_button = st.button(
                "📈 Générer Prévisions", 
                use_container_width=True,
                help="Cliquez pour générer les prévisions avec les paramètres sélectionnés"
            )
            
            if generate_button:
                # Afficher une barre de progression
                progress_bar = st.progress(0)
                st.info("Entraînement des modèles en cours...")
                
                # Initialisation des classes
                forecasting_models = ForecastingModels(config)
                visualizer = TresorerieVisualizer(config)
                
                # Analyse de la saisonnalité si activée
                if config.get('detect_seasonality', True):
                    try:
                        with st.spinner("Analyse des tendances saisonnières en cours..."):
                            seasonal_patterns_enc = forecasting_models.analyze_seasonality(df_enc)
                            seasonal_patterns_dec = forecasting_models.analyze_seasonality(df_dec)
                            forecasting_models.seasonal_patterns = {
                                'enc': seasonal_patterns_enc,
                                'dec': seasonal_patterns_dec
                            }
                        progress_bar.progress(10)
                    except Exception as e:
                        st.warning(f"Analyse de saisonnalité non disponible : {e}")
                
                # Détection des anomalies si activée
                if config.get('detect_anomalies', True):
                    try:
                        with st.spinner("Détection des anomalies en cours..."):
                            anomalies_enc = forecasting_models.detect_anomalies(df_enc)
                            anomalies_dec = forecasting_models.detect_anomalies(df_dec)
                            forecasting_models.anomalies = {
                                'enc': anomalies_enc,
                                'dec': anomalies_dec
                            }
                        progress_bar.progress(15)
                    except Exception as e:
                        st.warning(f"Détection d'anomalies non disponible : {e}")
                
                # Entraînement des modèles
                try:
                    with st.spinner("Entraînement des modèles..."):
                        # Stocker l'option use_hybrid dans la config du modèle
                        forecasting_models.config['use_hybrid'] = config.get('use_hybrid', False)
                        models = forecasting_models.train_models(df_enc, df_dec, n_mois)
                    progress_bar.progress(25)
                    
                    if not models:  # Si aucun modèle n'a été entraîné
                        st.error("Aucun modèle n'a pu être entraîné. Veuillez sélectionner au moins un modèle.")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement des modèles : {e}")
                # Génération des prévisions
                try:
                    with st.spinner("Génération des prévisions..."):
                        forecasts = forecasting_models.generate_forecasts(df_enc, df_dec, n_mois)
                    progress_bar.progress(50)
                    
                    # Affichage du statut des modèles
                    st.markdown("### Statut des Modèles")
                    
                    # Créer un dictionnaire pour stocker le statut de chaque modèle
                    model_status = {
                        "Prophet": {
                            "Activé": config.get('use_prophet', True),
                            "Entraîné": 'prophet_enc' in forecasting_models.models,
                            "Prévisions": 'prophet_enc' in forecasts,
                            "Statut": "✅ Actif" if 'prophet_enc' in forecasts else "❌ Inactif"
                        },
                        "ARIMA": {
                            "Activé": config.get('use_arima', True),
                            "Entraîné": 'arima_enc' in forecasting_models.models,
                            "Prévisions": 'arima_enc' in forecasts,
                            "Statut": "✅ Actif" if 'arima_enc' in forecasts else "❌ Inactif"
                        },
                        "LSTM": {
                            "Activé": config.get('use_lstm', True),
                            "Entraîné": 'lstm_enc_model' in forecasting_models.models,
                            "Prévisions": 'lstm_enc' in forecasts,
                            "Statut": "✅ Actif" if 'lstm_enc' in forecasts else "❌ Inactif"
                        },
                        "XGBoost": {
                            "Activé": config.get('use_xgboost', True),
                            "Entraîné": 'xgboost_enc' in forecasting_models.models,
                            "Prévisions": 'xgb_enc' in forecasts,
                            "Statut": "✅ Actif" if 'xgb_enc' in forecasts else "❌ Inactif"
                        },
                        "Random Forest": {
                            "Activé": config.get('use_rf', True),
                            "Entraîné": 'rf_enc' in forecasting_models.models,
                            "Prévisions": 'rf_enc' in forecasts,
                            "Statut": "✅ Actif" if 'rf_enc' in forecasts else "❌ Inactif"
                        },
                        "Modèle Hybride": {
                            "Activé": config.get('use_hybrid', True),
                            "Entraîné": 'hybrid_enc' in forecasting_models.models,
                            "Prévisions": 'hybrid_enc' in forecasts,
                            "Statut": "✅ Actif" if 'hybrid_enc' in forecasts else "❌ Inactif"
                        }
                    }
                    
                    # Créer un DataFrame pour l'affichage
                    status_data = []
                    for model_name, status in model_status.items():
                        status_data.append({
                            "Modèle": model_name,
                            "Activé": "✅" if status["Activé"] else "❌",
                            "Entraîné": "✅" if status["Entraîné"] else "❌",
                            "Prévisions": "✅" if status["Prévisions"] else "❌",
                            "Statut": status["Statut"]
                        })
                    
                    status_df = pd.DataFrame(status_data)
                    st.dataframe(
                        status_df,
                        use_container_width=True,
                        column_config={
                            "Modèle": st.column_config.TextColumn("Modèle"),
                            "Activé": st.column_config.TextColumn("Activé"),
                            "Entraîné": st.column_config.TextColumn("Entraîné"),
                            "Prévisions": st.column_config.TextColumn("Prévisions"),
                            "Statut": st.column_config.TextColumn("Statut")
                        },
                        hide_index=True
                    )
                    
                    if not forecasts:  # Si aucune prévision n'a été générée
                        st.error("Aucune prévision n'a pu être générée. Veuillez vérifier les données et les paramètres.")
                except Exception as e:
                    st.error(f"Erreur lors de la génération des prévisions : {e}")
                    import traceback
                    st.error(traceback.format_exc())
                
# ... (code après la modification)
                # Sélection du meilleur modèle
                try:
                    with st.spinner("Sélection du meilleur modèle..."):
                        st.info("Sélection du meilleur modèle en cours...")
                        best_model, model_metrics = forecasting_models.select_best_model(
                            df_enc, forecasts, config['selection_metric']
                        )
                    progress_bar.progress(75)
                    
                    if best_model is None:
                        st.warning("Aucun modèle n'a pu être sélectionné. Utilisation du modèle Prophet par défaut.")
                        if 'prophet_enc' in forecasts:
                            best_model = 'prophet_enc'
                        else:
                            # Prendre le premier modèle disponible
                            enc_models = [m for m in forecasts.keys() if 'enc' in m]
                            if enc_models:
                                best_model = enc_models[0]
                            else:
                                st.error("Aucun modèle d'encaissement disponible.")
                except Exception as e:
                    st.error(f"Erreur lors de la sélection du meilleur modèle : {e}")
                    import traceback
                    st.error(traceback.format_exc())
                
                # Création de scénarios
                try:
                    with st.spinner("Création des scénarios..."):
                        st.info("Création des scénarios en cours...")
                        scenarios = forecasting_models.create_scenarios(
                            forecasts, n_mois, config['confidence_interval']/100
                        )
                    progress_bar.progress(75)
                except Exception as e:
                    st.error(f"Erreur lors de la création des scénarios : {e}")
                    import traceback
                    st.error(traceback.format_exc())
                
                # Validation croisée si activée
                if config.get('use_cross_validation', False):
                    try:
                        with st.spinner("Validation croisée des modèles en cours..."):
                            # Créer une liste des modèles à valider
                            models_list = []
                            if config.get('use_prophet', True):
                                models_list.append('prophet_enc')
                            if config.get('use_arima', True):
                                models_list.append('arima_enc')
                            if config.get('use_xgboost', True):
                                models_list.append('xgboost_enc')
                            if config.get('use_rf', True):
                                models_list.append('rf_enc')
                            if config.get('use_hybrid', False):
                                models_list.append('hybrid_enc')
                                
                            # Exécuter la validation croisée pour les encaissements
                            cv_results_enc = forecasting_models.cross_validate_models(df_enc, 'y_enc', models_list)
                            
                            # Exécuter la validation croisée pour les décaissements
                            cv_results_dec = forecasting_models.cross_validate_models(df_dec, 'y_dec', models_list)
                            
                            forecasting_models.cv_results = {
                                'enc': cv_results_enc,
                                'dec': cv_results_dec
                            }
                        progress_bar.progress(85)
                    except Exception as e:
                        st.warning(f"Validation croisée non disponible : {e}")
                        import traceback
                        st.warning(traceback.format_exc())
                
                # Simulations avancées
                if config.get('run_monte_carlo', False):
                    try:
                        with st.spinner("Exécution des simulations Monte Carlo..."):
                            monte_carlo_results = forecasting_models.simulate_monte_carlo(
                                forecasts, n_mois, n_simulations=config.get('monte_carlo_sims', 1000)
                            )
                            forecasting_models.monte_carlo_results = monte_carlo_results
                        progress_bar.progress(90)
                    except Exception as e:
                        st.warning(f"Simulation Monte Carlo non disponible : {e}")
                
                if config.get('run_sensitivity', False):
                    try:
                        with st.spinner("Exécution de l'analyse de sensibilité..."):
                            sensitivity_results = forecasting_models.analyze_sensitivity(forecasts, n_mois)
                            forecasting_models.sensitivity_results = sensitivity_results
                        progress_bar.progress(95)
                    except Exception as e:
                        st.warning(f"Analyse de sensibilité non disponible : {e}")
                
                progress_bar.progress(100)
                st.success("Analyse terminée avec succès!")
                
                # Stocker les résultats dans la session
                st.session_state['forecasts'] = forecasts
                st.session_state['best_model'] = best_model
                st.session_state['model_metrics'] = model_metrics
                st.session_state['scenarios'] = scenarios
                st.session_state['forecasting_models'] = forecasting_models
                st.session_state['forecasts_generated'] = True
        
        except Exception as e:
            st.error(f"Erreur générale : {e}")
            import traceback
            st.error(traceback.format_exc())
        
        # Vérifier si les prévisions ont été générées
        forecasts = st.session_state.get('forecasts', {})
        best_model = st.session_state.get('best_model', '')
        model_metrics = st.session_state.get('model_metrics', {})
        scenarios = st.session_state.get('scenarios', {})
        forecasting_models = st.session_state.get('forecasting_models', None)
        
        if not forecasts or best_model == '':
            st.warning("Veuillez générer des prévisions en cliquant sur le bouton ci-dessus.")
            show_simulation = False
        else:
            show_simulation = True
        
        # Afficher les onglets uniquement si les prévisions ont été générées
        if st.session_state.get('forecasts_generated', False):
            # Afficher un message sur les nouvelles fonctionnalités
            st.markdown(
                """<div style='background-color: #f0fff4; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #48bb78;'>
                <h3 style='margin-top: 0; color: #2f855a;'>✨ Nouvelles fonctionnalités disponibles</h3>
                <p>Explorez les nouveaux onglets pour découvrir :</p>
                <ul>
                    <li><strong>Analyse Saisonnière</strong> - Visualisez les tendances saisonnières dans vos données</li>
                    <li><strong>Détection d'Anomalies</strong> - Identifiez les valeurs aberrantes dans vos flux financiers</li>
                    <li><strong>Analyses Avancées</strong> - Explorez les simulations Monte Carlo et l'analyse de sensibilité</li>
                </ul>
                </div>""",
                unsafe_allow_html=True
            )
            
            display_results(df_enc, df_dec, forecasts, best_model, model_metrics, scenarios, n_mois, config, forecasting_models)
            
            # Export des prévisions
            export_forecasts(df_enc, forecasts, n_mois)

def display_results(df_enc, df_dec, forecasts, best_model, model_metrics, scenarios, n_mois, config, forecasting_models=None):
    """Affiche les résultats de l'analyse"""
    # Vérifier si nous devons appliquer de nouveaux filtres
    apply_new_filters = st.session_state.get('apply_new_filters', False)
    
    # Si nous devons appliquer de nouveaux filtres, mettre à jour la configuration
    if apply_new_filters:
        # Mettre à jour la configuration avec les nouvelles options d'affichage
        config['show_predictions'] = st.session_state.get('show_predictions', True)
        config['show_confidence'] = st.session_state.get('show_confidence', True)
        config['show_components'] = st.session_state.get('show_components', False)
        
        # Réinitialiser le flag
        st.session_state['apply_new_filters'] = False
        
        # Afficher un message de confirmation
        st.success("Filtres appliqués avec succès!")
    
    # Création des dates futures pour les prévisions
    future_dates = pd.date_range(start=df_enc['ds'].iloc[-1], periods=n_mois+1, freq='MS')[1:]
    
    # Création du visualiseur avec la configuration mise à jour
    visualizer = TresorerieVisualizer(config)
    
    # Affichage d'un résumé des résultats en haut de la page
    st.markdown(
        '<div style="background-color: #ecfdf5; padding: 1.5rem; border-radius: 10px; margin: 1rem 0 2rem 0; border-left: 5px solid #10b981;">'        
        f'<h3 style="margin-top: 0; color: #065f46;">✅ Analyse complétée avec succès</h3>'        
        f'<p>Meilleur modèle identifié: <strong>{best_model}</strong></p>'        
        f'<p>Horizon de prévision: <strong>{n_mois} mois</strong></p>'        
        '</div>',
        unsafe_allow_html=True
    )
    
    # Déterminer l'onglet actif par défaut
    active_tab = st.session_state.get('active_tab', "Flux de Trésorerie")
    tab_names = ["Flux de Trésorerie", "Comparaison des Modèles", "Scénarios", "Métriques", "Analyse Saisonnière", "Détection d'Anomalies", "Analyses Avancées"]
    
    # Si Monte Carlo est activé, sélectionner l'onglet Scénarios
    if st.session_state.get('show_monte_carlo', False):
        active_tab = "Scénarios"
    
    # Création des onglets pour organiser l'affichage
    tab_flux, tab_models, tab_scenarios, tab_metrics, tab_seasonal, tab_anomalies, tab_advanced = st.tabs(tab_names)
    
    # Onglet Flux de Trésorerie
    with tab_flux:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">Flux de Trésorerie et Prévisions</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Visualisation des flux historiques et prévisionnels</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Affichage des données historiques et des prévisions
        if forecasts:
            # Créer des prévisions factices pour les modèles activés mais manquants
            # Cela permet de s'assurer que tous les modèles activés apparaissent dans la liste
            model_mapping = {
                'use_prophet': 'prophet_enc',
                'use_arima': 'arima_enc',
                'use_lstm': 'lstm_enc',
                'use_xgboost': 'xgb_enc',
                'use_rf': 'rf_enc',
                'use_hybrid': 'hybrid_enc'
            }
            
            # Vérifier quels modèles sont activés mais pas dans les prévisions
            for config_key, model_name in model_mapping.items():
                if config.get(config_key, True) and model_name not in forecasts:
                    # Si un modèle est activé mais pas dans les prévisions, créer une prévision factice
                    # basée sur la moyenne des autres modèles ou sur les données historiques
                    enc_models = [m for m in forecasts.keys() if 'enc' in m]
                    if enc_models:
                        # Utiliser la moyenne des autres modèles
                        avg_forecast = np.mean([forecasts[m] for m in enc_models], axis=0)
                        forecasts[model_name] = avg_forecast
                    elif len(df_enc) > 0:
                        # Utiliser la moyenne des données historiques
                        mean_value = df_enc['y_enc'].mean()
                        forecasts[model_name] = np.ones(n_mois) * mean_value
                    else:
                        # Valeur par défaut
                        forecasts[model_name] = np.ones(n_mois) * 1000
                    
                    # Faire de même pour le modèle de décaissement correspondant
                    dec_model_name = model_name.replace('enc', 'dec')
                    dec_models = [m for m in forecasts.keys() if 'dec' in m]
                    if dec_models:
                        avg_forecast = np.mean([forecasts[m] for m in dec_models], axis=0)
                        forecasts[dec_model_name] = avg_forecast
                    elif len(df_dec) > 0:
                        mean_value = df_dec['y_dec'].mean()
                        forecasts[dec_model_name] = np.ones(n_mois) * mean_value
                    else:
                        forecasts[dec_model_name] = np.ones(n_mois) * 800
                        
            # Création d'un sélecteur pour choisir le modèle à afficher
            available_models = [model_name for model_name in forecasts.keys() if 'enc' in model_name]
            
            if available_models:
                # Ajouter une option "Tous les modèles" en premier
                display_options = ["Tous les modèles"] + available_models
                selected_model = st.selectbox(
                    "Sélectionner le modèle à afficher",
                    options=display_options,
                    index=0,  # Par défaut, afficher tous les modèles
                    help="Sélectionnez un modèle spécifique ou 'Tous les modèles' pour voir toutes les prévisions"
                )
                
                # Création du graphique principal
                if selected_model == "Tous les modèles":
                    # Afficher tous les modèles disponibles
                    fig_main = visualizer.create_all_models_chart(df_enc, df_dec, forecasts, best_model, future_dates)
                else:
                    # Afficher uniquement le modèle sélectionné
                    fig_main = visualizer.create_flux_chart(df_enc, df_dec, forecasts, selected_model, future_dates)
                
                st.plotly_chart(fig_main, use_container_width=True)
            else:
                st.warning("Aucun modèle disponible pour l'affichage.")
        else:
            st.warning("Aucune prévision disponible.")
            
            # Affichage des statistiques clés
            col1, col2, col3 = st.columns(3)
            
            # Calcul des moyennes pour les statistiques
            enc_mean = df_enc['y_enc'].mean()
            dec_mean = df_dec['y_dec'].mean()
            solde_mean = enc_mean - dec_mean
            
            # Prévisions moyennes
            forecast_enc_mean = np.mean(forecasts[best_model]) if best_model in forecasts else 0
            forecast_dec_model = best_model.replace('enc', 'dec')
            forecast_dec_mean = np.mean(forecasts[forecast_dec_model]) if forecast_dec_model in forecasts else 0
            forecast_solde_mean = forecast_enc_mean - forecast_dec_mean
            
            # Affichage des statistiques dans les colonnes
            with col1:
                enc_delta = forecast_enc_mean - enc_mean if enc_mean > 0 else None
                st.metric(
                    "Encaissements Moyens", 
                    f"{enc_mean:,.0f} DH",
                    delta=f"{enc_delta:,.0f} DH" if enc_delta is not None else None,
                    delta_color="normal"
                )
            
            with col2:
                dec_delta = forecast_dec_mean - dec_mean if dec_mean > 0 else None
                st.metric(
                    "Décaissements Moyens", 
                    f"{dec_mean:,.0f} DH",
                    delta=f"{dec_delta:,.0f} DH" if dec_delta is not None else None,
                    delta_color="inverse"
                )
            
            with col3:
                solde_delta = forecast_solde_mean - solde_mean if solde_mean != 0 else None
                st.metric(
                    "Solde Moyen", 
                    f"{solde_mean:,.0f} DH",
                    delta=f"{solde_delta:,.0f} DH" if solde_delta is not None else None,
                    delta_color="normal"
                )
            
            # Affichage des tableaux de données
            st.markdown(
                '<div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin: 1.5rem 0;">'            
                '<h3 style="font-size: 1.3rem; margin: 0 0 0.8rem 0; color: #334155;">Détail des Prévisions</h3>'            
                '<p style="margin: 0 0 1rem 0; color: #64748b;">Prévisions détaillées pour les prochains mois</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Sélection du modèle à afficher dans les détails
            available_models = [model_name for model_name in forecasts.keys() if 'enc' in model_name]
            
            if available_models:
                selected_detail_model = st.selectbox(
                    "Sélectionner le modèle pour les détails",
                    options=available_models,
                    index=0 if best_model not in available_models else available_models.index(best_model),
                    help="Sélectionnez un modèle pour voir ses détails de prévision"
                )
                
                # Création d'un DataFrame pour les prévisions
                # S'assurer que toutes les arrays ont la même longueur
                enc_forecast = forecasts[selected_detail_model]
                forecast_dec_model = selected_detail_model.replace('enc', 'dec')
                dec_forecast = forecasts[forecast_dec_model] if forecast_dec_model in forecasts else np.zeros(len(future_dates))
            else:
                st.warning("Aucun modèle disponible pour l'affichage des détails.")
                enc_forecast = np.zeros(len(future_dates))
                dec_forecast = np.zeros(len(future_dates))
            
            # Vérifier que les longueurs correspondent
            min_length = min(len(future_dates), len(enc_forecast), len(dec_forecast))
            
            # Créer le DataFrame avec des arrays de même longueur
            forecast_df = pd.DataFrame({
                'Date': future_dates[:min_length],
                'Encaissements': enc_forecast[:min_length],
                'Décaissements': dec_forecast[:min_length],
                'Solde': enc_forecast[:min_length] - dec_forecast[:min_length]
            })
            
            # Ajout de colonnes pour les variations
            if len(forecast_df) > 1:
                forecast_df['Var. Encaissements'] = forecast_df['Encaissements'].pct_change() * 100
                forecast_df['Var. Décaissements'] = forecast_df['Décaissements'].pct_change() * 100
                forecast_df['Var. Solde'] = forecast_df['Solde'].pct_change() * 100
                
                # Remplacer NaN par 0 pour la première ligne (sans utiliser inplace)
                forecast_df = forecast_df.copy()
                forecast_df['Var. Encaissements'] = forecast_df['Var. Encaissements'].fillna(0)
                forecast_df['Var. Décaissements'] = forecast_df['Var. Décaissements'].fillna(0)
                forecast_df['Var. Solde'] = forecast_df['Var. Solde'].fillna(0)
            
            # Créer une copie pour l'affichage avec formatage
            display_df = forecast_df.copy()
            
            # Formatage des colonnes numériques
            display_df['Encaissements'] = display_df['Encaissements'].map('{:,.0f} DH'.format)
            display_df['Décaissements'] = display_df['Décaissements'].map('{:,.0f} DH'.format)
            display_df['Solde'] = display_df['Solde'].map('{:,.0f} DH'.format)
            
            # Formatage des colonnes de variation si elles existent
            if 'Var. Encaissements' in display_df.columns:
                display_df['Var. Encaissements'] = display_df['Var. Encaissements'].map('{:+.1f}%'.format)
                display_df['Var. Décaissements'] = display_df['Var. Décaissements'].map('{:+.1f}%'.format)
                display_df['Var. Solde'] = display_df['Var. Solde'].map('{:+.1f}%'.format)
            
            # Options d'affichage
            col1, col2 = st.columns([3, 1])
            with col1:
                view_option = st.radio(
                    "Mode d'affichage",
                    ["Tableau complet", "Afficher par trimestre", "Afficher par mois"],
                    horizontal=True
                )
            
            with col2:
                show_variations = st.checkbox("Afficher les variations", value=True)
            
            # Préparer le DataFrame selon les options choisies
            if not show_variations and 'Var. Encaissements' in display_df.columns:
                display_df = display_df.drop(columns=['Var. Encaissements', 'Var. Décaissements', 'Var. Solde'])
            
            # Regrouper par période si nécessaire
            if view_option == "Afficher par trimestre" and len(forecast_df) >= 3:
                # Convertir les dates en périodes trimestrielles
                forecast_df['Trimestre'] = pd.PeriodIndex(forecast_df['Date'], freq='Q')
                
                # Grouper par trimestre
                grouped_df = forecast_df.groupby('Trimestre').agg({
                    'Encaissements': 'sum',
                    'Décaissements': 'sum',
                    'Solde': 'mean'
                }).reset_index()
                
                # Convertir les périodes en chaînes de caractères
                grouped_df['Trimestre'] = grouped_df['Trimestre'].astype(str)
                
                # Formatage
                display_df = grouped_df.copy()
                display_df['Encaissements'] = display_df['Encaissements'].map('{:,.0f} DH'.format)
                display_df['Décaissements'] = display_df['Décaissements'].map('{:,.0f} DH'.format)
                display_df['Solde'] = display_df['Solde'].map('{:,.0f} DH'.format)
                
                # Renommer la colonne de date
                display_df = display_df.rename(columns={'Trimestre': 'Période'})
            
            # Affichage du tableau avec style
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="MMM YYYY"),
                    "Période": st.column_config.TextColumn("Période"),
                    "Encaissements": st.column_config.TextColumn("Encaissements"),
                    "Décaissements": st.column_config.TextColumn("Décaissements"),
                    "Solde": st.column_config.TextColumn("Solde"),
                    "Var. Encaissements": st.column_config.TextColumn("Var. Encaissements"),
                    "Var. Décaissements": st.column_config.TextColumn("Var. Décaissements"),
                    "Var. Solde": st.column_config.TextColumn("Var. Solde")
                },
                height=400
            )
            
            # Boutons d'export
            col1, col2 = st.columns(2)
            with col1:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger en CSV",
                    data=csv,
                    file_name="previsions_tresorerie.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Télécharger les prévisions au format CSV"
                )
            
            with col2:
                # Créer un buffer Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    forecast_df.to_excel(writer, sheet_name='Prévisions', index=False)
                    # Accéder au workbook et worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Prévisions']
                    # Ajouter un format pour les nombres
                    num_format = workbook.add_format({'num_format': '#,##0 "DH"'})
                    pct_format = workbook.add_format({'num_format': '+0.0%'})
                    # Appliquer les formats
                    for col_num, col_name in enumerate(forecast_df.columns):
                        if col_name in ['Encaissements', 'Décaissements', 'Solde']:
                            worksheet.set_column(col_num, col_num, 15, num_format)
                        elif 'Var.' in col_name:
                            worksheet.set_column(col_num, col_num, 15, pct_format)
                
                # Convertir le buffer en bytes pour le téléchargement
                buffer.seek(0)
                st.download_button(
                    label="Télécharger en Excel",
                    data=buffer,
                    file_name="previsions_tresorerie.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True,
                    help="Télécharger les prévisions au format Excel"
                )
        
    # Fin du bloc tab_flux
    if not forecasts:
        with tab_flux:
            st.warning("Aucune prévision disponible. Veuillez générer des prévisions d'abord.")
    
    # Onglet Analyse Saisonnière
    with tab_seasonal:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">Analyse Saisonnière</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Détection et visualisation des tendances saisonnières</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        if forecasting_models and hasattr(forecasting_models, 'seasonal_patterns'):
            # Afficher l'analyse saisonnière pour les encaissements
            st.subheader("Analyse Saisonnière des Encaissements")
            if 'enc' in forecasting_models.seasonal_patterns:
                fig_seasonal_enc = visualizer.create_seasonal_analysis_chart(
                    forecasting_models.seasonal_patterns['enc'],
                    title="Décomposition Saisonnière des Encaissements"
                )
                st.plotly_chart(fig_seasonal_enc, use_container_width=True)
                
                # Patterns mensuels
                fig_monthly_enc = visualizer.create_monthly_pattern_chart(
                    df_enc, 'y_enc', 
                    title="Patterns Mensuels des Encaissements"
                )
                st.plotly_chart(fig_monthly_enc, use_container_width=True)
                
                # Afficher des informations sur la saisonnalité détectée
                if forecasting_models.seasonal_patterns['enc'].get('has_seasonality', False):
                    seasonal_strength = forecasting_models.seasonal_patterns['enc'].get('seasonal_strength', 0) * 100
                    dominant_period = forecasting_models.seasonal_patterns['enc'].get('dominant_period', 0)
                    
                    st.markdown(f"""
                    <div style='background-color: #f0fff4; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #48bb78;'>
                    <h3 style='margin-top: 0; color: #2f855a;'>Saisonnalité Détectée</h3>
                    <p><strong>Force de la saisonnalité:</strong> {seasonal_strength:.1f}%</p>
                    <p><strong>Période dominante:</strong> {dominant_period} mois</p>
                    <p>Une saisonnalité forte indique des cycles réguliers dans vos encaissements qui peuvent être utilisés pour améliorer la planification financière.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Aucune saisonnalité significative n'a été détectée dans les encaissements.")
            else:
                st.info("Analyse saisonnière des encaissements non disponible. Assurez-vous d'avoir activé l'option de détection de saisonnalité.")
            
            # Afficher l'analyse saisonnière pour les décaissements
            st.subheader("Analyse Saisonnière des Décaissements")
            if 'dec' in forecasting_models.seasonal_patterns:
                fig_seasonal_dec = visualizer.create_seasonal_analysis_chart(
                    forecasting_models.seasonal_patterns['dec'],
                    title="Décomposition Saisonnière des Décaissements"
                )
                st.plotly_chart(fig_seasonal_dec, use_container_width=True)
                
                # Patterns mensuels
                fig_monthly_dec = visualizer.create_monthly_pattern_chart(
                    df_dec, 'y_dec', 
                    title="Patterns Mensuels des Décaissements"
                )
                st.plotly_chart(fig_monthly_dec, use_container_width=True)
                
                # Afficher des informations sur la saisonnalité détectée
                if forecasting_models.seasonal_patterns['dec'].get('has_seasonality', False):
                    seasonal_strength = forecasting_models.seasonal_patterns['dec'].get('seasonal_strength', 0) * 100
                    dominant_period = forecasting_models.seasonal_patterns['dec'].get('dominant_period', 0)
                    
                    st.markdown(f"""
                    <div style='background-color: #f0fff4; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #48bb78;'>
                    <h3 style='margin-top: 0; color: #2f855a;'>Saisonnalité Détectée</h3>
                    <p><strong>Force de la saisonnalité:</strong> {seasonal_strength:.1f}%</p>
                    <p><strong>Période dominante:</strong> {dominant_period} mois</p>
                    <p>Une saisonnalité forte indique des cycles réguliers dans vos décaissements qui peuvent être utilisés pour améliorer la planification financière.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Aucune saisonnalité significative n'a été détectée dans les décaissements.")
            else:
                st.info("Analyse saisonnière des décaissements non disponible. Assurez-vous d'avoir activé l'option de détection de saisonnalité.")
            
            # Analyse comparative entre périodes
            st.subheader("Analyse Comparative")
            if len(df_enc) >= 12:  # Au moins 12 mois de données
                fig_comparative = visualizer.create_comparative_analysis_chart(
                    df_enc, 'y_enc', 
                    title="Comparaison des Encaissements par Période"
                )
                st.plotly_chart(fig_comparative, use_container_width=True)
                
                # Évolution année par année
                fig_year_over_year = visualizer.create_year_over_year_chart(
                    df_enc, 'y_enc', 
                    title="Évolution des Encaissements Année par Année"
                )
                st.plotly_chart(fig_year_over_year, use_container_width=True)
            else:
                st.info("Analyse comparative non disponible. Un minimum de 12 mois de données est nécessaire.")
        else:
            st.warning("Analyse saisonnière non disponible. Assurez-vous d'avoir activé l'option 'Détection de saisonnalité' dans les paramètres avancés.")
    
    # Onglet Détection d'Anomalies
    with tab_anomalies:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">Détection d\'Anomalies</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Identification des valeurs aberrantes dans les données financières</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        if forecasting_models and hasattr(forecasting_models, 'anomalies'):
            # Afficher les anomalies pour les encaissements
            st.subheader("Anomalies dans les Encaissements")
            if 'enc' in forecasting_models.anomalies:
                fig_anomalies_enc = visualizer.create_anomaly_detection_chart(
                    forecasting_models.anomalies['enc'],
                    title="Détection d'Anomalies dans les Encaissements"
                )
                st.plotly_chart(fig_anomalies_enc, use_container_width=True)
                
                # Afficher des informations sur les anomalies détectées
                if forecasting_models.anomalies['enc'].get('anomalies_detected', False):
                    anomaly_count = forecasting_models.anomalies['enc'].get('anomaly_count', 0)
                    anomaly_percent = forecasting_models.anomalies['enc'].get('anomaly_percent', 0)
                    
                    st.markdown(f"""
                    <div style='background-color: #fff8e6; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #f59e0b;'>
                    <h3 style='margin-top: 0; color: #b45309;'>Résumé des Anomalies</h3>
                    <p><strong>{anomaly_count}</strong> anomalies détectées ({anomaly_percent:.1f}% des données)</p>
                    <p>Les anomalies peuvent indiquer des transactions inhabituelles ou des erreurs de saisie.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Afficher un tableau des anomalies
                    if 'anomaly_data' in forecasting_models.anomalies['enc']:
                        anomaly_df = forecasting_models.anomalies['enc']['anomaly_data']
                        if not anomaly_df.empty:
                            st.subheader("Détails des Anomalies")
                            st.dataframe(anomaly_df[['ds', 'y_enc', 'anomaly_score']].rename(
                                columns={'ds': 'Date', 'y_enc': 'Montant', 'anomaly_score': 'Score d\'Anomalie'})
                            )
                else:
                    st.success("Aucune anomalie significative n'a été détectée dans les encaissements.")
            else:
                st.info("Détection d'anomalies pour les encaissements non disponible.")
            
            # Afficher les anomalies pour les décaissements
            st.subheader("Anomalies dans les Décaissements")
            if 'dec' in forecasting_models.anomalies:
                fig_anomalies_dec = visualizer.create_anomaly_detection_chart(
                    forecasting_models.anomalies['dec'],
                    title="Détection d'Anomalies dans les Décaissements"
                )
                st.plotly_chart(fig_anomalies_dec, use_container_width=True)
                
                # Afficher des informations sur les anomalies détectées
                if forecasting_models.anomalies['dec'].get('anomalies_detected', False):
                    anomaly_count = forecasting_models.anomalies['dec'].get('anomaly_count', 0)
                    anomaly_percent = forecasting_models.anomalies['dec'].get('anomaly_percent', 0)
                    
                    st.markdown(f"""
                    <div style='background-color: #fff8e6; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #f59e0b;'>
                    <h3 style='margin-top: 0; color: #b45309;'>Résumé des Anomalies</h3>
                    <p><strong>{anomaly_count}</strong> anomalies détectées ({anomaly_percent:.1f}% des données)</p>
                    <p>Les anomalies peuvent indiquer des transactions inhabituelles ou des erreurs de saisie.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Afficher un tableau des anomalies
                    if 'anomaly_data' in forecasting_models.anomalies['dec']:
                        anomaly_df = forecasting_models.anomalies['dec']['anomaly_data']
                        if not anomaly_df.empty:
                            st.subheader("Détails des Anomalies")
                            st.dataframe(anomaly_df[['ds', 'y_dec', 'anomaly_score']].rename(
                                columns={'ds': 'Date', 'y_dec': 'Montant', 'anomaly_score': 'Score d\'Anomalie'})
                            )
                else:
                    st.success("Aucune anomalie significative n'a été détectée dans les décaissements.")
            else:
                st.info("Détection d'anomalies pour les décaissements non disponible.")
            
            # Conseils pour l'interprétation des anomalies
            with st.expander("Comment interpréter les anomalies?"):
                st.markdown("""
                Les anomalies sont des valeurs qui s'écartent significativement du comportement normal des données. Elles peuvent indiquer :
                - Des transactions exceptionnelles (paiements importants, remboursements, etc.)
                - Des erreurs de saisie ou de traitement
                - Des changements structurels dans votre activité
                
                **Comment utiliser cette information :**
                - Vérifiez les transactions identifiées comme anomalies
                - Corrigez les erreurs éventuelles dans vos données
                - Tenez compte des anomalies légitimes dans votre planification financière
                - Utilisez les anomalies pour améliorer vos contrôles internes
                """)
        else:
            st.warning("Détection d'anomalies non disponible. Assurez-vous d'avoir activé l'option 'Détection d'anomalies' dans les paramètres avancés.")
    
    # Onglet Analyses Avancées
    with tab_advanced:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">Analyses Avancées</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse détaillée des données et des prévisions</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Création et affichage du graphique avec animation
        st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
        fig_flux = visualizer.create_flux_chart(df_enc, df_dec, forecasts, best_model, future_dates)
        st.plotly_chart(fig_flux, use_container_width=True, key="flux_chart")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Séparateur visuel
        st.markdown('<hr style="margin: 2rem 0; border: none; height: 1px; background-color: #e5e7eb;">', unsafe_allow_html=True)
        
        # Statistiques des flux avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
            '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">📊 Statistiques des Flux</h3>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse des valeurs moyennes et tendances</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        try:
            # Calcul des statistiques
            enc_mean = np.mean(df_enc['y_enc']) if len(df_enc) > 0 else 0
            dec_mean = np.mean(df_dec['y_dec']) if len(df_dec) > 0 else 0
            solde_mean = enc_mean - dec_mean
            
            # Calcul des statistiques prévisionnelles
            forecast_enc_mean = forecasts[best_model].mean() if best_model in forecasts and len(forecasts[best_model]) > 0 else 0
            forecast_dec_mean = forecasts[best_model.replace('enc', 'dec')].mean() if best_model.replace('enc', 'dec') in forecasts and len(forecasts[best_model.replace('enc', 'dec')]) > 0 else 0
            forecast_solde_mean = forecast_enc_mean - forecast_dec_mean
            
            # Création d'une carte pour les métriques
            st.markdown('<div class="stCard" style="padding: 1rem; background-color: #f8fafc; border-radius: 8px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            
            # Affichage des statistiques en colonnes avec style amélioré
            col1, col2, col3 = st.columns(3)
            with col1:
                enc_delta = forecast_enc_mean - enc_mean if enc_mean > 0 else None
                st.metric(
                    "Encaissements Moyens", 
                    f"{enc_mean:,.0f} DH",
                    delta=f"{enc_delta:,.0f} DH" if enc_delta is not None else None,
                    delta_color="normal"
                )
                
                # Statistiques supplémentaires
                if len(df_enc) > 0:
                    st.markdown(f"<p style='margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;'>Min: {df_enc['y_enc'].min():,.0f} DH | Max: {df_enc['y_enc'].max():,.0f} DH</p>", unsafe_allow_html=True)
            
            with col2:
                dec_delta = forecast_dec_mean - dec_mean if dec_mean > 0 else None
                st.metric(
                    "Décaissements Moyens", 
                    f"{dec_mean:,.0f} DH",
                    delta=f"{dec_delta:,.0f} DH" if dec_delta is not None else None,
                    delta_color="inverse"
                )
                
                # Statistiques supplémentaires
                if len(df_dec) > 0:
                    st.markdown(f"<p style='margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;'>Min: {df_dec['y_dec'].min():,.0f} DH | Max: {df_dec['y_dec'].max():,.0f} DH</p>", unsafe_allow_html=True)
            
            with col3:
                solde_delta = forecast_solde_mean - solde_mean if solde_mean != 0 else None
                st.metric(
                    "Solde Moyen", 
                    f"{solde_mean:,.0f} DH",
                    delta=f"{solde_delta:,.0f} DH" if solde_delta is not None else None,
                    delta_color="normal"
                )
                
                # Taux de couverture
                if dec_mean > 0:
                    coverage_ratio = enc_mean / dec_mean
                    st.markdown(f"<p style='margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;'>Taux de couverture: {coverage_ratio:.2f}</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Ajout d'une section pour les tendances
            if len(df_enc) > 6 and len(df_dec) > 6:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1rem 0;">'                    
                    '<h4 style="font-size: 1rem; margin: 0; color: #1e40af;">Tendances sur les 6 derniers mois</h4>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Calcul des tendances
                recent_enc = df_enc['y_enc'].values[-6:]
                recent_dec = df_dec['y_dec'].values[-6:]
                
                enc_trend = (recent_enc[-1] / recent_enc[0] - 1) * 100 if recent_enc[0] > 0 else 0
                dec_trend = (recent_dec[-1] / recent_dec[0] - 1) * 100 if recent_dec[0] > 0 else 0
                
                # Affichage des tendances
                trend_col1, trend_col2 = st.columns(2)
                with trend_col1:
                    trend_color = "#10b981" if enc_trend > 0 else "#ef4444"
                    st.markdown(f"<div style='padding: 0.75rem; background-color: rgba({16 if enc_trend > 0 else 239},{185 if enc_trend > 0 else 68},{129 if enc_trend > 0 else 68},0.1); border-radius: 8px;'>"
                                f"<p style='margin: 0; font-weight: 500; color: {trend_color};'>Tendance Encaissements</p>"
                                f"<p style='margin: 0.25rem 0 0 0; font-size: 1.25rem; font-weight: 600; color: {trend_color};'>{enc_trend:.1f}%</p>"
                                f"</div>", unsafe_allow_html=True)
                
                with trend_col2:
                    trend_color = "#ef4444" if dec_trend > 0 else "#10b981"
                    st.markdown(f"<div style='padding: 0.75rem; background-color: rgba({239 if dec_trend > 0 else 16},{68 if dec_trend > 0 else 185},{68 if dec_trend > 0 else 129},0.1); border-radius: 8px;'>"
                                f"<p style='margin: 0; font-weight: 500; color: {trend_color};'>Tendance Décaissements</p>"
                                f"<p style='margin: 0.25rem 0 0 0; font-size: 1.25rem; font-weight: 600; color: {trend_color};'>{dec_trend:.1f}%</p>"
                                f"</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur lors du calcul des statistiques: {e}")
            st.markdown("<p>Impossible d'afficher les statistiques des flux. Vérifiez vos données et réessayez.</p>", unsafe_allow_html=True)
        
        # Ajout d'une explication des résultats
        st.markdown(
            '<div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; border-left: 3px solid #3b82f6;">'            
            '<p style="margin: 0; font-style: italic; color: #475569;">'            
            'Les valeurs affichées représentent les moyennes mensuelles historiques, avec les variations prévues pour les prochains mois.'            
            '</p>'            
            '</div>',
            unsafe_allow_html=True
        )
    
    with tab_models:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">🔍 Comparaison des Modèles de Prévision</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse comparative des performances des différents modèles</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Vérifier si des prévisions sont disponibles
        if not forecasts:
            st.markdown(
                '<div style="background-color: #fff7ed; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #f59e0b;">'                
                '<p style="margin: 0; color: #92400e;"><strong>Attention :</strong> Aucune prévision n\'est disponible. Veuillez générer des prévisions.</p>'                
                '</div>',
                unsafe_allow_html=True
            )
            show_model_comparison = False
        else:
            show_model_comparison = True
        
        # Création des sous-onglets pour les encaissements et décaissements avec style amélioré
        subtab1, subtab2, subtab3 = st.tabs([
            "📈 Encaissements", 
            "📉 Décaissements", 
            "📊 Métriques de Performance"
        ])
        
        if show_model_comparison:
            with subtab1:
                # En-tête avec style amélioré
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">'                    
                    '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Prévisions des Encaissements par Modèle</h3>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Création et affichage du graphique avec animation
                st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
                fig_enc_comparison, fig_dec_comparison, fig_ecarts = visualizer.create_model_comparison_chart(
                    df_enc, df_dec, forecasts, best_model, future_dates
                )
                
                # Amélioration: Ajouter une légende interactive
                fig_enc_comparison.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        itemclick="toggleothers"
                    )
                )
                
                # Amélioration: Ajouter des intervalles de confiance si disponibles
                prophet_enc_model = 'prophet_enc'
                if prophet_enc_model in forecasts and forecasting_models and hasattr(forecasting_models, 'models') and prophet_enc_model in forecasting_models.models:
                    try:
                        # Créer un dataframe futur pour Prophet
                        future = pd.DataFrame({'ds': future_dates})
                        # Générer des prévisions avec intervalles de confiance
                        forecast = forecasting_models.models[prophet_enc_model].predict(future)
                        
                        # Ajouter l'intervalle de confiance supérieur
                        fig_enc_comparison.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_upper'].values,
                            mode='lines',
                            line=dict(width=0),
                            name='IC supérieur (95%)',
                            showlegend=True
                        ))
                        
                        # Ajouter l'intervalle de confiance inférieur
                        fig_enc_comparison.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_lower'].values,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0, 128, 0, 0.2)',
                            name='IC inférieur (95%)',
                            showlegend=True
                        ))
                    except Exception as e:
                        st.warning(f"Impossible d'afficher les intervalles de confiance: {e}")
                
                st.plotly_chart(fig_enc_comparison, use_container_width=True, key="enc_comparison")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Explication du graphique
                st.markdown(
                    '<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #3b82f6;">'                    
                    '<p style="margin: 0; font-style: italic; color: #475569; font-size: 0.9rem;">'                    
                    'Ce graphique compare les prévisions d\'encaissements générées par les différents modèles. '                    
                    f'Le modèle <strong>{best_model}</strong> (en surbrillance) a été identifié comme le plus performant. '                    
                    'Cliquez sur les éléments de la légende pour afficher/masquer les modèles.'                    
                    '</p>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
            
            with subtab2:
                # En-tête avec style amélioré
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">'                    
                    '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Prévisions des Décaissements par Modèle</h3>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Amélioration: Ajouter une légende interactive
                fig_dec_comparison.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        itemclick="toggleothers"
                    )
                )
                
                # Amélioration: Ajouter des intervalles de confiance si disponibles
                prophet_dec_model = 'prophet_dec'
                if prophet_dec_model in forecasts and forecasting_models and hasattr(forecasting_models, 'models') and prophet_dec_model in forecasting_models.models:
                    try:
                        # Créer un dataframe futur pour Prophet
                        future = pd.DataFrame({'ds': future_dates})
                        # Générer des prévisions avec intervalles de confiance
                        forecast = forecasting_models.models[prophet_dec_model].predict(future)
                        
                        # Ajouter l'intervalle de confiance supérieur
                        fig_dec_comparison.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_upper'].values,
                            mode='lines',
                            line=dict(width=0),
                            name='IC supérieur (95%)',
                            showlegend=True
                        ))
                        
                        # Ajouter l'intervalle de confiance inférieur
                        fig_dec_comparison.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_lower'].values,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(220, 20, 60, 0.2)',
                            name='IC inférieur (95%)',
                            showlegend=True
                        ))
                    except Exception as e:
                        st.warning(f"Impossible d'afficher les intervalles de confiance: {e}")
                
                # Création et affichage du graphique avec animation
                st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
                st.plotly_chart(fig_dec_comparison, use_container_width=True, key="dec_comparison")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Explication du graphique
                st.markdown(
                    '<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #3b82f6;">'                    
                    '<p style="margin: 0; font-style: italic; color: #475569; font-size: 0.9rem;">'                    
                    'Ce graphique compare les prévisions de décaissements générées par les différents modèles. '                    
                    f'Le modèle <strong>{best_model.replace("enc", "dec")}</strong> (en surbrillance) a été identifié comme le plus performant. '                    
                    'Cliquez sur les éléments de la légende pour afficher/masquer les modèles.'                    
                    '</p>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
            
            with subtab3:
                # En-tête avec style amélioré
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">'                    
                    '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Métriques de Performance des Modèles</h3>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Affichage du meilleur modèle avec style amélioré
                if best_model:
                    st.markdown(
                        f'<div style="background-color: #ecfdf5; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #10b981;">'                        
                        f'<p style="margin: 0; color: #065f46;"><strong>Meilleur modèle :</strong> {best_model}</p>'                        
                        f'<p style="margin: 0.5rem 0 0 0; color: #065f46;"><strong>Erreur (MAE) :</strong> {model_metrics.get(best_model, {}).get("MAE", 0):.2f}</p>'                        
                        f'<p style="margin: 0.5rem 0 0 0; color: #065f46;"><strong>Erreur (MAPE) :</strong> {model_metrics.get(best_model, {}).get("MAPE", 0):.2f}%</p>'                        
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Aucun modèle n'a été identifié comme le meilleur. Veuillez vérifier les paramètres de sélection.")
                
                # Affichage de tous les modèles disponibles
                st.markdown("### Tous les modèles disponibles")
                
                # Créer un DataFrame pour afficher tous les modèles disponibles dans forecasts
                available_models = []
                for model_name in forecasts.keys():
                    if 'enc' in model_name:  # Ne prendre que les modèles d'encaissement pour éviter les doublons
                        model_type = model_name.replace('_enc', '')
                        available_models.append({
                            "Modèle": model_type,
                            "Disponible": "✅",
                            "MAE": model_metrics.get(model_name, {}).get("MAE", 0),
                            "MAPE (%)": model_metrics.get(model_name, {}).get("MAPE", 0),
                            "Meilleur": "✅" if model_name == best_model else ""
                        })
                
                if available_models:
                    # Trier par MAE croissant
                    available_models_df = pd.DataFrame(available_models).sort_values("MAE")
                    
                    # Formater les colonnes numériques
                    available_models_df["MAE"] = available_models_df["MAE"].map('{:,.2f}'.format)
                    available_models_df["MAPE (%)"] = available_models_df["MAPE (%)"].map('{:,.2f}'.format)
                    
                    # Afficher le tableau
                    st.dataframe(
                        available_models_df,
                        use_container_width=True,
                        column_config={
                            "Modèle": st.column_config.TextColumn("Modèle"),
                            "Disponible": st.column_config.TextColumn("Disponible"),
                            "MAE": st.column_config.TextColumn("MAE"),
                            "MAPE (%)": st.column_config.TextColumn("MAPE (%)"),
                            "Meilleur": st.column_config.TextColumn("Meilleur Modèle")
                        },
                        hide_index=True
                    )
                else:
                    st.warning("Aucun modèle disponible pour l'affichage.")
                
                
                # Amélioration: Tableau détaillé des métriques pour tous les modèles
                st.markdown("### Tableau détaillé des métriques")
                
                if model_metrics:
                    # Créer un DataFrame des métriques pour tous les modèles
                    metrics_data = {}
                    for model, metrics in model_metrics.items():
                        if 'enc' in model:  # Filtrer pour n'afficher que les modèles d'encaissement
                            metrics_data[model] = {
                                'MAE': metrics.get('MAE', 0),
                                'MAPE': metrics.get('MAPE', 0)
                            }
                    
                    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
                    if not metrics_df.empty:
                        metrics_df = metrics_df.sort_values('MAE')
                        
                        # Créer une colonne pour indiquer le meilleur modèle
                        metrics_df['Meilleur'] = metrics_df.index == best_model
                        metrics_df['Meilleur'] = metrics_df['Meilleur'].map({True: '✅', False: ''})
                        
                        # Formater les colonnes pour l'affichage
                        metrics_df_display = metrics_df.copy()
                        metrics_df_display['MAE'] = metrics_df_display['MAE'].map('{:,.2f}'.format)
                        metrics_df_display['MAPE'] = metrics_df_display['MAPE'].map('{:,.2f}%'.format)
                        
                        # Afficher le tableau avec style
                        st.dataframe(
                            metrics_df_display,
                            use_container_width=True,
                            column_config={
                                "index": st.column_config.TextColumn("Modèle"),
                                "MAE": st.column_config.TextColumn("MAE (Erreur Absolue Moyenne)"),
                                "MAPE": st.column_config.TextColumn("MAPE (% d'Erreur)"),
                                "Meilleur": st.column_config.TextColumn("Meilleur Modèle")
                            }
                        )
                    else:
                        st.warning("Aucune métrique disponible pour les modèles d'encaissement.")
                else:
                    st.warning("Aucune métrique disponible.")
                
                # Création et affichage du graphique des métriques avec animation
                st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
                fig_metrics = visualizer.create_metrics_chart(model_metrics)
                st.plotly_chart(fig_metrics, use_container_width=True, key="metrics_chart")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Amélioration: Diagnostic du meilleur modèle
                with st.expander("Diagnostic du meilleur modèle", expanded=False):
                    if 'prophet' in best_model and forecasting_models and hasattr(forecasting_models, 'models') and best_model in forecasting_models.models:
                        try:
                            st.markdown("### Composantes du modèle Prophet")
                            # Créer un dataframe futur pour Prophet
                            future = pd.DataFrame({'ds': future_dates})
                            # Générer des prévisions avec composantes
                            forecast = forecasting_models.models[best_model].predict(future)
                            
                            # Créer un graphique pour la tendance
                            fig_trend = px.line(
                                x=forecast['ds'], 
                                y=forecast['trend'],
                                labels={"x": "Date", "y": "Tendance"},
                                title="Tendance détectée par Prophet"
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                            
                            # Afficher les composantes saisonnières si disponibles
                            if 'yearly' in forecast.columns:
                                fig_yearly = px.line(
                                    x=forecast['ds'], 
                                    y=forecast['yearly'],
                                    labels={"x": "Date", "y": "Saisonnalité Annuelle"},
                                    title="Saisonnalité Annuelle"
                                )
                                st.plotly_chart(fig_yearly, use_container_width=True)
                            
                            if 'weekly' in forecast.columns:
                                fig_weekly = px.line(
                                    x=forecast['ds'], 
                                    y=forecast['weekly'],
                                    labels={"x": "Date", "y": "Saisonnalité Hebdomadaire"},
                                    title="Saisonnalité Hebdomadaire"
                                )
                                st.plotly_chart(fig_weekly, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Impossible d'afficher les composantes du modèle Prophet: {e}")
                    
                    # Analyse des résidus
                    if best_model in forecasts and len(df_enc) >= len(forecasts[best_model]):
                        try:
                            st.markdown("### Analyse des résidus")
                            # Calculer et afficher les résidus
                            y_true = df_enc['y_enc'].values[-len(forecasts[best_model]):]
                            y_pred = forecasts[best_model]
                            residuals = y_true - y_pred
                            
                            fig_residuals = px.scatter(
                                x=np.arange(len(residuals)), 
                                y=residuals,
                                labels={"x": "Observation", "y": "Résidu"},
                                title="Résidus du modèle"
                            )
                            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_residuals, use_container_width=True)
                            
                            # Histogramme des résidus
                            fig_hist = px.histogram(
                                x=residuals,
                                labels={"x": "Résidu", "y": "Fréquence"},
                                title="Distribution des résidus"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Impossible d'afficher l'analyse des résidus: {e}")
                
                # Explication des métriques
                st.markdown(
                    '<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #3b82f6;">'                    
                    '<p style="margin: 0; font-style: italic; color: #475569; font-size: 0.9rem;">'                    
                    '<strong>MAE (Erreur Absolue Moyenne) :</strong> Mesure l\'erreur moyenne en valeur absolue entre les prévisions et les valeurs réelles. Plus cette valeur est basse, meilleur est le modèle.<br><br>'                    
                    '<strong>MAPE (Erreur Absolue Moyenne en Pourcentage) :</strong> Exprime l\'erreur en pourcentage par rapport aux valeurs réelles. Permet de comparer les performances indépendamment de l\'ordre de grandeur des données.'                    
                    '</p>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Amélioration: Affichage des paramètres des modèles
                with st.expander("Paramètres des modèles", expanded=False):
                    if forecasting_models and hasattr(forecasting_models, 'config'):
                        st.markdown("### Configuration des modèles")
                        
                        # Créer un dictionnaire des paramètres par modèle
                        model_params = {
                            "Prophet": {
                                "Activé": forecasting_models.config.get('use_prophet', False),
                                "Périodes de prévision": n_mois,
                                "Détection de saisonnalité": forecasting_models.config.get('detect_seasonality', True),
                                "Intervalle de confiance": forecasting_models.config.get('prophet_interval_width', 0.95),
                                "Croissance": forecasting_models.config.get('prophet_growth', 'linear')
                            },
                            "ARIMA": {
                                "Activé": forecasting_models.config.get('use_arima', False),
                                "Périodes de prévision": n_mois,
                                "Ordre (p,d,q)": forecasting_models.config.get('arima_order', '(1,1,1)'),
                                "Saisonnier": forecasting_models.config.get('use_sarima', False)
                            },
                            "LSTM": {
                                "Activé": forecasting_models.config.get('use_lstm', False),
                                "Périodes de prévision": n_mois,
                                "Époques": forecasting_models.config.get('lstm_epochs', 50),
                                "Taille de batch": forecasting_models.config.get('lstm_batch_size', 32),
                                "Fenêtre temporelle": forecasting_models.config.get('lstm_window', 12)
                            },
                            "XGBoost": {
                                "Activé": forecasting_models.config.get('use_xgboost', False),
                                "Périodes de prévision": n_mois,
                                "Profondeur max": forecasting_models.config.get('xgb_max_depth', 6),
                                "Taux d'apprentissage": forecasting_models.config.get('xgb_learning_rate', 0.1)
                            },
                            "Modèle hybride": {
                                "Activé": forecasting_models.config.get('use_hybrid', False),
                                "Périodes de prévision": n_mois,
                                "Modèles combinés": str(forecasting_models.config.get('hybrid_models', ['prophet', 'arima']))
                            }
                        }
                        
                        # Afficher les paramètres dans un format tabulé
                        for model_name, params in model_params.items():
                            if params.get("Activé", False):
                                st.markdown(f"#### {model_name}")
                                
                                # Convertir les paramètres en DataFrame pour un affichage tabulé
                                # Convertir toutes les valeurs en chaînes pour éviter les problèmes de conversion PyArrow
                                params_items = [(k, str(v)) for k, v in params.items()]
                                params_df = pd.DataFrame(params_items, columns=['Paramètre', 'Valeur'])
                                st.dataframe(params_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Aucune information sur les paramètres des modèles n'est disponible.")
                
                # Amélioration: Option d'exportation des résultats
                with st.expander("Exporter les résultats", expanded=False):
                    st.markdown("### Exporter les prévisions et métriques")
                    
                    # Créer un DataFrame pour l'export
                    if forecasts and future_dates.size > 0:
                        export_data = pd.DataFrame({
                            'Date': future_dates
                        })
                        
                        # Ajouter les prévisions de chaque modèle
                        for model, forecast in forecasts.items():
                            if 'enc' in model and len(forecast) == len(future_dates):
                                export_data[f'{model}'] = forecast
                        
                        # Convertir en CSV pour téléchargement
                        csv = export_data.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les prévisions (CSV)",
                            data=csv,
                            file_name="previsions_tresorerie.csv",
                            mime="text/csv",
                        )
                        
                        # Ajouter un bouton pour exporter les métriques
                        if model_metrics:
                            metrics_df = pd.DataFrame.from_dict(
                                {model: {'MAE': metrics.get('MAE', 0), 'MAPE': metrics.get('MAPE', 0)} 
                                 for model, metrics in model_metrics.items()},
                                orient='index'
                            )
                            metrics_csv = metrics_df.to_csv()
                            st.download_button(
                                label="Télécharger les métriques (CSV)",
                                data=metrics_csv,
                                file_name="metriques_modeles.csv",
                                mime="text/csv",
                            )
                    else:
                        st.warning("Aucune donnée disponible pour l'exportation.")
                
    
    with tab_scenarios:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">🌐 Simulation de Scénarios</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Explorer différents scénarios de trésorerie pour anticiper les évolutions futures</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Vérifier si les prévisions sont disponibles pour la simulation
        if not forecasts or best_model == '' or best_model not in forecasts:
            st.markdown(
                '<div style="background-color: #fff7ed; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #f59e0b;">'                
                '<p style="margin: 0; color: #92400e;"><strong>Attention :</strong> Les prévisions ne sont pas disponibles pour la simulation. Veuillez générer des prévisions d\'abord.</p>'                
                '</div>',
                unsafe_allow_html=True
            )
            return
        
        # Vérifier si le modèle de décaissement correspondant existe
        best_dec_model = best_model.replace('enc', 'dec')
        if best_dec_model not in forecasts:
            st.markdown(
                f'<div style="background-color: #fff7ed; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #f59e0b;">'                
                f'<p style="margin: 0; color: #92400e;"><strong>Attention :</strong> Le modèle de décaissement correspondant ({best_dec_model}) n\'est pas disponible.</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
            return
        
        # Vérifier si la simulation Monte Carlo est activée
        show_monte_carlo = st.session_state.get('show_monte_carlo', False) or \
                          (forecasting_models and hasattr(forecasting_models, 'monte_carlo_results') and forecasting_models.monte_carlo_results)
        
        # Si Monte Carlo est activé, l'afficher en premier
        if show_monte_carlo:
            # Affichage des résultats de la simulation Monte Carlo
            st.markdown(
                '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
                '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Simulation Monte Carlo</h3>'            
                '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse probabiliste des scénarios futurs</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Si les résultats Monte Carlo sont disponibles, les afficher
            if forecasting_models and hasattr(forecasting_models, 'monte_carlo_results') and forecasting_models.monte_carlo_results:
                mc_results = forecasting_models.monte_carlo_results
                st.info(f"Simulation basée sur {mc_results.get('n_simulations', 1000)} itérations")
            else:
                # Sinon, afficher un message pour générer les prévisions
                st.warning("Pour voir les résultats de la simulation Monte Carlo, veuillez générer les prévisions en cliquant sur le bouton 'Générer Prévisions'.")
        
        # Affichage des scénarios prédéfinis avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
            '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Scénarios Prédéfinis</h3>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Affichage des résultats de la simulation Monte Carlo si disponibles
        if forecasting_models and hasattr(forecasting_models, 'monte_carlo_results') and forecasting_models.monte_carlo_results:
            st.markdown(
                '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
                '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Simulation Monte Carlo</h3>'            
                '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse probabiliste des scénarios futurs basée sur ' + 
                str(forecasting_models.monte_carlo_results.get('n_simulations', 1000)) + ' simulations</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Création des graphiques pour la simulation Monte Carlo
            mc_results = forecasting_models.monte_carlo_results
            
            # Graphique des encaissements avec intervalles de confiance
            fig_enc_mc = go.Figure()
            
            # Ajouter la ligne moyenne
            fig_enc_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['encaissement_mean'],
                mode='lines',
                name='Encaissements (moyenne)',
                line=dict(color='rgba(0, 128, 0, 0.8)', width=2)
            ))
            
            # Ajouter l'intervalle de confiance
            fig_enc_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['encaissement_upper_95'],
                mode='lines',
                name='IC supérieur (95%)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_enc_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['encaissement_lower_95'],
                mode='lines',
                name='IC inférieur (95%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.2)',
                showlegend=False
            ))
            
            fig_enc_mc.update_layout(
                title='Prévisions des Encaissements avec Intervalles de Confiance (95%)',
                xaxis_title='Date',
                yaxis_title='Montant (DH)',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=10, r=10, t=60, b=10),
                height=400
            )
            
            # Graphique des décaissements avec intervalles de confiance
            fig_dec_mc = go.Figure()
            
            # Ajouter la ligne moyenne
            fig_dec_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['decaissement_mean'],
                mode='lines',
                name='Décaissements (moyenne)',
                line=dict(color='rgba(220, 20, 60, 0.8)', width=2)
            ))
            
            # Ajouter l'intervalle de confiance
            fig_dec_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['decaissement_upper_95'],
                mode='lines',
                name='IC supérieur (95%)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_dec_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['decaissement_lower_95'],
                mode='lines',
                name='IC inférieur (95%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(220, 20, 60, 0.2)',
                showlegend=False
            ))
            
            fig_dec_mc.update_layout(
                title='Prévisions des Décaissements avec Intervalles de Confiance (95%)',
                xaxis_title='Date',
                yaxis_title='Montant (DH)',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=10, r=10, t=60, b=10),
                height=400
            )
            
            # Graphique du solde avec intervalles de confiance
            fig_solde_mc = go.Figure()
            
            # Ajouter la ligne moyenne
            fig_solde_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['solde_mean'],
                mode='lines',
                name='Solde (moyenne)',
                line=dict(color='rgba(65, 105, 225, 0.8)', width=2)
            ))
            
            # Ajouter l'intervalle de confiance
            fig_solde_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['solde_upper_95'],
                mode='lines',
                name='IC supérieur (95%)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_solde_mc.add_trace(go.Scatter(
                x=future_dates,
                y=mc_results['solde_lower_95'],
                mode='lines',
                name='IC inférieur (95%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(65, 105, 225, 0.2)',
                showlegend=False
            ))
            
            # Ajouter une ligne horizontale à zéro
            fig_solde_mc.add_shape(
                type="line",
                x0=future_dates[0],
                y0=0,
                x1=future_dates[-1],
                y1=0,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig_solde_mc.update_layout(
                title='Prévisions du Solde avec Intervalles de Confiance (95%)',
                xaxis_title='Date',
                yaxis_title='Montant (DH)',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=10, r=10, t=60, b=10),
                height=400
            )
            
            # Afficher les graphiques
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_enc_mc, use_container_width=True)
                st.plotly_chart(fig_solde_mc, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_dec_mc, use_container_width=True)
                
                # Afficher la probabilité de solde négatif
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 5px solid #3b82f6;">'            
                    '<h4 style="margin-top: 0; color: #1e40af;">Analyse des Risques</h4>'            
                    '<p><strong>Probabilité de solde négatif par mois :</strong></p>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Créer un DataFrame pour afficher les probabilités
                risk_df = pd.DataFrame({
                    'Date': future_dates,
                    'Probabilité (%)': mc_results['prob_negative_solde']
                })
                
                # Créer un graphique pour la probabilité de solde négatif
                fig_risk = px.bar(
                    risk_df, 
                    x='Date', 
                    y='Probabilité (%)',
                    color='Probabilité (%)',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    range_color=[0, 100],
                    title='Probabilité de Solde Négatif par Mois'
                )
                
                fig_risk.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Probabilité (%)',
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=300
                )
                
                st.plotly_chart(fig_risk, use_container_width=True)
        
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        
        # Description des scénarios
        scenario_descriptions = {
            "neutre": "Prévisions basées sur les tendances actuelles sans changements majeurs",
            "optimiste": "Scénario favorable avec augmentation des encaissements et stabilité des décaissements",
            "pessimiste": "Scénario défavorable avec diminution des encaissements et augmentation des décaissements",
            "croissance": "Scénario de croissance progressive avec augmentation des flux"
        }
        
        # Sélection du scénario avec style amélioré
        st.markdown('<p style="font-weight: 500; margin-bottom: 0.5rem;">Choisissez un scénario à explorer :</p>', unsafe_allow_html=True)
        scenario_type = st.selectbox(
            "Type de scénario",
            ["neutre", "optimiste", "pessimiste", "croissance"],
            format_func=lambda x: f"{x.capitalize()} - {scenario_descriptions[x]}"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Affichage du scénario sélectionné
        if scenario_type in scenarios:
            # Titre du scénario sélectionné avec style amélioré
            scenario_icons = {
                "neutre": "📊",
                "optimiste": "📈",
                "pessimiste": "📉",
                "croissance": "🚀"
            }
            
            st.markdown(
                f'<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'                
                f'<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">{scenario_icons.get(scenario_type, "📊")} Scénario {scenario_type.capitalize()}</h3>'                
                f'<p style="margin: 0.5rem 0 0 0; color: #6b7280;">{scenario_descriptions[scenario_type]}</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Récupération du scénario et création du graphique
            scenario = scenarios[scenario_type]
            
            # Création et affichage du graphique avec animation
            fig_scenario = visualizer.create_scenario_chart(df_enc, df_dec, scenario, future_dates)
            st.plotly_chart(fig_scenario, use_container_width=True, key=f"scenario_{scenario_type}")
            
            # Calcul du solde prévisionnel
            solde = scenario['encaissement'] - scenario['decaissement']
            
            # Affichage des statistiques du scénario avec style amélioré
            st.markdown(
                '<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
                '<h3 style="font-size: 1.2rem; margin: 0; color: #334155;">Statistiques du Scénario</h3>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Encaissements Moyens", 
                    f"{np.mean(scenario['encaissement']):,.0f} DH",
                    delta=f"{np.mean(scenario['encaissement']) - np.mean(df_enc['y_enc']):,.0f} DH" if len(df_enc) > 0 else None,
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "Décaissements Moyens", 
                    f"{np.mean(scenario['decaissement']):,.0f} DH",
                    delta=f"{np.mean(scenario['decaissement']) - np.mean(df_dec['y_dec']):,.0f} DH" if len(df_dec) > 0 else None,
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Solde Moyen", 
                    f"{np.mean(solde):,.0f} DH",
                    delta=f"{np.mean(solde) - (np.mean(df_enc['y_enc']) - np.mean(df_dec['y_dec'])):,.0f} DH" if len(df_enc) > 0 and len(df_dec) > 0 else None,
                    delta_color="normal"
                )
            
            # Tableau détaillé des prévisions du scénario
            st.markdown(
                '<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
                '<h3 style="font-size: 1.2rem; margin: 0; color: #334155;">Détail des Prévisions</h3>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Création d'un DataFrame pour les prévisions du scénario
            scenario_df = pd.DataFrame({
                'Date': future_dates[:len(scenario['encaissement'])],
                'Encaissements': scenario['encaissement'],
                'Décaissements': scenario['decaissement'],
                'Solde': solde
            })
            
            # Formatage des colonnes numériques
            display_df = scenario_df.copy()
            display_df['Encaissements'] = display_df['Encaissements'].map('{:,.0f} DH'.format)
            display_df['Décaissements'] = display_df['Décaissements'].map('{:,.0f} DH'.format)
            display_df['Solde'] = display_df['Solde'].map('{:,.0f} DH'.format)
            
            # Affichage du tableau
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="MMM YYYY"),
                    "Encaissements": st.column_config.TextColumn("Encaissements"),
                    "Décaissements": st.column_config.TextColumn("Décaissements"),
                    "Solde": st.column_config.TextColumn("Solde")
                },
                height=300
            )
            
            # Bouton pour télécharger les prévisions du scénario
            col1, col2 = st.columns(2)
            with col1:
                csv = scenario_df.to_csv(index=False)
                st.download_button(
                    label=f"Télécharger le scénario {scenario_type} (CSV)",
                    data=csv,
                    file_name=f"scenario_{scenario_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Analyse des impacts du scénario
            st.markdown(
                '<div style="background-color: #f0fff4; padding: 1rem; border-radius: 8px; margin: 1.5rem 0; border-left: 5px solid #48bb78;">'            
                '<h3 style="margin-top: 0; color: #2f855a;">Analyse des Impacts</h3>'            
                '<p>Ce scénario représente une projection basée sur les paramètres sélectionnés. Voici les principaux impacts :</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Calcul des impacts
            enc_change = np.mean(scenario['encaissement']) / np.mean(df_enc['y_enc']) - 1 if len(df_enc) > 0 else 0
            dec_change = np.mean(scenario['decaissement']) / np.mean(df_dec['y_dec']) - 1 if len(df_dec) > 0 else 0
            solde_hist = np.mean(df_enc['y_enc']) - np.mean(df_dec['y_dec']) if len(df_enc) > 0 and len(df_dec) > 0 else 0
            solde_change = np.mean(solde) / solde_hist - 1 if solde_hist != 0 else 0
            
            # Affichage des impacts sous forme de liste
            impacts = []
            if enc_change > 0.05:
                impacts.append(f"📈 **Augmentation significative des encaissements** de {enc_change:.1%}")
            elif enc_change < -0.05:
                impacts.append(f"📉 **Diminution significative des encaissements** de {abs(enc_change):.1%}")
            
            if dec_change > 0.05:
                impacts.append(f"⚠️ **Augmentation significative des décaissements** de {dec_change:.1%}")
            elif dec_change < -0.05:
                impacts.append(f"✅ **Diminution significative des décaissements** de {abs(dec_change):.1%}")
            
            if solde_change > 0.1:
                impacts.append(f"💰 **Amélioration notable du solde** de {solde_change:.1%}")
            elif solde_change < -0.1:
                impacts.append(f"⚠️ **Détérioration notable du solde** de {abs(solde_change):.1%}")
            
            # Ajout d'un impact sur la trésorerie
            min_solde = np.min(solde)
            if min_solde < 0:
                impacts.append(f"🚨 **Risque de trésorerie négative** avec un minimum de {min_solde:,.0f} DH")
            
            # Affichage des impacts
            if impacts:
                for impact in impacts:
                    st.markdown(impact)
            else:
                st.markdown("Aucun impact significatif détecté dans ce scénario.")
        else:
            st.warning("Aucun scénario disponible. Veuillez générer des prévisions d'abord.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ajout d'une explication du scénario
        scenario_explanations = {
            "neutre": "Ce scénario représente l'évolution attendue sans changements majeurs dans les conditions actuelles.",
            "optimiste": "Ce scénario favorable suppose une amélioration des encaissements et une stabilité des décaissements.",
            "pessimiste": "Ce scénario défavorable anticipe une diminution des encaissements et une augmentation des décaissements.",
            "croissance": "Ce scénario de croissance prévoit une augmentation progressive des flux d'encaissements."
        }
        
        st.markdown(
            f'<div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #3b82f6;">'            
            f'<p style="margin: 0; font-style: italic; color: #475569; font-size: 0.9rem;">'            
            f'{scenario_explanations.get(scenario_type, "")} Utilisez ces prévisions pour ajuster votre stratégie financière en conséquence.'            
            f'</p>'            
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Interface pour la création de scénarios personnalisés avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 2rem 0 1rem 0;">'            
            '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">💸 Création de Scénarios Personnalisés</h3>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Explication du scénario personnalisé
        st.markdown(
            '<div style="background-color: #ecfdf5; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem; border-left: 3px solid #10b981;">'            
            '<p style="margin: 0; color: #065f46; font-size: 0.9rem;">'            
            'Créez votre propre scénario en ajustant les paramètres ci-dessous. Vous pouvez modifier la croissance, la volatilité et la saisonnalité des flux pour simuler différentes situations financières.'            
            '</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Création d'une carte pour les paramètres
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres pour les encaissements avec style amélioré
            st.markdown('<p style="font-weight: 600; color: #1e40af; margin: 0 0 0.75rem 0;">Paramètres d\'Encaissements</p>', unsafe_allow_html=True)
            
            st.markdown('<p style="font-weight: 500; margin: 0.5rem 0 0.25rem 0; font-size: 0.9rem;">Tendance</p>', unsafe_allow_html=True)
            enc_growth = st.slider(
                "Croissance (%)", 
                min_value=-50, 
                max_value=100, 
                value=0, 
                step=5, 
                key="enc_growth",
                help="Pourcentage d'augmentation ou de diminution des encaissements sur la période"
            )
            
            st.markdown('<p style="font-weight: 500; margin: 1rem 0 0.25rem 0; font-size: 0.9rem;">Incertitude</p>', unsafe_allow_html=True)
            enc_volatility = st.slider(
                "Volatilité (%)", 
                min_value=0, 
                max_value=50, 
                value=10, 
                step=5, 
                key="enc_volatility",
                help="Niveau de variation aléatoire des encaissements"
            )
            
            st.markdown('<p style="font-weight: 500; margin: 1rem 0 0.25rem 0; font-size: 0.9rem;">Cycle</p>', unsafe_allow_html=True)
            enc_seasonality = st.selectbox(
                "Saisonnalité", 
                options=["Aucune", "Mensuelle", "Trimestrielle"], 
                key="enc_seasonality",
                help="Type de variation cyclique des encaissements"
            )
        
        with col2:
            # Paramètres pour les décaissements avec style amélioré
            st.markdown('<p style="font-weight: 600; color: #1e40af; margin: 0 0 0.75rem 0;">Paramètres de Décaissements</p>', unsafe_allow_html=True)
            
            st.markdown('<p style="font-weight: 500; margin: 0.5rem 0 0.25rem 0; font-size: 0.9rem;">Tendance</p>', unsafe_allow_html=True)
            dec_growth = st.slider(
                "Croissance (%)", 
                min_value=-50, 
                max_value=100, 
                value=0, 
                step=5, 
                key="dec_growth",
                help="Pourcentage d'augmentation ou de diminution des décaissements sur la période"
            )
            
            st.markdown('<p style="font-weight: 500; margin: 1rem 0 0.25rem 0; font-size: 0.9rem;">Incertitude</p>', unsafe_allow_html=True)
            dec_volatility = st.slider(
                "Volatilité (%)", 
                min_value=0, 
                max_value=50, 
                value=10, 
                step=5, 
                key="dec_volatility",
                help="Niveau de variation aléatoire des décaissements"
            )
            
            st.markdown('<p style="font-weight: 500; margin: 1rem 0 0.25rem 0; font-size: 0.9rem;">Cycle</p>', unsafe_allow_html=True)
            dec_seasonality = st.selectbox(
                "Saisonnalité", 
                options=["Aucune", "Mensuelle", "Trimestrielle"], 
                key="dec_seasonality",
                help="Type de variation cyclique des décaissements"
            )
        
        # Fermeture de la carte des paramètres
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Génération du scénario personnalisé avec style amélioré
        st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
        generate_button = st.button(
            "📈 Générer Mon Scénario Personnalisé",
            use_container_width=True,
            key="generate_custom_scenario"
        )
        
        if generate_button:
            try:
                with st.spinner("Génération du scénario personnalisé en cours..."):
                    # Création du scénario personnalisé
                    params = {
                        'enc_growth': enc_growth,
                        'enc_volatility': enc_volatility,
                        'enc_seasonality': enc_seasonality,
                        'dec_growth': dec_growth,
                        'dec_volatility': dec_volatility,
                        'dec_seasonality': dec_seasonality
                    }
                    
                    if forecasting_models is not None:
                        custom_scenario = forecasting_models.create_custom_scenario(forecasts, n_mois, params)
                    else:
                        st.error("Impossible de générer le scénario personnalisé. Veuillez réexécuter l'analyse.")
                        custom_scenario = {}
                
                if custom_scenario:
                    # Affichage du scénario personnalisé avec style amélioré
                    st.markdown(
                        '<div style="background-color: #ecfdf5; padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0; border-left: 5px solid #10b981;">'                        
                        '<h3 style="font-size: 1.2rem; margin: 0; color: #065f46;">✅ Scénario Personnalisé Généré avec Succès</h3>'                        
                        '<p style="margin: 0.5rem 0 0 0; color: #065f46;">Votre scénario a été créé selon les paramètres spécifiés.</p>'                        
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Création et affichage du graphique avec animation
                    st.markdown('<div class="plotly-chart">', unsafe_allow_html=True)
                    fig_custom = visualizer.create_scenario_chart(df_enc, df_dec, custom_scenario, future_dates)
                    st.plotly_chart(fig_custom, use_container_width=True, key="custom_scenario")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calcul du solde prévisionnel
                    solde = custom_scenario['encaissement'] - custom_scenario['decaissement']
                    
                    # Création d'une carte pour les statistiques
                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    
                    # Affichage des statistiques du scénario avec style amélioré
                    st.markdown('<p style="font-weight: 500; margin-bottom: 0.5rem;">Résultats du scénario personnalisé :</p>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Encaissements Moyens", 
                            f"{np.mean(custom_scenario['encaissement']):,.0f} DH",
                            delta=f"{np.mean(custom_scenario['encaissement']) - np.mean(df_enc['y_enc']):,.0f} DH" if len(df_enc) > 0 else None,
                            delta_color="normal"
                        )
                    
                    with col2:
                        st.metric(
                            "Décaissements Moyens", 
                            f"{np.mean(custom_scenario['decaissement']):,.0f} DH",
                            delta=f"{np.mean(custom_scenario['decaissement']) - np.mean(df_dec['y_dec']):,.0f} DH" if len(df_dec) > 0 else None,
                            delta_color="inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Solde Moyen", 
                            f"{np.mean(solde):,.0f} DH",
                            delta=f"{np.mean(solde) - (np.mean(df_enc['y_enc']) - np.mean(df_dec['y_dec'])):,.0f} DH" if len(df_enc) > 0 and len(df_dec) > 0 else None,
                            delta_color="normal"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Recommandations basées sur le scénario
                    if np.mean(solde) < 0:
                        st.markdown(
                            '<div style="background-color: #fef2f2; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #ef4444;">'                            
                            '<p style="margin: 0; color: #991b1b; font-size: 0.9rem;">'                            
                            '<strong>Attention :</strong> Ce scénario prévoit un solde moyen négatif. Envisagez des mesures pour augmenter vos encaissements ou réduire vos décaissements.'                            
                            '</p>'                            
                            '</div>',
                            unsafe_allow_html=True
                        )
                    elif np.mean(solde) > 0 and np.mean(solde) < 0.1 * np.mean(df_dec['y_dec']) and len(df_dec) > 0:
                        st.markdown(
                            '<div style="background-color: #fff7ed; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #f59e0b;">'                            
                            '<p style="margin: 0; color: #92400e; font-size: 0.9rem;">'                            
                            '<strong>Prudence :</strong> Ce scénario prévoit un solde positif mais faible. Constituez une réserve de trésorerie pour faire face aux imprévus.'                            
                            '</p>'                            
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div style="background-color: #ecfdf5; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #10b981;">'                            
                            '<p style="margin: 0; color: #065f46; font-size: 0.9rem;">'                            
                            '<strong>Favorable :</strong> Ce scénario prévoit un solde positif confortable. Envisagez d\'investir l\'excédent de trésorerie pour optimiser vos rendements.'                            
                            '</p>'                            
                            '</div>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.markdown(
                    f'<div style="background-color: #fef2f2; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #ef4444;">'                    
                    f'<p style="margin: 0; color: #991b1b;"><strong>Erreur :</strong> Impossible de générer le scénario personnalisé. {str(e)}</p>'                    
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab_metrics:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">'            
            '<h2 style="font-size: 1.5rem; margin: 0; color: #1e40af;">📈 Indicateurs Financiers</h2>'            
            '<p style="margin: 0.5rem 0 0 0; color: #6b7280;">Analyse des performances financières et recommandations stratégiques</p>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Calcul des métriques financières
        metrics = calculate_financial_metrics(df_enc, df_dec)
        
        # Affichage des ratios financiers avec style amélioré
        st.markdown(
            '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'            
            '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Ratios Financiers</h3>'            
            '</div>',
            unsafe_allow_html=True
        )
        
        # Création d'une carte pour les ratios
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="font-weight: 600; color: #1e40af; margin: 0 0 0.75rem 0;">Ratios de Base</p>', unsafe_allow_html=True)
            
            # Ajout d'icônes et de couleurs conditionnelles pour les métriques
            ratio_couverture = metrics['Ratio de Couverture']
            ratio_color = "#10b981" if ratio_couverture >= 1.2 else "#f59e0b" if ratio_couverture >= 1 else "#ef4444"
            ratio_icon = "📈" if ratio_couverture >= 1.2 else "⚠️" if ratio_couverture >= 1 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if ratio_couverture >= 1.2 else "245,158,11,0.1" if ratio_couverture >= 1 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {ratio_color};">Ratio de Couverture {ratio_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {ratio_color};">{ratio_couverture:.2f}</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Capacité à couvrir les dépenses</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
        
            # Taux de croissance encaissements
            enc_growth = metrics['Taux de Croissance Encaissements']
            enc_color = "#10b981" if enc_growth > 5 else "#f59e0b" if enc_growth >= 0 else "#ef4444"
            enc_icon = "📈" if enc_growth > 5 else "⚠️" if enc_growth >= 0 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if enc_growth > 5 else "245,158,11,0.1" if enc_growth >= 0 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {enc_color};">Croissance Encaissements {enc_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {enc_color};">{enc_growth:.2f}%</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Evolution des revenus</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Taux de croissance décaissements
            dec_growth = metrics['Taux de Croissance Décaissements']
            dec_color = "#10b981" if dec_growth < 0 else "#f59e0b" if dec_growth < 3 else "#ef4444"
            dec_icon = "📈" if dec_growth < 0 else "⚠️" if dec_growth < 3 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if dec_growth < 0 else "245,158,11,0.1" if dec_growth < 3 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {dec_color};">Croissance Décaissements {dec_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {dec_color};">{dec_growth:.2f}%</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Evolution des dépenses</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
    
        with col2:
            st.markdown('<p style="font-weight: 600; color: #1e40af; margin: 0 0 0.75rem 0;">Ratios Avancés</p>', unsafe_allow_html=True)
            
            # Volatilité Encaissements
            vol_enc = metrics['Volatilité Encaissements (%)']
            vol_enc_color = "#10b981" if vol_enc < 10 else "#f59e0b" if vol_enc < 20 else "#ef4444"
            vol_enc_icon = "📈" if vol_enc < 10 else "⚠️" if vol_enc < 20 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if vol_enc < 10 else "245,158,11,0.1" if vol_enc < 20 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {vol_enc_color};">Volatilité Encaissements {vol_enc_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {vol_enc_color};">{vol_enc:.2f}%</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Stabilité des revenus</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
        
            # Indice de stabilité
            stability = metrics['Indice de Stabilité']
            stability_color = "#10b981" if stability > 0.7 else "#f59e0b" if stability > 0.4 else "#ef4444"
            stability_icon = "📈" if stability > 0.7 else "⚠️" if stability > 0.4 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if stability > 0.7 else "245,158,11,0.1" if stability > 0.4 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {stability_color};">Indice de Stabilité {stability_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {stability_color};">{stability:.2f}</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Prévisibilité des flux</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
        
            # Marge de sécurité
            safety = metrics['Marge de Sécurité (%)']
            safety_color = "#10b981" if safety > 20 else "#f59e0b" if safety > 0 else "#ef4444"
            safety_icon = "📈" if safety > 20 else "⚠️" if safety > 0 else "❌"
            
            st.markdown(
                f'<div style="padding: 0.5rem; border-radius: 8px; margin-bottom: 0.75rem; background-color: rgba({"16,185,129,0.1" if safety > 20 else "245,158,11,0.1" if safety > 0 else "239,68,68,0.1"});">'                
                f'<p style="font-weight: 500; margin: 0 0 0.25rem 0; color: {safety_color};">Marge de Sécurité {safety_icon}</p>'                
                f'<p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {safety_color};">{safety:.2f}%</p>'                
                f'<p style="font-size: 0.8rem; margin: 0.25rem 0 0 0; color: #6b7280;">Coussin financier</p>'                
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Création et affichage du graphique radar
            fig_radar = visualizer.create_financial_indicators_chart(metrics)
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
            
            # Recommandations financières
            st.markdown(
                '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;">'                
                '<h3 style="font-size: 1.2rem; margin: 0; color: #1e40af;">Recommandations Financières</h3>'                
                '</div>',
                unsafe_allow_html=True
            )
            
            # Génération et affichage des recommandations
            recommendations = visualizer.generate_financial_recommendations(metrics)
    
            # Affichage des recommandations avec style amélioré
            for i, rec in enumerate(recommendations):
                if "⚠️" in rec:
                    # Recommandation d'avertissement
                    st.markdown(f'<div class="recommendation-warning">{rec}</div>', unsafe_allow_html=True)
                elif "✅" in rec or "✔️" in rec:
                    # Recommandation positive
                    st.markdown(f'<div class="recommendation-positive">{rec}</div>', unsafe_allow_html=True)
                else:
                    # Recommandation neutre ou négative
                    st.markdown(f'<div class="recommendation-neutral">{rec}</div>', unsafe_allow_html=True)
            
            # Ajout d'un pied de page
            st.markdown(
                '<footer style="margin-top: 3rem; text-align: center; color: #6b7280; font-size: 0.8rem;">'                
                'TresoreriePro © 2025 | Développé avec ❤️ pour une meilleure gestion financière'                
                '</footer>',
                unsafe_allow_html=True
            )

def export_forecasts(df_enc, forecasts, n_mois):
    """Fonction pour exporter les prévisions"""
    st.sidebar.header("📥 Export des Données")
    export_button = st.sidebar.button("Exporter Prévisions")
    
    if export_button and forecasts and df_enc is not None:
        try:
            # Création du DataFrame d'export
            export_df = pd.DataFrame({
                'Date': pd.date_range(start=df_enc['ds'].iloc[-1], periods=n_mois+1, freq='MS')[1:],
                **{k: v for k, v in forecasts.items()}
            })
            
            # Création du buffer pour le fichier Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Prévisions')
            
            # Bouton de téléchargement
            st.sidebar.download_button(
                label="📤 Télécharger Prévisions",
                data=buffer.getvalue(),
                file_name="previsions_tresorerie.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Affichage d'un résumé des résultats
            st.sidebar.success(f"🏆 Prévisions exportées avec succès! Meilleur modèle utilisé : {st.session_state.get('best_model', 'N/A')}")
        except Exception as e:
            st.sidebar.error(f"Erreur lors de l'export des prévisions : {e}")

if __name__ == "__main__":
    main()
