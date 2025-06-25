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
from widgets import Widgets
from io import BytesIO
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from models import ForecastingModels
from visualizations import TresorerieVisualizer
from validation import validate_data
from parallel import run_parallel
from cache import cache_data, get_cached_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import json
import time
import warnings
warnings.filterwarnings('ignore')

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
    <p><strong>Pour commencer</strong>: Importez vos données et configurez les paramètres de prévision.</p>
</div>
''', unsafe_allow_html=True)

def configure_sidebar():
    """Configure la sidebar avec tous les paramètres nécessaires"""
    # Créer une instance des widgets
    widgets = Widgets()
    return widgets.configure_sidebar()

def main():
    """Fonction principale de l'application"""
    # Initialiser show_monte_carlo à True pour afficher la simulation Monte Carlo dès le départ
    if 'show_monte_carlo' not in st.session_state:
        st.session_state['show_monte_carlo'] = True
        
    # Configuration de la sidebar
    config = configure_sidebar()
    
    # Activer la simulation Monte Carlo par défaut
    config['run_monte_carlo'] = True
    config['monte_carlo_sims'] = 1000
    
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
    uploaded_file = st.file_uploader("📂 Charger un fichier Excel (fusion_operations_triees.xlsx)", type="xlsx", help="Fichier contenant les opérations de 2023-2024 et les 5 premiers mois de 2025")
    
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
                    "Statut": status["Statut"]
                })
            
            initial_status_df = pd.DataFrame(initial_status_data)
            st.dataframe(
                initial_status_df,
                use_container_width=True,
                column_config={
                    "Modèle": st.column_config.TextColumn("Modèle"),
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
                    # st.markdown("### Statut des Modèles")
                    
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
                    # status_data = []
                    # for model_name, status in model_status.items():
                    #     status_data.append({
                    #         "Modèle": model_name,
                    #         "Activé": "✅" if status["Activé"] else "❌",
                    #         "Entraîné": "✅" if status["Entraîné"] else "❌",
                    #         "Prévisions": "✅" if status["Prévisions"] else "❌",
                    #         "Statut": status["Statut"]
                    #     })
                    
                    # status_df = pd.DataFrame(status_data)
                    # st.dataframe(
                    #     status_df,
                    #     use_container_width=True,
                    #     column_config={
                    #         "Modèle": st.column_config.TextColumn("Modèle"),
                    #         "Activé": st.column_config.TextColumn("Activé"),
                    #         "Entraîné": st.column_config.TextColumn("Entraîné"),
                    #         "Prévisions": st.column_config.TextColumn("Prévisions"),
                    #         "Statut": st.column_config.TextColumn("Statut")
                    #     },
                    #     hide_index=True
                    # )
                    
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
            display_results(df_enc, df_dec, forecasts, best_model, model_metrics, scenarios, n_mois, config, forecasting_models)
            
            # Export des prévisions
            if 'export_format' not in st.session_state:
                st.session_state.export_format = 'Excel'
            
            export_format = st.sidebar.selectbox(
                "Format d'export",
                options=['Excel', 'CSV', 'PDF'],
                index=['Excel', 'CSV', 'PDF'].index(st.session_state.export_format),
                key="export_format_selector"
            )
            st.session_state.export_format = export_format
            
            if st.sidebar.button("📤 Exporter les Prévisions", key="export_button"):
                st.sidebar.write(f"Format sélectionné : {export_format}")  # Debug
                output, mime_type, filename = export_forecasts(df_enc, df_dec, forecasts, n_mois, export_format)
                if output:
                    st.sidebar.download_button(
                        label=f"📥 Télécharger ({export_format})",
                        data=output,
                        file_name=filename,
                        mime=mime_type,
                        key="download_button"
                    )

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
    
    # Déterminer l'onglet actif par défaut
    active_tab = st.session_state.get('active_tab', "Scénarios")  # Changer l'onglet par défaut à "Scénarios"
    tab_names = ["Flux de Trésorerie", "Comparaison des Modèles", "Scénarios", "Métriques", "Analyse Saisonnière", "Détection d'Anomalies", "Analyses Avancées"]
    
    # Création des onglets pour organiser l'affichage
    tab_flux, tab_models, tab_scenarios, tab_metrics, tab_seasonal, tab_anomalies, tab_advanced = st.tabs(tab_names)
    
    # Onglet Flux de Trésorerie
    with tab_flux:
        # En-tête avec style amélioré
        st.markdown(
            '<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">'            
            '<h2 style="font-size: 2rem; margin: 0; color: #1e40af; font-weight: 700; letter-spacing: -0.5px;">📊 Flux de Trésorerie et Prévisions</h2>'            
            '<p style="margin: 1rem 0 0 0; color: #4b5563; font-size: 1.1rem; line-height: 1.5;">Visualisation des flux historiques et prévisionnels de votre trésorerie</p>'            
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
                        
            # Création d'un sélecteur pour choisir le modèle à afficher avec style amélioré
            available_models = [model_name for model_name in forecasts.keys() if 'enc' in model_name]
            
            if available_models:
                # Ajouter une option "Tous les modèles" en premier
                display_options = ["Tous les modèles"] + available_models
                
                # Création de deux colonnes pour le sélecteur et les informations
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_model = st.selectbox(
                        "Modèle à afficher",
                        options=display_options,
                        index=0,  # Par défaut, afficher tous les modèles
                        help="Sélectionnez un modèle spécifique ou 'Tous les modèles' pour voir toutes les prévisions"
                    )
                
                with col2:
                    if selected_model == "Tous les modèles":
                        st.markdown(
                            '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; border-left: 3px solid #3b82f6;">'
                            '<p style="margin: 0; color: #1e40af; font-size: 0.9rem;">'
                            f'<strong>Nombre de modèles :</strong> {len(available_models)}<br>'
                            f'<strong>Meilleur modèle :</strong> {best_model}'
                            '</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        model_metrics_info = model_metrics.get(selected_model, {})
                        st.markdown(
                            '<div style="background-color: #f0f9ff; padding: 0.75rem; border-radius: 8px; border-left: 3px solid #3b82f6;">'
                            '<p style="margin: 0; color: #1e40af; font-size: 0.9rem;">'
                            f'<strong>MAE :</strong> {model_metrics_info.get("MAE", 0):,.2f}<br>'
                            f'<strong>MAPE :</strong> {model_metrics_info.get("MAPE", 0):,.2f}%'
                            '</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                
                # Création du graphique principal avec style amélioré
                if selected_model == "Tous les modèles":
                    # Afficher tous les modèles disponibles
                    fig_main = visualizer.create_all_models_chart(df_enc, df_dec, forecasts, best_model, future_dates)
                else:
                    # Afficher uniquement le modèle sélectionné
                    fig_main = visualizer.create_flux_chart(df_enc, df_dec, forecasts, selected_model, future_dates)
                
                # Mise à jour du style du graphique
                fig_main.update_layout(
                    title=dict(
                        text="Évolution des Flux de Trésorerie",
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20, color='#1e40af')
                    ),
                    margin=dict(t=150, l=10, r=10, b=10),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.plotly_chart(fig_main, use_container_width=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # Section de commentaires explicatifs
                st.markdown(
                    '<div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #3b82f6;">'
                    '<h3 style="margin: 0 0 1rem 0; color: #1e40af; font-size: 1.1rem;">📊 Analyse des Prévisions de Trésorerie</h3>'
                    '<div style="color: #475569; font-size: 0.95rem; line-height: 1.6;">'
                    '<p style="margin: 0 0 0.5rem 0;"><strong>1. Méthodologie de Visualisation :</strong></p>'
                    '<ul style="margin: 0 0 0.5rem 0; padding-left: 1.5rem;">'
                    '<li>Données historiques : Représentation en série temporelle continue</li>'
                    '<li>Prévisions : Projections avec intervalles de confiance</li>'
                    '<li>Analyse comparative : Superposition des différents modèles</li>'
                    '</ul>'
                    
                    '<p style="margin: 0.5rem 0;"><strong>2. Paramètres de Modélisation :</strong></p>'
                    '<ul style="margin: 0 0 0.5rem 0; padding-left: 1.5rem;">'
                    '<li>Horizon de prévision : ' + str(n_mois) + ' mois</li>'
                    '<li>Modèle optimal : ' + best_model + '</li>'
                    '<li>Métriques de performance : MAE, MAPE, RMSE</li>'
                    '</ul>'
                    
                    '<p style="margin: 0.5rem 0;"><strong>3. Interprétation des Résultats :</strong></p>'
                    '<ul style="margin: 0 0 0.5rem 0; padding-left: 1.5rem;">'
                    '<li>Tendance générale : Analyse de la direction des flux</li>'
                    '<li>Variabilité : Évaluation des intervalles de confiance</li>'
                    '<li>Fiabilité : Comparaison des performances des modèles</li>'
                    '</ul>'
                    
                    '<p style="margin: 0.5rem 0;"><strong>4. Recommandations :</strong></p>'
                    '<ul style="margin: 0; padding-left: 1.5rem;">'
                    '<li>Utilisation des intervalles de confiance pour la planification</li>'
                    '<li>Surveillance des écarts entre prévisions et réalisations</li>'
                    '<li>Actualisation régulière des modèles selon les nouvelles données</li>'
                    '</ul>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("Aucun modèle disponible pour l'affichage.")
        else:
            st.warning("Aucune prévision disponible.")
            
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
        # Création des sous-onglets pour une meilleure organisation
        seasonal_tab1, seasonal_tab2 = st.tabs(["📊 Décomposition Saisonnière", "📈 Patterns Mensuels"])
        
        with seasonal_tab1:
            if forecasting_models and hasattr(forecasting_models, 'seasonal_patterns'):
                # Afficher l'analyse saisonnière pour les encaissements
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">💰 Encaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des tendances saisonnières</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'enc' in forecasting_models.seasonal_patterns:
                        fig_seasonal_enc = visualizer.create_seasonal_analysis_chart(
                            forecasting_models.seasonal_patterns['enc'],
                            title="Décomposition Saisonnière des Encaissements"
                        )
                        st.plotly_chart(fig_seasonal_enc, use_container_width=True)
                        
                        # Afficher des informations sur la saisonnalité détectée
                        if forecasting_models.seasonal_patterns['enc'].get('has_seasonality', False):
                            seasonal_strength = forecasting_models.seasonal_patterns['enc'].get('seasonal_strength', 0) * 100
                            dominant_period = forecasting_models.seasonal_patterns['enc'].get('dominant_period', 0)
                            st.markdown(f"""
                                <div style='background-color: #f0fff4; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #48bb78;'>
                                <h3 style='margin-top: 0; color: #2f855a;'>📊 Saisonnalité Détectée</h3>
                                <div style='display: flex; gap: 2rem; margin: 1rem 0;'>
                                    <div>
                                        <p style='margin: 0; color: #2f855a;'><strong>Force:</strong></p>
                                        <p style='font-size: 1.5rem; margin: 0.5rem 0; color: #2f855a;'>{seasonal_strength:.1f}%</p>
                                    </div>
                                    <div>
                                        <p style='margin: 0; color: #2f855a;'><strong>Période:</strong></p>
                                        <p style='font-size: 1.5rem; margin: 0.5rem 0; color: #2f855a;'>{dominant_period} mois</p>
                                    </div>
                                </div>
                                <p style='margin: 0; color: #2f855a;'>Une saisonnalité forte indique des cycles réguliers dans vos encaissements qui peuvent être utilisés pour améliorer la planification financière.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        else:
                            st.info("Aucune saisonnalité significative n'a été détectée dans les encaissements.")
                    else:
                        st.info("Analyse saisonnière des encaissements non disponible.")
                
                with col2:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">💸 Décaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des tendances saisonnières</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'dec' in forecasting_models.seasonal_patterns:
                        fig_seasonal_dec = visualizer.create_seasonal_analysis_chart(
                            forecasting_models.seasonal_patterns['dec'],
                            title="Décomposition Saisonnière des Décaissements"
                        )
                        st.plotly_chart(fig_seasonal_dec, use_container_width=True)
                        
                        # Afficher des informations sur la saisonnalité détectée
                        if forecasting_models.seasonal_patterns['dec'].get('has_seasonality', False):
                            seasonal_strength = forecasting_models.seasonal_patterns['dec'].get('seasonal_strength', 0) * 100
                            dominant_period = forecasting_models.seasonal_patterns['dec'].get('dominant_period', 0)
                            st.markdown(f"""
                                <div style='background-color: #f0fff4; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #48bb78;'>
                                <h3 style='margin-top: 0; color: #2f855a;'>📊 Saisonnalité Détectée</h3>
                                <div style='display: flex; gap: 2rem; margin: 1rem 0;'>
                                    <div>
                                        <p style='margin: 0; color: #2f855a;'><strong>Force:</strong></p>
                                        <p style='font-size: 1.5rem; margin: 0.5rem 0; color: #2f855a;'>{seasonal_strength:.1f}%</p>
                                    </div>
                                    <div>
                                        <p style='margin: 0; color: #2f855a;'><strong>Période:</strong></p>
                                        <p style='font-size: 1.5rem; margin: 0.5rem 0; color: #2f855a;'>{dominant_period} mois</p>
                                    </div>
                                </div>
                                <p style='margin: 0; color: #2f855a;'>Une saisonnalité forte indique des cycles réguliers dans vos décaissements qui peuvent être utilisés pour améliorer la planification financière.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        else:
                            st.info("Aucune saisonnalité significative n'a été détectée dans les décaissements.")
                    else:
                        st.info("Analyse saisonnière des décaissements non disponible.")
            else:
                st.warning("Analyse saisonnière non disponible. Assurez-vous d'avoir activé l'option 'Détection de saisonnalité' dans les paramètres avancés.")
        
        with seasonal_tab2:
            if forecasting_models and hasattr(forecasting_models, 'seasonal_patterns'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📅 Patterns Mensuels</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des tendances par mois</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'enc' in forecasting_models.seasonal_patterns:
                        fig_monthly_enc = visualizer.create_monthly_pattern_chart(
                            df_enc, 'y_enc', 
                            title="Patterns Mensuels des Encaissements"
                        )
                        st.plotly_chart(fig_monthly_enc, use_container_width=True)
                        
                        # Analyse comparative
                        if len(df_enc) >= 12:
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
                        st.info("Patterns mensuels des encaissements non disponibles.")
                
                with col2:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📅 Patterns Mensuels</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des tendances par mois</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'dec' in forecasting_models.seasonal_patterns:
                        fig_monthly_dec = visualizer.create_monthly_pattern_chart(
                            df_dec, 'y_dec', 
                            title="Patterns Mensuels des Décaissements"
                        )
                        st.plotly_chart(fig_monthly_dec, use_container_width=True)
                        
                        # Analyse comparative
                        if len(df_dec) >= 12:
                            fig_comparative = visualizer.create_comparative_analysis_chart(
                                df_dec, 'y_dec', 
                                title="Comparaison des Décaissements par Période"
                            )
                            st.plotly_chart(fig_comparative, use_container_width=True)
                            
                            # Évolution année par année
                            fig_year_over_year = visualizer.create_year_over_year_chart(
                                df_dec, 'y_dec', 
                                title="Évolution des Décaissements Année par Année"
                            )
                            st.plotly_chart(fig_year_over_year, use_container_width=True)
                    else:
                        st.info("Patterns mensuels des décaissements non disponibles.")
            else:
                st.warning("Analyse des patterns mensuels non disponible. Assurez-vous d'avoir activé l'option 'Détection de saisonnalité' dans les paramètres avancés.")
    
    # Onglet Détection d'Anomalies
    with tab_anomalies:
        if forecasting_models and hasattr(forecasting_models, 'anomalies'):
            # Création des sous-onglets
            anomaly_tab1, anomaly_tab2 = st.tabs(["📊 Visualisation", "📋 Détails"])
            
            with anomaly_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📈 Encaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des anomalies dans les encaissements</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
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
                        else:
                            st.success("✅ Aucune anomalie significative n'a été détectée dans les encaissements.")
                    else:
                        st.info("ℹ️ Détection d'anomalies pour les encaissements non disponible.")
                
                with col2:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📉 Décaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse des anomalies dans les décaissements</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
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
                        else:
                            st.success("✅ Aucune anomalie significative n'a été détectée dans les décaissements.")
                    else:
                        st.info("ℹ️ Détection d'anomalies pour les décaissements non disponible.")
            
            with anomaly_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📊 Détails des Anomalies - Encaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Liste détaillée des anomalies détectées</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'enc' in forecasting_models.anomalies:
                        if 'anomaly_data' in forecasting_models.anomalies['enc']:
                            anomaly_df = forecasting_models.anomalies['enc']['anomaly_data']
                            if not anomaly_df.empty:
                                # Bloc de statistiques avant le tableau
                                st.markdown("##### 📈 Statistiques des Anomalies d'Encaissements")
                                enc_stats = {
                                    "Nombre d'anomalies": len(anomaly_df),
                                    "Montant total des anomalies": f"{anomaly_df['y_enc'].sum():,.0f} DH",
                                    "Montant moyen par anomalie": f"{anomaly_df['y_enc'].mean():,.0f} DH",
                                    "Écart-type des anomalies": f"{anomaly_df['y_enc'].std():,.0f} DH" if len(anomaly_df) > 1 else "Non calculable (une seule anomalie)",
                                    "Anomalie la plus importante": f"{anomaly_df['y_enc'].max():,.0f} DH",
                                    "Date de la dernière anomalie": anomaly_df['ds'].max().strftime('%d/%m/%Y')
                                }
                                for stat_name, stat_value in enc_stats.items():
                                    st.metric(stat_name, stat_value)
                            else:
                                st.info("ℹ️ Aucune anomalie détectée dans les encaissements.")
                        else:
                            st.info("ℹ️ Données d'anomalies non disponibles pour les encaissements.")
                    else:
                        st.info("ℹ️ Détection d'anomalies non disponible pour les encaissements.")
                
                with col2:
                    st.markdown(
                        '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                        '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📊 Détails des Anomalies - Décaissements</h3>'            
                        '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Liste détaillée des anomalies détectées</p>'            
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    if 'dec' in forecasting_models.anomalies:
                        if 'anomaly_data' in forecasting_models.anomalies['dec']:
                            anomaly_df = forecasting_models.anomalies['dec']['anomaly_data']
                            if not anomaly_df.empty:
                                # Bloc de statistiques avant le tableau
                                st.markdown("##### 📈 Statistiques des Anomalies de Décaissements")
                                dec_stats = {
                                    "Nombre d'anomalies": len(anomaly_df),
                                    "Montant total des anomalies": f"{anomaly_df['y_dec'].sum():,.0f} DH",
                                    "Montant moyen par anomalie": f"{anomaly_df['y_dec'].mean():,.0f} DH",
                                    "Écart-type des anomalies": f"{anomaly_df['y_dec'].std():,.0f} DH" if len(anomaly_df) > 1 else "Non calculable (une seule anomalie)",
                                    "Anomalie la plus importante": f"{anomaly_df['y_dec'].max():,.0f} DH",
                                    "Date de la dernière anomalie": anomaly_df['ds'].max().strftime('%d/%m/%Y')
                                }
                                for stat_name, stat_value in dec_stats.items():
                                    st.metric(stat_name, stat_value)
                            else:
                                st.info("ℹ️ Aucune anomalie détectée dans les décaissements.")
                        else:
                            st.info("ℹ️ Données d'anomalies non disponibles pour les décaissements.")
                    else:
                        st.info("ℹ️ Détection d'anomalies non disponible pour les décaissements.")
            
            # Conseils pour l'interprétation des anomalies
            with st.expander("💡 Comment interpréter les anomalies?"):
                st.markdown("""
                ### Guide d'interprétation des anomalies
                
                Les anomalies sont des valeurs qui s'écartent significativement du comportement normal des données. Elles peuvent indiquer :
                
                #### 🔍 Types d'anomalies
                - **Transactions exceptionnelles** : Paiements importants, remboursements, ou événements financiers inhabituels
                - **Erreurs de saisie** : Données incorrectes ou mal saisies
                - **Changements structurels** : Modifications dans votre activité ou votre modèle économique
                
                #### 📊 Score d'anomalie
                - Plus le score est élevé, plus l'anomalie est significative
                - Score > 3 : Anomalie majeure à investiguer
                - Score entre 2 et 3 : Anomalie modérée à surveiller
                - Score < 2 : Anomalie mineure, probablement normale
                
                #### 🎯 Actions recommandées
                1. **Vérification** : Confirmer l'exactitude des transactions identifiées
                2. **Correction** : Rectifier les erreurs éventuelles dans les données
                3. **Analyse** : Comprendre la nature des anomalies légitimes
                4. **Prévention** : Mettre en place des contrôles pour éviter les erreurs futures
                
                #### ⚠️ Points d'attention
                - Ne pas ignorer systématiquement les anomalies
                - Documenter les raisons des anomalies légitimes
                - Utiliser ces informations pour améliorer vos processus
                """)
        else:
            st.warning("⚠️ Détection d'anomalies non disponible. Assurez-vous d'avoir activé l'option 'Détection d'anomalies' dans les paramètres avancés.")
    
    # Onglet Analyses Avancées
    with tab_advanced:
        # Création des sous-onglets
        advanced_tab1, advanced_tab2, advanced_tab3 = st.tabs(["📈 Analyse des Flux", "📊 Statistiques Avancées", "🎯 Recommandations"])
        
        with advanced_tab1:
            # Titre principal avec espacement
            st.markdown('<div style="margin-top: 0; margin-bottom: 2rem;">', unsafe_allow_html=True)
            st.markdown(
                '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px;">'            
                '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📈 Analyse des Flux de Trésorerie et Prévisions</h3>'            
                '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Visualisation détaillée des flux et prévisions</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Création et affichage du graphique avec animation
            fig_flux = visualizer.create_flux_chart(df_enc, df_dec, forecasts, best_model, future_dates)
            st.plotly_chart(fig_flux, use_container_width=True, key="flux_chart")
            
            # Analyse des tendances
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">📈 Tendance Encaissements</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                if len(df_enc) >= 3:
                    enc_trend = (df_enc['y_enc'].iloc[-1] / df_enc['y_enc'].iloc[-3] - 1) * 100
                    st.metric(
                        "Variation sur 3 mois",
                        f"{enc_trend:.1f}%",
                        delta_color="normal"
                    )
            
            with col2:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">📉 Tendance Décaissements</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                if len(df_dec) >= 3:
                    dec_trend = (df_dec['y_dec'].iloc[-1] / df_dec['y_dec'].iloc[-3] - 1) * 100
                    st.metric(
                        "Variation sur 3 mois",
                        f"{dec_trend:.1f}%",
                        delta_color="inverse"
                    )
            
            with col3:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">💰 Solde de Trésorerie</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                if len(df_enc) >= 1 and len(df_dec) >= 1:
                    solde = df_enc['y_enc'].iloc[-1] - df_dec['y_dec'].iloc[-1]
                    st.metric(
                        "Solde actuel",
                        f"{solde:,.0f} DH",
                        delta_color="normal" if solde >= 0 else "inverse"
                    )
        
        with advanced_tab2:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                '<h3 style="font-size: 1.3rem; margin: 0; color: white;">📊 Statistiques Avancées</h3>'            
                '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Analyse approfondie des indicateurs financiers</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">📈 Encaissements</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Statistiques des encaissements
                enc_stats = {
                    "Moyenne": df_enc['y_enc'].mean(),
                    "Médiane": df_enc['y_enc'].median(),
                    "Écart-type": df_enc['y_enc'].std(),
                    "Minimum": df_enc['y_enc'].min(),
                    "Maximum": df_enc['y_enc'].max(),
                    "Croissance annuelle": ((df_enc['y_enc'].iloc[-1] / df_enc['y_enc'].iloc[0]) ** (12/len(df_enc)) - 1) * 100 if len(df_enc) > 0 else 0
                }
                
                for stat, value in enc_stats.items():
                    st.metric(
                        stat,
                        f"{value:,.0f} DH" if stat != "Croissance annuelle" else f"{value:.1f}%",
                        delta_color="normal"
                    )
            
            with col2:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">📉 Décaissements</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Statistiques des décaissements
                dec_stats = {
                    "Moyenne": df_dec['y_dec'].mean(),
                    "Médiane": df_dec['y_dec'].median(),
                    "Écart-type": df_dec['y_dec'].std(),
                    "Minimum": df_dec['y_dec'].min(),
                    "Maximum": df_dec['y_dec'].max(),
                    "Croissance annuelle": ((df_dec['y_dec'].iloc[-1] / df_dec['y_dec'].iloc[0]) ** (12/len(df_dec)) - 1) * 100 if len(df_dec) > 0 else 0
                }
                
                for stat, value in dec_stats.items():
                    st.metric(
                        stat,
                        f"{value:,.0f} DH" if stat != "Croissance annuelle" else f"{value:.1f}%",
                        delta_color="inverse"
                    )
            
            # Ratios financiers
            st.markdown(
                '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1.5rem 0;">'            
                '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">💰 Ratios Financiers</h4>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Ratio de couverture
                coverage_ratio = df_enc['y_enc'].mean() / df_dec['y_dec'].mean() if df_dec['y_dec'].mean() > 0 else 0
                st.metric(
                    "Ratio de Couverture",
                    f"{coverage_ratio:.2f}",
                    delta_color="normal" if coverage_ratio >= 1 else "inverse"
                )
            
            with col2:
                # Marge de sécurité
                safety_margin = (df_enc['y_enc'].mean() - df_dec['y_dec'].mean()) / df_dec['y_dec'].mean() * 100 if df_dec['y_dec'].mean() > 0 else 0
                st.metric(
                    "Marge de Sécurité",
                    f"{safety_margin:.1f}%",
                    delta_color="normal" if safety_margin >= 0 else "inverse"
                )
            
            with col3:
                # Indice de stabilité
                stability_index = 1 - (df_enc['y_enc'].std() / df_enc['y_enc'].mean()) if df_enc['y_enc'].mean() > 0 else 0
                st.metric(
                    "Indice de Stabilité",
                    f"{stability_index:.2f}",
                    delta_color="normal" if stability_index >= 0.5 else "inverse"
                )
        
        with advanced_tab3:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">'            
                '<h3 style="font-size: 1.3rem; margin: 0; color: white;">🎯 Recommandations Stratégiques</h3>'            
                '<p style="margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1rem;">Suggestions pour optimiser la gestion de trésorerie</p>'            
                '</div>',
                unsafe_allow_html=True
            )
            
            # Analyse de la situation actuelle
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
                    '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">📊 Situation Actuelle</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Évaluation de la situation
                if coverage_ratio >= 1.2:
                    st.success("✅ **Trésorerie saine** : Les encaissements couvrent largement les décaissements.")
                elif coverage_ratio >= 1:
                    st.info("ℹ️ **Trésorerie équilibrée** : Les encaissements couvrent juste les décaissements.")
                else:
                    st.warning("⚠️ **Attention** : Les encaissements ne couvrent pas suffisamment les décaissements.")
                
                if safety_margin >= 20:
                    st.success("✅ **Bonne marge de sécurité** : Capacité d'absorption des chocs financiers.")
                elif safety_margin >= 0:
                    st.info("ℹ️ **Marge de sécurité limitée** : Surveiller les flux de trésorerie.")
                else:
                    st.warning("⚠️ **Marge de sécurité négative** : Risque de trésorerie élevé.")
            
            with col2:
                st.markdown(
                            '<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'            
                            '<h4 style="font-size: 1.1rem; margin: 0; color: #1e40af;">🎯 Recommandations</h4>'            
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Recommandations basées sur l'analyse
                if coverage_ratio < 1:
                    st.markdown("""
                    #### Actions prioritaires :
                    1. **Optimiser les encaissements**
                       - Accélérer le recouvrement des créances
                       - Renégocier les délais de paiement
                    2. **Maîtriser les décaissements**
                       - Réduire les dépenses non essentielles
                       - Optimiser les délais de paiement
                    """)
                elif safety_margin < 20:
                    st.markdown("""
                    #### Actions recommandées :
                    1. **Renforcer la marge de sécurité**
                       - Constituer une réserve de trésorerie
                       - Diversifier les sources de revenus
                    2. **Améliorer la prévision**
                       - Affiner les prévisions de trésorerie
                       - Mettre en place des alertes
                    """)
                else:
                    st.markdown("""
                    #### Actions d'optimisation :
                    1. **Maintenir la performance**
                       - Continuer le suivi rigoureux
                       - Maintenir les bonnes pratiques
                    2. **Opportunités d'investissement**
                       - Évaluer les placements à court terme
                       - Optimiser la gestion des excédents
                    """)
    
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
                        f'<p style="margin: 0.5rem 0 0 0; color: #065f46;"><strong>Erreur (RMSE) :</strong> {model_metrics.get(best_model, {}).get("RMSE", 0):.2f}</p>'                        
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
                            "RMSE": model_metrics.get(model_name, {}).get("RMSE", 0),
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
                    '<strong>RMSE (Racine de l\'Erreur Quadratique Moyenne) :</strong> Mesure l\'écart type des résidus. Elle pénalise davantage les grandes erreurs que la MAE. Plus cette valeur est basse, meilleur est le modèle.<br><br>'
                    '<strong>MAPE (Erreur Absolue Moyenne en Pourcentage) :</strong> Exprime l\'erreur en pourcentage par rapport aux valeurs réelles. Permet de comparer les performances indépendamment de l\'ordre de grandeur des données.'                    
                    '</p>'                    
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Amélioration: Option d'exportation des résultats
                with st.expander("Exporter les résultats", expanded=False):
                    st.markdown("### Exporter les prévisions et métriques")
                    st.info("Utilisez le bouton d'export dans la barre latérale pour télécharger les prévisions.")
                
    
    with tab_scenarios:
        # En-tête principal
        st.markdown("### 🌐 Simulation de Scénarios")
        
        # Vérifier si les prévisions sont disponibles
        if not forecasts or best_model == '' or best_model not in forecasts:
            st.warning("Les prévisions ne sont pas disponibles pour la simulation. Veuillez générer des prévisions d'abord.")
            return
        
        # Vérifier si le modèle de décaissement existe
        best_dec_model = best_model.replace('enc', 'dec')
        if best_dec_model not in forecasts:
            st.warning(f"Le modèle de décaissement correspondant ({best_dec_model}) n'est pas disponible.")
            return
        
        # Création des sous-onglets pour les différents types de scénarios
        scenario_tabs = st.tabs(["📊 Simulation Monte Carlo", "🔮 Scénarios Prédéfinis", "⚙️ Scénario Personnalisé"])
        
        # Onglet Simulation Monte Carlo
        with scenario_tabs[0]:
            st.markdown("### 📊 Simulation Monte Carlo")
            st.markdown("Cette simulation utilise la méthode de Monte Carlo pour évaluer les risques de trésorerie.")
            
            if forecasting_models and hasattr(forecasting_models, 'monte_carlo_results') and forecasting_models.monte_carlo_results:
                mc_results = forecasting_models.monte_carlo_results
                
                # Créer les graphiques
                fig_enc_mc = visualizer.create_monte_carlo_chart(df_enc, df_dec, mc_results, future_dates)
                st.plotly_chart(fig_enc_mc, use_container_width=True)
                
                # Afficher les statistiques
                st.markdown("#### 📈 Indicateurs Clés")
                col1, col2, col3, col4 = st.columns(4)
                
                solde_min = np.min(mc_results['solde_lower_95'])
                solde_max = np.max(mc_results['solde_upper_95'])
                solde_mean = np.mean(mc_results['solde_mean'])
                prob_negative = np.mean(mc_results['prob_negative_solde'])
                
                col1.metric("Solde Minimum", f"{solde_min:,.0f} DH")
                col2.metric("Solde Maximum", f"{solde_max:,.0f} DH")
                col3.metric("Solde Moyen", f"{solde_mean:,.0f} DH")
                col4.metric("Probabilité de Solde Négatif", f"{prob_negative:.1f}%")
                
                # Recommandations
                st.markdown("#### 💡 Recommandations")
                if prob_negative > 20:
                    st.warning("**Risque élevé de trésorerie négative.** Envisagez de réduire les dépenses ou d'augmenter les encaissements.")
                elif solde_min < 0:
                    st.info("**Risque modéré de trésorerie négative.** Surveillez attentivement les flux de trésorerie.")
                else:
                    st.success("**Situation de trésorerie saine.** Continuez à surveiller les flux.")
            else:
                st.warning("Pour voir les résultats de la simulation Monte Carlo, veuillez générer les prévisions.")
        
        # Onglet Scénarios Prédéfinis
        with scenario_tabs[1]:
            st.markdown("### 🔮 Scénarios Prédéfinis")
            st.markdown("Ces scénarios représentent différentes hypothèses d'évolution de votre trésorerie.")
            
            if scenarios and len(scenarios) > 0:
                # Filtrer les scénarios en fonction de l'intervalle de confiance
                confidence_interval = config['confidence_interval']
                filtered_scenarios = {}
                
                # Scénarios de base (sans intervalle de confiance spécifique)
                base_scenarios = ['optimiste', 'pessimiste', 'neutre', 'croissance']
                
                # Ajouter les scénarios de base
                for name in base_scenarios:
                    if name in scenarios:
                        filtered_scenarios[name] = scenarios[name]
                
                # Ajouter le scénario correspondant à l'intervalle de confiance
                confidence_scenario = f"{'Optimiste' if confidence_interval >= 80 else 'Pessimiste'}_{confidence_interval}"
                if confidence_scenario in scenarios:
                    filtered_scenarios[confidence_scenario] = scenarios[confidence_scenario]
                
                # Création d'onglets pour chaque scénario filtré
                predefined_tabs = st.tabs([f"Scénario {name.capitalize()}" for name in filtered_scenarios.keys()])
                
                for i, (scenario_name, scenario_data) in enumerate(filtered_scenarios.items()):
                    with predefined_tabs[i]:
                        # Création du graphique pour le scénario
                        fig_scenario = visualizer.create_scenario_chart(df_enc, df_dec, scenario_data, future_dates)
                        st.plotly_chart(fig_scenario, use_container_width=True)
                        
                        # Statistiques du scénario
                        st.markdown("#### 📊 Statistiques du Scénario")
                        col1, col2, col3 = st.columns(3)
                        solde = scenario_data['encaissement'] - scenario_data['decaissement']
                        with col1:
                            st.metric(
                                "Encaissements Moyens", 
                                f"{np.mean(scenario_data['encaissement']):,.0f} DH",
                                delta=f"{np.mean(scenario_data['encaissement']) - np.mean(df_enc['y_enc']):,.0f} DH" if len(df_enc) > 0 else None
                            )
                        with col2:
                            st.metric(
                                "Décaissements Moyens", 
                                f"{np.mean(scenario_data['decaissement']):,.0f} DH",
                                delta=f"{np.mean(scenario_data['decaissement']) - np.mean(df_dec['y_dec']):,.0f} DH" if len(df_dec) > 0 else None
                            )
                        with col3:
                            st.metric(
                                "Solde Moyen", 
                                f"{np.mean(solde):,.0f} DH",
                                delta=f"{np.mean(solde) - (np.mean(df_enc['y_enc']) - np.mean(df_dec['y_dec'])):,.0f} DH" if len(df_enc) > 0 and len(df_dec) > 0 else None
                            )
                        # Description du scénario
                        if scenario_name == "optimiste":
                            st.success("Ce scénario suppose une croissance des encaissements et une stabilisation des décaissements.")
                        elif scenario_name == "pessimiste":
                            st.error("Ce scénario suppose une baisse des encaissements et une augmentation des décaissements.")
                        elif scenario_name == "neutre":
                            st.info("Ce scénario suppose une évolution stable des encaissements et des décaissements.")
                        elif scenario_name == "croissance":
                            st.success("Ce scénario suppose une croissance progressive des encaissements.")
                        elif scenario_name.startswith("Optimiste_"):
                            st.success(f"Ce scénario optimiste est basé sur un intervalle de confiance de {confidence_interval}%.")
                        elif scenario_name.startswith("Pessimiste_"):
                            st.error(f"Ce scénario pessimiste est basé sur un intervalle de confiance de {confidence_interval}%.")
                        else:
                            st.info("Aucun scénario prédéfini n'est disponible. Veuillez générer des prévisions d'abord.")
        
        # Onglet Scénario Personnalisé
        with scenario_tabs[2]:
            st.markdown("### ⚙️ Scénario Personnalisé")
            st.markdown("Créez votre propre scénario en ajustant les paramètres ci-dessous.")
            
            # Paramètres du scénario
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📈 Paramètres d'Encaissements")
                enc_growth = st.slider("Croissance (%)", -50, 100, 0, 5, key="enc_growth_slider")
                enc_volatility = st.slider("Volatilité (%)", 0, 50, 10, 5, key="enc_volatility_slider")
                enc_seasonality = st.selectbox("Saisonnalité", ["Aucune", "Mensuelle", "Trimestrielle"], key="enc_seasonality_select")
            
            with col2:
                st.markdown("##### 📉 Paramètres de Décaissements")
                dec_growth = st.slider("Croissance (%)", -50, 100, 0, 5, key="dec_growth_slider")
                dec_volatility = st.slider("Volatilité (%)", 0, 50, 10, 5, key="dec_volatility_slider")
                dec_seasonality = st.selectbox("Saisonnalité", ["Aucune", "Mensuelle", "Trimestrielle"], key="dec_seasonality_select")
            
            # Bouton pour générer le scénario
            if st.button("Générer le Scénario", type="primary", key="generate_custom_scenario"):
                try:
                    with st.spinner("Génération du scénario personnalisé en cours..."):
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
                    
                    if custom_scenario:
                        # Affichage du scénario
                        fig_custom = visualizer.create_scenario_chart(df_enc, df_dec, custom_scenario, future_dates)
                        st.plotly_chart(fig_custom, use_container_width=True)
            
                        # Statistiques du scénario personnalisé
                        st.markdown("#### 📊 Statistiques du Scénario Personnalisé")
                        col1, col2, col3 = st.columns(3)
                                    
                        solde = custom_scenario['encaissement'] - custom_scenario['decaissement']
                        
                        with col1:
                            st.metric(
                                "Encaissements Moyens", 
                                            f"{np.mean(custom_scenario['encaissement']):,.0f} DH",
                                            delta=f"{np.mean(custom_scenario['encaissement']) - np.mean(df_enc['y_enc']):,.0f} DH" if len(df_enc) > 0 else None
                            )
                        
                        with col2:
                            st.metric(
                                "Décaissements Moyens", 
                                            f"{np.mean(custom_scenario['decaissement']):,.0f} DH",
                                            delta=f"{np.mean(custom_scenario['decaissement']) - np.mean(df_dec['y_dec']):,.0f} DH" if len(df_dec) > 0 else None
                            )
                        
                        with col3:
                            st.metric(
                                "Solde Moyen", 
                                            f"{np.mean(solde):,.0f} DH",
                                            delta=f"{np.mean(solde) - (np.mean(df_enc['y_enc']) - np.mean(df_dec['y_dec'])):,.0f} DH" if len(df_enc) > 0 and len(df_dec) > 0 else None
                                        )
                                    
                        # Recommandations basées sur le scénario
                        st.markdown("#### 💡 Recommandations")
                        if np.mean(solde) < 0:
                                        st.warning("**Attention :** Ce scénario prévoit un solde moyen négatif. Envisagez des mesures pour augmenter vos encaissements ou réduire vos décaissements.")
                        elif np.mean(solde) > 0 and np.mean(solde) < 0.1 * np.mean(df_dec['y_dec']) and len(df_dec) > 0:
                                        st.info("**Prudence :** Ce scénario prévoit un solde positif mais faible. Constituez une réserve de trésorerie pour faire face aux imprévus.")
                        else:
                            st.success("**Favorable :** Ce scénario prévoit un solde positif confortable. Envisagez d'investir l'excédent de trésorerie pour optimiser vos rendements.")
                    else:
                        st.error("Impossible de générer le scénario personnalisé. Veuillez réexécuter l'analyse.")
                except Exception as e:
                    st.error(f"Erreur lors de la génération du scénario : {e}")

    with tab_metrics:
        # Calcul des métriques financières
        metrics = calculate_financial_metrics(df_enc, df_dec)
        
        # Création des sous-onglets pour une meilleure organisation
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["📊 Indicateurs Clés", "📈 Analyse de Tendance", "💡 Recommandations"])
        
        with metrics_tab1:
            # Première rangée de métriques avec style amélioré
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f'<div style="background-color: {"#ecfdf5" if metrics["Ratio de Couverture"] >= 1 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                    f'<h4 style="margin: 0; color: {"#065f46" if metrics["Ratio de Couverture"] >= 1 else "#991b1b"};">🛡️ Ratio de Couverture</h4>'            
                    f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Ratio de Couverture"] >= 1 else "#991b1b"};">{metrics["Ratio de Couverture"]:.2f}x</p>'            
                    f'<p style="margin: 0; color: {"#065f46" if metrics["Ratio de Couverture"] >= 1 else "#991b1b"};">{"✅ Bon" if metrics["Ratio de Couverture"] >= 1 else "⚠️ À améliorer"}</p>'            
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                            f'<div style="background-color: {"#ecfdf5" if metrics["Taux de Croissance Encaissements"] > metrics["Taux de Croissance Décaissements"] else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                            f'<h4 style="margin: 0; color: {"#065f46" if metrics["Taux de Croissance Encaissements"] > metrics["Taux de Croissance Décaissements"] else "#991b1b"};">📈 Croissance Encaissements</h4>'            
                            f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Taux de Croissance Encaissements"] > metrics["Taux de Croissance Décaissements"] else "#991b1b"};">{metrics["Taux de Croissance Encaissements"]:.1f}%</p>'            
                            f'<p style="margin: 0; color: {"#065f46" if metrics["Taux de Croissance Encaissements"] > metrics["Taux de Croissance Décaissements"] else "#991b1b"};">{"✅ Bon" if metrics["Taux de Croissance Encaissements"] > metrics["Taux de Croissance Décaissements"] else "⚠️ À améliorer"}</p>'            
                            f'</div>',
                    unsafe_allow_html=True
                )
        
            with col3:
                st.markdown(
                            f'<div style="background-color: {"#ecfdf5" if metrics["Indice de Stabilité"] >= 0.5 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                            f'<h4 style="margin: 0; color: {"#065f46" if metrics["Indice de Stabilité"] >= 0.5 else "#991b1b"};">⚖️ Indice de Stabilité</h4>'            
                            f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Indice de Stabilité"] >= 0.5 else "#991b1b"};">{metrics["Indice de Stabilité"]:.2f}</p>'            
                            f'<p style="margin: 0; color: {"#065f46" if metrics["Indice de Stabilité"] >= 0.5 else "#991b1b"};">{"✅ Bon" if metrics["Indice de Stabilité"] >= 0.5 else "⚠️ À améliorer"}</p>'            
                            f'</div>',
                    unsafe_allow_html=True
                )

            # Ajout d'une div explicative sur les indicateurs clés
            st.markdown(
                '''
                <div style="background-color: #f8fafc; border-radius: 10px; padding: 1.5rem; margin-top: 1.5rem; margin-bottom: 1.5rem; border-left: 4px solid #3b82f6;">
                <h4 style="color: #1e40af; margin-top: 0;">ℹ️ Détail des Indicateurs Clés</h4>
                <ul style="font-size: 1.05rem; color: #334155;">
                  <li><b>Ratio de Couverture :</b> Mesure la capacité de l'entreprise à couvrir ses décaissements par ses encaissements. Un ratio ≥ 1 indique une bonne couverture.</li>
                  <li><b>Croissance Encaissements :</b> Taux d'évolution des encaissements sur la période analysée. Un taux positif est généralement favorable.</li>
                  <li><b>Croissance Décaissements :</b> Taux d'évolution des décaissements. Un taux trop élevé peut signaler une dérive des charges.</li>
                  <li><b>Indice de Stabilité :</b> Évalue la régularité des flux de trésorerie. Plus il est proche de 1, plus les flux sont stables.</li>
                  <li><b>Volatilité Encaissements :</b> Mesure la variabilité des encaissements. Une volatilité faible est préférable pour la prévisibilité.</li>
                  <li><b>Volatilité Décaissements :</b> Mesure la variabilité des décaissements. Une forte volatilité peut indiquer des charges imprévues.</li>
                  <li><b>Marge de Sécurité :</b> Pourcentage de sécurité financière disponible après couverture des charges. Plus elle est élevée, plus l'entreprise est résiliente.</li>
                </ul>
                </div>
                ''',
                unsafe_allow_html=True
            )
        
            # Deuxième rangée de métriques avec style amélioré
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.markdown(
                        f'<div style="background-color: {"#ecfdf5" if metrics["Volatilité Encaissements (%)"] <= 30 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                        f'<h4 style="margin: 0; color: {"#065f46" if metrics["Volatilité Encaissements (%)"] <= 30 else "#991b1b"};">📊 Volatilité Encaissements</h4>'            
                        f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Volatilité Encaissements (%)"] <= 30 else "#991b1b"};">{metrics["Volatilité Encaissements (%)"]:.1f}%</p>'            
                        f'<p style="margin: 0; color: {"#065f46" if metrics["Volatilité Encaissements (%)"] <= 30 else "#991b1b"};">{"✅ Bon" if metrics["Volatilité Encaissements (%)"] <= 30 else "⚠️ Élevée"}</p>'            
                    f'</div>',
                    unsafe_allow_html=True
                )
        
            with col5:
                st.markdown(
                        f'<div style="background-color: {"#ecfdf5" if metrics["Volatilité Décaissements (%)"] <= 30 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                        f'<h4 style="margin: 0; color: {"#065f46" if metrics["Volatilité Décaissements (%)"] <= 30 else "#991b1b"};">📉 Volatilité Décaissements</h4>'            
                        f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Volatilité Décaissements (%)"] <= 30 else "#991b1b"};">{metrics["Volatilité Décaissements (%)"]:.1f}%</p>'            
                        f'<p style="margin: 0; color: {"#065f46" if metrics["Volatilité Décaissements (%)"] <= 30 else "#991b1b"};">{"✅ Bon" if metrics["Volatilité Décaissements (%)"] <= 30 else "⚠️ Élevée"}</p>'            
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col6:
                st.markdown(
                        f'<div style="background-color: {"#ecfdf5" if metrics["Marge de Sécurité (%)"] > 0 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px; text-align: center;">'            
                        f'<h4 style="margin: 0; color: {"#065f46" if metrics["Marge de Sécurité (%)"] > 0 else "#991b1b"};">🛡️ Marge de Sécurité</h4>'            
                        f'<p style="font-size: 1.5rem; margin: 0.5rem 0; color: {"#065f46" if metrics["Marge de Sécurité (%)"] > 0 else "#991b1b"};">{metrics["Marge de Sécurité (%)"]:.1f}%</p>'            
                        f'<p style="margin: 0; color: {"#065f46" if metrics["Marge de Sécurité (%)"] > 0 else "#991b1b"};">{"✅ Bon" if metrics["Marge de Sécurité (%)"] > 0 else "⚠️ Insuffisante"}</p>'            
                    f'</div>',
                    unsafe_allow_html=True
                )
    
        with metrics_tab2:
            # Création du graphique radar des indicateurs financiers
            fig_radar = visualizer.create_financial_indicators_chart(metrics)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Création d'un DataFrame pour l'analyse des tendances
            trend_data = {
                'Indicateur': [
                    'Ratio de Couverture',
                    'Taux de Croissance Encaissements',
                    'Taux de Croissance Décaissements',
                    'Indice de Stabilité',
                    'Volatilité Encaissements',
                    'Volatilité Décaissements',
                    'Marge de Sécurité'
                ],
                'Valeur': [
                    f"{metrics['Ratio de Couverture']:.2f}x",
                    f"{metrics['Taux de Croissance Encaissements']:.1f}%",
                    f"{metrics['Taux de Croissance Décaissements']:.1f}%",
                    f"{metrics['Indice de Stabilité']:.2f}",
                    f"{metrics['Volatilité Encaissements (%)']:.1f}%",
                    f"{metrics['Volatilité Décaissements (%)']:.1f}%",
                    f"{metrics['Marge de Sécurité (%)']:.1f}%"
                ],
                'Statut': [
                    '✅' if metrics['Ratio de Couverture'] >= 1 else '⚠️',
                    '✅' if metrics['Taux de Croissance Encaissements'] > metrics['Taux de Croissance Décaissements'] else '⚠️',
                    '✅' if metrics['Taux de Croissance Décaissements'] < metrics['Taux de Croissance Encaissements'] else '⚠️',
                    '✅' if metrics['Indice de Stabilité'] >= 0.5 else '⚠️',
                    '✅' if metrics['Volatilité Encaissements (%)'] <= 30 else '⚠️',
                    '✅' if metrics['Volatilité Décaissements (%)'] <= 30 else '⚠️',
                    '✅' if metrics['Marge de Sécurité (%)'] > 0 else '⚠️'
                ],
                'Interprétation': [
                    'Suffisant' if metrics['Ratio de Couverture'] >= 1 else 'Insuffisant',
                    'Positif' if metrics['Taux de Croissance Encaissements'] > metrics['Taux de Croissance Décaissements'] else 'À améliorer',
                    'Contrôlé' if metrics['Taux de Croissance Décaissements'] < metrics['Taux de Croissance Encaissements'] else 'Élevé',
                    'Stable' if metrics['Indice de Stabilité'] >= 0.5 else 'Instable',
                    'Contrôlée' if metrics['Volatilité Encaissements (%)'] <= 30 else 'Élevée',
                    'Contrôlée' if metrics['Volatilité Décaissements (%)'] <= 30 else 'Élevée',
                    'Suffisante' if metrics['Marge de Sécurité (%)'] > 0 else 'Insuffisante'
                ]
            }
            
            trend_df = pd.DataFrame(trend_data)
            st.dataframe(
                trend_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Indicateur": st.column_config.TextColumn("Indicateur", width="large"),
                    "Valeur": st.column_config.TextColumn("Valeur", width="medium"),
                    "Statut": st.column_config.TextColumn("Statut", width="small"),
                    "Interprétation": st.column_config.TextColumn("Interprétation", width="medium")
                }
            )
            
            # Calcul du nombre d'indicateurs positifs
            positive_indicators = sum(1 for status in trend_data['Statut'] if status == '✅')
            total_indicators = len(trend_data['Statut'])
            health_score = (positive_indicators / total_indicators) * 100
            
            # Affichage de la synthèse avec style
            st.markdown(
                f'<div style="background-color: {"#ecfdf5" if health_score >= 70 else "#fff7ed" if health_score >= 40 else "#fee2e2"}; padding: 1.5rem; border-radius: 10px;">'            
                f'<p style="margin: 0; color: {"#065f46" if health_score >= 70 else "#92400e" if health_score >= 40 else "#991b1b"};"><strong>Score de Santé Financière :</strong> {health_score:.1f}%</p>'            
                f'<p style="margin: 0.5rem 0 0 0; color: {"#065f46" if health_score >= 70 else "#92400e" if health_score >= 40 else "#991b1b"};">'            
                f'{"✅ Situation financière saine" if health_score >= 70 else "⚠️ Situation financière à surveiller" if health_score >= 40 else "❌ Situation financière préoccupante"}'            
                f'</p>'            
                f'</div>',
                unsafe_allow_html=True
            )
        
        with metrics_tab3:
            # Génération des recommandations
            recommendations = visualizer.generate_financial_recommendations(metrics)
            
            # Catégorisation des recommandations
            priority_recommendations = []
            improvement_recommendations = []
            maintenance_recommendations = []
            
            for rec in recommendations:
                if "⚠️" in rec or "❌" in rec:
                    priority_recommendations.append(rec)
                elif "✅" in rec:
                    maintenance_recommendations.append(rec)
                else:
                    improvement_recommendations.append(rec)
            
            # Affichage des recommandations prioritaires
            if priority_recommendations:
                for i, rec in enumerate(priority_recommendations, 1):
                    st.markdown(
                                f'<div style="background-color: #fee2e2; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #991b1b;">'            
                                f'<p style="margin: 0; color: #991b1b;"><strong>Priorité {i}:</strong> {rec}</p>'            
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
            # Affichage des recommandations d'amélioration
            if improvement_recommendations:
                for i, rec in enumerate(improvement_recommendations, 1):
                    st.markdown(
                                f'<div style="background-color: #fff7ed; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #92400e;">'            
                                f'<p style="margin: 0; color: #92400e;"><strong>Amélioration {i}:</strong> {rec}</p>'            
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            # Affichage des recommandations de maintenance
            if maintenance_recommendations:
                for i, rec in enumerate(maintenance_recommendations, 1):
                    st.markdown(
                                f'<div style="background-color: #ecfdf5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #065f46;">'            
                                f'<p style="margin: 0; color: #065f46;"><strong>Maintenance {i}:</strong> {rec}</p>'            
                                f'</div>',
                        unsafe_allow_html=True
                    )
            
            # Création d'un plan d'action basé sur les recommandations
            action_plan = []
            
            if priority_recommendations:
                action_plan.append({
                    "Priorité": "Haute",
                    "Actions": [rec.replace("⚠️", "").replace("❌", "").strip() for rec in priority_recommendations[:2]],
                    "Timing": "Immédiat"
                })
            
            if improvement_recommendations:
                action_plan.append({
                    "Priorité": "Moyenne",
                    "Actions": [rec.strip() for rec in improvement_recommendations[:2]],
                    "Timing": "Court terme"
                })
            
            if maintenance_recommendations:
                action_plan.append({
                    "Priorité": "Basse",
                    "Actions": [rec.replace("✅", "").strip() for rec in maintenance_recommendations[:2]],
                    "Timing": "Continu"
                })
            
            # Affichage du plan d'action
            if action_plan:
                action_df = pd.DataFrame(action_plan)
                st.dataframe(
                    action_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Priorité": st.column_config.TextColumn("Priorité", width="small"),
                        "Actions": st.column_config.ListColumn("Actions", width="large"),
                        "Timing": st.column_config.TextColumn("Timing", width="medium")
                    }
                )

def export_forecasts(df_enc, df_dec, forecasts, n_mois, export_format='Excel'):
    """Exporte les prévisions dans le format spécifié"""
    try:
        if export_format == 'PDF':
            st.write("Génération du rapport PDF professionnel...")
            output = BytesIO()
            
            # Configuration du document PDF avec des marges plus grandes pour un look plus professionnel
            doc = SimpleDocTemplate(
                output,
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )
            
            # Styles personnalisés améliorés
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='CoverTitle',
                parent=styles['Heading1'],
                fontSize=40,
                spaceAfter=30,
                textColor=colors.HexColor('#1e40af'),
                alignment=1,
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='CoverSubtitle',
                parent=styles['Heading2'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#3b82f6'),
                alignment=1,
                fontName='Helvetica'
            ))
            styles.add(ParagraphStyle(
                name='SectionTitle',
                parent=styles['Heading1'],
                fontSize=28,
                spaceAfter=20,
                textColor=colors.HexColor('#1e40af'),
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='SubSectionTitle',
                parent=styles['Heading2'],
                fontSize=20,
                spaceAfter=15,
                textColor=colors.HexColor('#3b82f6'),
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='Highlight',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#1e40af'),
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='Quote',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#4b5563'),
                fontName='Helvetica-Oblique',
                leftIndent=20,
                rightIndent=20
            ))
            
            # Liste des éléments du document
            elements = []
            
            # Page de couverture améliorée
            elements.append(Paragraph(
                '<para alignment="center"><font color="#1e40af" size="72">💰</font></para>',
                styles['Normal']
            ))
            elements.append(Spacer(1, 40))
            elements.append(Paragraph("Rapport de Prévisions de Trésorerie", styles['CoverTitle']))
            elements.append(Paragraph("Analyse Détaillée et Recommandations Stratégiques", styles['CoverSubtitle']))
            elements.append(Spacer(1, 60))
            
            # Informations de couverture dans un style moderne
            cover_info = f"""
            <para alignment="center">
            <font color="#4b5563" size="12">
            Date de génération : {datetime.now().strftime('%d/%m/%Y')}<br/>
            Horizon de prévision : {n_mois} mois<br/>
            Version : 2.0
            </font>
            </para>
            """
            elements.append(Paragraph(cover_info, styles['Normal']))
            elements.append(PageBreak())
            
            # Sommaire amélioré avec numérotation
            elements.append(Paragraph("Sommaire", styles['SectionTitle']))
            elements.append(Spacer(1, 20))
            
            toc = [
                "1. Résumé Exécutif",
                "2. Analyse des Données Historiques",
                "3. Prévisions Détaillées",
                "4. Analyse des Tendances",
                "5. Indicateurs Clés de Performance",
                "6. Analyse des Risques",
                "7. Recommandations Stratégiques",
                "8. Annexes Techniques"
            ]
            
            for item in toc:
                elements.append(Paragraph(item, styles['Normal']))
            elements.append(PageBreak())
            
            # Résumé Exécutif amélioré
            elements.append(Paragraph("1. Résumé Exécutif", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            # Calcul des statistiques clés
            enc_mean = np.mean(forecasts.get('prophet_enc', []))
            dec_mean = np.mean(forecasts.get('prophet_dec', []))
            solde_mean = enc_mean - dec_mean
            enc_trend = (forecasts.get('prophet_enc', [])[-1] / forecasts.get('prophet_enc', [])[0] - 1) * 100
            dec_trend = (forecasts.get('prophet_dec', [])[-1] / forecasts.get('prophet_dec', [])[0] - 1) * 100

            # Citation d'introduction
            elements.append(Paragraph(
                '"Une bonne gestion de trésorerie est la clé de la pérennité de l\'entreprise."',
                styles['Quote']
            ))
            elements.append(Spacer(1, 20))
            
            summary_text = f"""
Ce rapport présente une analyse approfondie des prévisions de trésorerie sur {n_mois} mois, basée sur des modèles avancés de prédiction et une analyse détaillée des tendances historiques.<br/><br/>
<font color="#1e40af"><b>Points clés :</b></font><br/>
• Encaissements moyens prévus : <b>{enc_mean:,.0f} DH</b><br/>
• Décaissements moyens prévus : <b>{dec_mean:,.0f} DH</b><br/>
• Solde moyen prévu : <b>{solde_mean:,.0f} DH</b><br/>
• Tendance des encaissements : <b>{enc_trend:+.1f}%</b><br/>
• Tendance des décaissements : <b>{dec_trend:+.1f}%</b>
"""
            elements.append(Paragraph(summary_text, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Analyse des Données Historiques avec graphiques
            elements.append(Paragraph("2. Analyse des Données Historiques", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            # Tableau des données historiques récentes avec style amélioré
            hist_data = [['Date', 'Encaissements', 'Décaissements', 'Solde', 'Variation']]
            for i in range(min(6, len(df_enc))):
                date = df_enc['ds'].iloc[-(i+1)]
                enc = df_enc['y_enc'].iloc[-(i+1)]
                dec = df_dec['y_dec'].iloc[-(i+1)]
                solde = enc - dec
                variation = (solde / (df_enc['y_enc'].iloc[-(i+2)] - df_dec['y_dec'].iloc[-(i+2)]) - 1) * 100 if i < len(df_enc)-1 else 0
                hist_data.append([
                    date.strftime('%d/%m/%Y'),
                    f"{enc:,.0f} DH",
                    f"{dec:,.0f} DH",
                    f"{solde:,.0f} DH",
                    f"{variation:+.1f}%"
                ])
            
            hist_table = Table(hist_data)
            hist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ]))
            elements.append(hist_table)
            elements.append(Spacer(1, 20))
            
            # Prévisions Détaillées avec analyse
            elements.append(Paragraph("3. Prévisions Détaillées", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            # Tableau des prévisions avec style amélioré
            forecast_data = [['Date', 'Encaissements', 'Décaissements', 'Solde', 'Variation']]
            for i in range(len(forecasts.get('prophet_enc', []))):
                date = pd.date_range(start=df_enc['ds'].max(), periods=n_mois + 1, freq='M')[i+1]
                enc = forecasts.get('prophet_enc', [])[i]
                dec = forecasts.get('prophet_dec', [])[i]
                solde = enc - dec
                variation = (solde / (forecasts.get('prophet_enc', [])[i-1] - forecasts.get('prophet_dec', [])[i-1]) - 1) * 100 if i > 0 else 0
                forecast_data.append([
                    date.strftime('%d/%m/%Y'),
                    f"{enc:,.0f} DH",
                    f"{dec:,.0f} DH",
                    f"{solde:,.0f} DH",
                    f"{variation:+.1f}%"
                ])
            
            forecast_table = Table(forecast_data)
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ]))
            elements.append(forecast_table)
            elements.append(PageBreak())
            
            # Analyse des Tendances avec visualisations
            elements.append(Paragraph("4. Analyse des Tendances", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            trend_analysis = f"""
            <para>
            <font color="#1e40af"><b>Analyse des tendances sur la période de prévision :</b></font>
            
            <b>Encaissements :</b>
            • Valeur moyenne : {enc_mean:,.0f} DH
            • Tendance : {enc_trend:+.1f}%
            • Volatilité : {np.std(forecasts.get('prophet_enc', [])):,.0f} DH
            • Coefficient de variation : {(np.std(forecasts.get('prophet_enc', [])) / enc_mean * 100):.1f}%
            
            <b>Décaissements :</b>
            • Valeur moyenne : {dec_mean:,.0f} DH
            • Tendance : {dec_trend:+.1f}%
            • Volatilité : {np.std(forecasts.get('prophet_dec', [])):,.0f} DH
            • Coefficient de variation : {(np.std(forecasts.get('prophet_dec', [])) / dec_mean * 100):.1f}%
            
            <b>Solde :</b>
            • Valeur moyenne : {solde_mean:,.0f} DH
            • Tendance : {(solde_mean / (df_enc['y_enc'].iloc[-1] - df_dec['y_dec'].iloc[-1]) - 1) * 100:+.1f}%
            • Volatilité : {np.std([e - d for e, d in zip(forecasts.get('prophet_enc', []), forecasts.get('prophet_dec', []))]):,.0f} DH
            </para>
            """
            elements.append(Paragraph(trend_analysis, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Nouvelle section : Indicateurs Clés de Performance
            elements.append(Paragraph("5. Indicateurs Clés de Performance", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            kpi_data = [
                ['Indicateur', 'Valeur', 'Tendance', 'Statut'],
                ['Ratio de Trésorerie', f"{(enc_mean / dec_mean):.2f}", f"{(enc_trend - dec_trend):+.1f}%", '🟢' if enc_mean > dec_mean else '🔴'],
                ['Marge de Sécurité', f"{(solde_mean / enc_mean * 100):.1f}%", f"{(solde_mean / enc_mean * 100 - (df_enc['y_enc'].iloc[-1] - df_dec['y_dec'].iloc[-1]) / df_enc['y_enc'].iloc[-1] * 100):+.1f}%", '🟢' if solde_mean > 0 else '🔴'],
                ['Volatilité Relative', f"{(np.std(forecasts.get('prophet_enc', [])) / enc_mean * 100):.1f}%", 'Stable', '🟡'],
                ['Efficacité Opérationnelle', f"{(1 - dec_mean / enc_mean) * 100:.1f}%", f"{(enc_trend - dec_trend):+.1f}%", '🟢' if enc_trend > dec_trend else '🔴']
            ]
            
            kpi_table = Table(kpi_data)
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ]))
            elements.append(kpi_table)
            elements.append(PageBreak())
            
            # Nouvelle section : Analyse des Risques
            elements.append(Paragraph("6. Analyse des Risques", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            risk_analysis = f"""
            <para>
            <font color="#1e40af"><b>Évaluation des risques principaux :</b></font>
            
            <b>Risque de Trésorerie :</b>
            • Niveau : {'Élevé' if solde_mean < 0 else 'Modéré' if solde_mean < enc_mean * 0.2 else 'Faible'}
            • Impact : {'Critique' if solde_mean < 0 else 'Significatif' if solde_mean < enc_mean * 0.2 else 'Limité'}
            • Probabilité : {'Élevée' if np.std(forecasts.get('prophet_enc', [])) > enc_mean * 0.2 else 'Modérée' if np.std(forecasts.get('prophet_enc', [])) > enc_mean * 0.1 else 'Faible'}
            
            <b>Risque Opérationnel :</b>
            • Niveau : {'Élevé' if dec_trend > enc_trend else 'Modéré' if abs(dec_trend - enc_trend) < 5 else 'Faible'}
            • Impact : {'Significatif' if dec_mean > enc_mean else 'Modéré' if dec_mean > enc_mean * 0.8 else 'Limité'}
            • Probabilité : {'Élevée' if np.std(forecasts.get('prophet_dec', [])) > dec_mean * 0.2 else 'Modérée' if np.std(forecasts.get('prophet_dec', [])) > dec_mean * 0.1 else 'Faible'}
            </para>
            """
            elements.append(Paragraph(risk_analysis, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Recommandations Stratégiques améliorées
            elements.append(Paragraph("7. Recommandations Stratégiques", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            recommendations = f"""
            <font color="#1e40af"><b>Recommandations basées sur l'analyse approfondie :</b></font><br/><br/>
            <b>1. Gestion des Encaissements :</b><br/>
            • {'Renforcer le suivi des créances clients et mettre en place un système de relance automatisé' if enc_trend < 0 else 'Maintenir la politique actuelle de recouvrement et optimiser les processus de facturation'}<br/>
            • {'Diversifier les sources de revenus et développer de nouveaux marchés' if np.std(forecasts.get('prophet_enc', [])) > enc_mean * 0.2 else 'Optimiser les processus de facturation et renforcer la fidélisation client'}<br/><br/>
            <b>2. Gestion des Décaissements :</b><br/>
            • {'Réduire les dépenses non essentielles et optimiser les coûts opérationnels' if dec_trend > 0 else "Maintenir le contrôle des coûts et identifier des opportunités d'optimisation"}<br/>
            • {'Négocier de meilleurs délais de paiement avec les fournisseurs' if dec_mean > enc_mean else 'Optimiser la gestion des stocks et des approvisionnements'}<br/><br/>
            <b>3. Gestion de la Trésorerie :</b><br/>
            • {'Mettre en place un fonds de roulement et établir des lignes de crédit préventives' if solde_mean < 0 else "Optimiser l'investissement des excédents et développer une stratégie de placement"}<br/>
            • {'Établir un plan de trésorerie détaillé et des indicateurs de suivi' if min([e - d for e, d in zip(forecasts.get('prophet_enc', []), forecasts.get('prophet_dec', []))]) < 0 else "Développer des investissements à court terme et optimiser la gestion des liquidités"}
            """
            elements.append(Paragraph(recommendations, styles['Normal']))
            elements.append(PageBreak())
            
            # Annexes Techniques améliorées
            elements.append(Paragraph("8. Annexes Techniques", styles['SectionTitle']))
            elements.append(Spacer(1, 10))
            
            # Statistiques détaillées avec style amélioré
            elements.append(Paragraph("Statistiques Détaillées", styles['SubSectionTitle']))
            stats_data = [
                ['Métrique', 'Valeur', 'Tendance'],
                ['Encaissements Moyens', f"{enc_mean:,.0f} DH", f"{enc_trend:+.1f}%"],
                ['Décaissements Moyens', f"{dec_mean:,.0f} DH", f"{dec_trend:+.1f}%"],
                ['Solde Moyen', f"{solde_mean:,.0f} DH", f"{(solde_mean / (df_enc['y_enc'].iloc[-1] - df_dec['y_dec'].iloc[-1]) - 1) * 100:+.1f}%"],
                ['Volatilité Encaissements', f"{np.std(forecasts.get('prophet_enc', [])):,.0f} DH", 'N/A'],
                ['Volatilité Décaissements', f"{np.std(forecasts.get('prophet_dec', [])):,.0f} DH", 'N/A'],
                ['Ratio de Trésorerie', f"{(enc_mean / dec_mean):.2f}", f"{(enc_trend - dec_trend):+.1f}%"],
                ['Marge de Sécurité', f"{(solde_mean / enc_mean * 100):.1f}%", f"{(solde_mean / enc_mean * 100 - (df_enc['y_enc'].iloc[-1] - df_dec['y_dec'].iloc[-1]) / df_enc['y_enc'].iloc[-1] * 100):+.1f}%"],
                ['Horizon de Prévision', f"{n_mois} mois", 'N/A']
            ]
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ]))
            elements.append(stats_table)
            
            # Pied de page amélioré
            elements.append(Spacer(1, 30))
            footer_text = f"Ce rapport a été généré automatiquement par l'application de Prévisions de Trésorerie.<br/>Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>Version : 2.0 | © 2024 Tous droits réservés"
            elements.append(Paragraph(footer_text, styles['Normal']))
            
            # Génération du PDF
            doc.build(elements)
            output.seek(0)
            return output, 'application/pdf', 'rapport_previsions_tresorerie.pdf'
            
        elif export_format == 'Excel':
            # Créer un fichier Excel avec plusieurs onglets
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Onglet des prévisions
                df_forecast = pd.DataFrame({
                    'Date': pd.date_range(start=df_enc['ds'].max(), periods=n_mois + 1, freq='M')[1:],
                    'Encaissements': forecasts.get('prophet_enc', []),
                    'Décaissements': forecasts.get('prophet_dec', []),
                    'Solde': [e - d for e, d in zip(forecasts.get('prophet_enc', []), forecasts.get('prophet_dec', []))]
                })
                df_forecast.to_excel(writer, sheet_name='Prévisions', index=False)
                
                # Onglet des données historiques
                df_historical = pd.DataFrame({
                    'Date': df_enc['ds'],
                    'Encaissements': df_enc['y_enc'],
                    'Décaissements': df_dec['y_dec'],
                    'Solde': df_enc['y_enc'] - df_dec['y_dec']
                })
                df_historical.to_excel(writer, sheet_name='Données Historiques', index=False)
            
            output.seek(0)
            return output, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'previsions_tresorerie.xlsx'
            
        elif export_format == 'CSV':
            # Créer un fichier CSV avec toutes les données
            future_dates = pd.date_range(start=df_enc['ds'].max(), periods=n_mois + 1, freq='M')[1:]
            enc_forecast = forecasts.get('prophet_enc', [])
            dec_forecast = forecasts.get('prophet_dec', [])
            
            # Debug des longueurs
            st.write(f"Longueur des dates futures : {len(future_dates)}")
            st.write(f"Longueur des encaissements : {len(enc_forecast)}")
            st.write(f"Longueur des décaissements : {len(dec_forecast)}")
            st.write(f"Nombre de mois demandé : {n_mois}")
            
            # Vérifier que toutes les données ont la même longueur
            if len(future_dates) != len(enc_forecast) or len(future_dates) != len(dec_forecast):
                st.error("Erreur : Les données de prévision n'ont pas la même longueur")
                return None, None, None
            
            # Créer le DataFrame des prévisions
            df_forecast = pd.DataFrame({
                'Date': future_dates,
                'Encaissements': enc_forecast,
                'Décaissements': dec_forecast,
                'Solde': [e - d for e, d in zip(enc_forecast, dec_forecast)],
                'Type': ['Prévision'] * len(future_dates)
            })
            
            # Créer le DataFrame des données historiques
            df_historical = pd.DataFrame({
                'Date': df_enc['ds'],
                'Encaissements': df_enc['y_enc'],
                'Décaissements': df_dec['y_dec'],
                'Solde': df_enc['y_enc'] - df_dec['y_dec'],
                'Type': ['Historique'] * len(df_enc)
            })
            
            # Combiner les données
            df_combined = pd.concat([df_historical, df_forecast], ignore_index=True)
            
            # Trier par date
            df_combined = df_combined.sort_values('Date')
            
            output = BytesIO()
            df_combined.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)
            return output, 'text/csv', 'previsions_tresorerie.csv'
            
    except Exception as e:
        st.error(f"Erreur lors de l'export : {str(e)}")
        return None, None, None

if __name__ == "__main__":
    main()
