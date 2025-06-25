"""
Module de gestion des widgets interactifs pour l'interface utilisateur
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Widgets:
    """Gestionnaire des widgets interactifs"""
    
    def __init__(self):
        self.config = {}
        self.key_counter = 0
    
    def _generate_unique_key(self, base_key: str) -> str:
        """Génère une clé unique pour chaque élément Streamlit"""
        self.key_counter += 1
        return f'{base_key}_{id(self)}_{self.key_counter}'

    def _add_section_title(self, title: str, icon: str):
        """Ajoute un titre de section avec une icône"""
        st.markdown(f'<div class="widget-section"><h3 class="widget-title">{icon} {title}</h3>', unsafe_allow_html=True)
        return st.container()

    def _add_model_prediction_section(self):
        """Configure la section Modèle et Prévision"""
        with self._add_section_title("Modèle et Prévision", "🎯"):
            # Par défaut, utiliser 'ensemble' comme type de modèle
            # mais ne pas l'afficher dans le sidebar
            self.config['model_type'] = 'ensemble'
            
            # Initialiser la session state pour n_mois s'il n'existe pas
            if 'n_mois' not in st.session_state:
                st.session_state['n_mois'] = 6
                
            # Utiliser la valeur de session_state comme valeur par défaut
            self.config['n_mois'] = st.slider(
                "Horizon de prévision (mois)",
                min_value=1, max_value=24, value=st.session_state['n_mois'], step=1,
                help="Nombre de mois à prévoir",
                key='n_mois_slider',
                on_change=self._update_n_mois
            )
            
            # Initialiser la session state pour confidence_interval s'il n'existe pas
            if 'confidence_interval' not in st.session_state:
                st.session_state['confidence_interval'] = 95
                
            # Utiliser la valeur de session_state comme valeur par défaut
            self.config['confidence_interval'] = st.slider(
                "Intervalle de confiance (%)",
                min_value=50, max_value=99, value=st.session_state['confidence_interval'], step=1,
                help="Niveau de confiance pour les prévisions",
                key='confidence_interval_slider',
                on_change=self._update_confidence_interval
            )
            st.markdown('</div>', unsafe_allow_html=True)

    def _add_evaluation_section(self):
        """Configure la section Évaluation"""
        with self._add_section_title("Évaluation", "📊"):
            # Initialiser la session state pour selection_metric s'il n'existe pas
            if 'selection_metric' not in st.session_state:
                st.session_state['selection_metric'] = 'MAE'
                
            # Utiliser la valeur de session_state comme valeur par défaut
            metric_index = ['MAE', 'RMSE', 'MAPE'].index(st.session_state['selection_metric']) if st.session_state['selection_metric'] in ['MAE', 'RMSE', 'MAPE'] else 0
            
            self.config['selection_metric'] = st.selectbox(
                "Métrique principale",
                ['MAE', 'RMSE', 'MAPE'],
                index=metric_index,
                help="Métrique utilisée pour comparer les modèles",
                key='selection_metric_selectbox',
                on_change=self._update_selection_metric
            )
            
            # Initialiser la session state pour use_cross_validation s'il n'existe pas
            if 'use_cross_validation' not in st.session_state:
                st.session_state['use_cross_validation'] = False
                
            # Utiliser la valeur de session_state comme valeur par défaut
            # et corriger le nom du paramètre pour qu'il soit cohérent avec app.py et models.py
            self.config['use_cross_validation'] = st.toggle(
                "Validation croisée",
                value=st.session_state['use_cross_validation'],
                help="Activer la validation croisée",
                key='use_cross_validation_toggle',
                on_change=self._update_use_cross_validation
            )
            
            if self.config['use_cross_validation']:
                # Initialiser la session state pour n_folds s'il n'existe pas
                if 'n_folds' not in st.session_state:
                    st.session_state['n_folds'] = 5
                    
                self.config['n_folds'] = st.slider(
                    "Nombre de folds",
                    min_value=2, max_value=10, value=st.session_state['n_folds'], step=1,
                    help="Nombre de folds pour la validation croisée",
                    key='n_folds_slider',
                    on_change=self._update_n_folds
                )
            st.markdown('</div>', unsafe_allow_html=True)

    def _add_performance_section(self):
        """Configure la section Performance"""
        with self._add_section_title("Performance", "⚡"):
            self.config['parallel'] = st.toggle(
                "Parallélisation",
                value=True,
                help="Activer le calcul parallèle",
                key=self._generate_unique_key('parallel_toggle')
            )
            
            self.config['sensitivity_analysis'] = st.toggle(
                "Analyse sensibilité",
                value=False,
                help="Analyser la sensibilité du modèle",
                key=self._generate_unique_key('sensitivity_analysis_toggle')
            )
            st.markdown('</div>', unsafe_allow_html=True)

    def _update_model_type(self):
        """Met à jour la valeur de model_type dans session_state"""
        st.session_state['model_type'] = st.session_state['model_type_selectbox']
        
    def _update_n_mois(self):
        """Met à jour la valeur de n_mois dans session_state"""
        st.session_state['n_mois'] = st.session_state['n_mois_slider']
        
    def _update_confidence_interval(self):
        """Met à jour la valeur de confidence_interval dans session_state"""
        st.session_state['confidence_interval'] = st.session_state['confidence_interval_slider']
        
    def _update_selection_metric(self):
        """Met à jour la valeur de selection_metric dans session_state"""
        st.session_state['selection_metric'] = st.session_state['selection_metric_selectbox']
        
    def _update_use_cross_validation(self):
        """Met à jour la valeur de use_cross_validation dans session_state"""
        st.session_state['use_cross_validation'] = st.session_state['use_cross_validation_toggle']
        
    def _update_n_folds(self):
        """Met à jour la valeur de n_folds dans session_state"""
        st.session_state['n_folds'] = st.session_state['n_folds_slider']

    def configure_sidebar(self):
        """Configure le panneau latéral avec tous les widgets de configuration"""
        with st.sidebar:
            # Style CSS pour la sidebar
            st.markdown('''
                <style>
                .widget-title {
                    font-size: 1.2rem;
                    font-weight: bold;
                    margin-bottom: 1rem;
                    color: #1e3a8a;
                }
                .widget-section {
                    background-color: #f8fafc;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
                </style>
                ''', unsafe_allow_html=True)
            
            # Ajout des sections
            self._add_model_prediction_section()
            self._add_evaluation_section()
            self._add_performance_section()

        
        return self.config

    def get_config(self) -> Dict:
        """Retourne la configuration actuelle"""
        return self.config
    
    def update_config(self, new_config: Dict):
        """Met à jour la configuration"""
        self.config.update(new_config)

    def create_forecast_controls(self, df: pd.DataFrame) -> Dict:
        """
        Crée les contrôles interactifs pour la prévision.

        Args:
            df (pd.DataFrame): Données historiques

        Returns:
            Dict: Configuration des contrôles de prévision
        """
        controls = {}
        
        # Sélecteur de date de début
        controls['start_date'] = st.date_input(
            "Date de début de la prévision",
            value=df['ds'].min(),
            min_value=df['ds'].min(),
            max_value=df['ds'].max(),
            key=self._generate_unique_key('start_date_input')
        )
        
        # Sélecteur de date de fin
        controls['end_date'] = st.date_input(
            "Date de fin de la prévision",
            value=df['ds'].max() + pd.DateOffset(months=self.config['n_mois']),
            min_value=controls['start_date'] + pd.DateOffset(months=1),
            max_value=df['ds'].max() + pd.DateOffset(months=24),
            key=self._generate_unique_key('end_date_input')
        )
        
        # Sélecteur de fréquence
        controls['frequency'] = st.selectbox(
            "Fréquence des données",
            ['M', 'W', 'D'],
            index=0,
            help="Choisissez la fréquence des données (M: Mensuel, W: Hebdomadaire, D: Journalier)",
            key=self._generate_unique_key('frequency_selectbox')
        )
        
        return controls

    def create_model_selector(self, available_models: List[str]) -> str:
        """
        Crée un sélecteur de modèle interactif.

        Args:
            available_models (List[str]): Liste des modèles disponibles

        Returns:
            str: Nom du modèle sélectionné
        """
        return st.selectbox(
            "Modèle de prévision",
            available_models,
            index=0,
            help="Choisissez le modèle de prévision à utiliser",
            key=self._generate_unique_key('model_selector_selectbox')
        )

    def create_metrics_selector(self, available_metrics: List[str]) -> List[str]:
        """
        Crée un sélecteur de métriques interactif.

        Args:
            available_metrics (List[str]): Liste des métriques disponibles

        Returns:
            List[str]: Liste des métriques sélectionnées
        """
        return st.multiselect(
            "Métriques à afficher",
            available_metrics,
            default=available_metrics[:3],
            help="Sélectionnez les métriques à afficher",
            key=self._generate_unique_key('metrics_selector_multiselect')
        )

if __name__ == "__main__":
    # Code de test pour la classe Widgets
    widgets = Widgets()
    config = widgets.get_config()
    print("Configuration initiale:", config)
