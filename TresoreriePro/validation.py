"""
Module de validation des données et gestion des erreurs
"""
from typing import Union, List, Dict, Any
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum
import logging

class DataValidationError(Exception):
    """Exception personnalisée pour les erreurs de validation des données"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(Exception):
    """Exception personnalisée pour les erreurs de configuration"""
    def __init__(self, message: str, config: dict = None):
        self.message = message
        self.config = config or {}
        super().__init__(self.message)

class DataSchema(BaseModel):
    """Schéma de validation pour les données de trésorerie"""
    model_config = {
        'arbitrary_types_allowed': True,  # Permet l'utilisation de types arbitraires
        'from_attributes': True  # Permet l'utilisation d'objets avec attributs
    }
    
    ds: str = Field(..., description="Date de la donnée")
    y: float = Field(..., description="Valeur numérique", ge=0)
    
    @validator('ds', pre=True)
    def validate_date(cls, v):
        """
        Valide et convertit la date en string ISO format
        
        Args:
            v: La valeur à valider
            
        Returns:
            str: La date au format ISO
            
        Raises:
            ValueError: Si la date est invalide
        """
        try:
            # Si c'est déjà une Timestamp, on la convertit en string
            if isinstance(v, pd.Timestamp):
                return v.isoformat()
            # Si c'est une string, on tente la conversion
            elif isinstance(v, str):
                return pd.to_datetime(v).isoformat()
            raise ValueError("Type de date non supporté")
        except Exception as e:
            raise ValueError(f"Format de date invalide: {str(e)}")

class ForecastConfig(BaseModel):
    """Schéma de validation pour la configuration des prévisions"""
    model_config = {
        'arbitrary_types_allowed': True,
        'from_attributes': True
    }
    
    n_mois: int = Field(default=12, gt=0, le=24, description="Nombre de mois à prévoir")
    model_type: str = Field(default='ensemble', description="Type de modèle à utiliser")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration spécifique")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """
        Valide le type de modèle
        
        Args:
            v: Le type de modèle à valider
            
        Returns:
            str: Le type de modèle validé
            
        Raises:
            ValueError: Si le modèle n'est pas supporté
        """
        allowed_models = ['prophet', 'arima', 'xgboost', 'ensemble']
        if v not in allowed_models:
            raise ValueError(f"Modèle non supporté: {v}. Modèles disponibles: {allowed_models}")
        return v

class DataValidator:
    """Validateur de données pour l'application de trésorerie"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict:
        """
        Valide un DataFrame de données de trésorerie
        
        Args:
            df (pd.DataFrame): DataFrame à valider
            
        Returns:
            Dict: Rapport de validation
            
        Raises:
            DataValidationError: Si les données ne sont pas valides
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Vérifier les colonnes requises
        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report['valid'] = False
            report['errors'].append(f"Colonnes manquantes: {missing_cols}")
            raise DataValidationError(
                "Colonnes manquantes dans le DataFrame",
                {'missing_columns': missing_cols}
            )
        
        # Vérifier le type des données
        try:
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'])
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"Erreur de conversion des types: {str(e)}")
            raise DataValidationError(
                "Erreur de type dans les données",
                {'error': str(e)}
            )
        
        # Vérifier les valeurs négatives
        if (df['y'] < 0).any():
            report['warnings'].append("Valeurs négatives détectées dans les données")
            report['info']['negative_values'] = df[df['y'] < 0].shape[0]
        
        # Vérifier les valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            report['warnings'].append("Valeurs manquantes détectées")
            report['info']['missing_values'] = missing_values.to_dict()
        
        # Vérifier la continuité temporelle
        df_sorted = df.sort_values('ds')
        if len(df_sorted) > 1:
            time_diff = df_sorted['ds'].diff().dropna()
            if not time_diff.std() < pd.Timedelta(days=1):
                report['warnings'].append("Séries temporelles non régulières")
                report['info']['time_irregularities'] = True
        
        return report
    
    @staticmethod
    def validate_config(config: Dict) -> Dict:
        """
        Valide la configuration des prévisions
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Dict: Rapport de validation
            
        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        try:
            # Créer une configuration complète avec les valeurs par défaut
            default_config = {
                'n_mois': 12,
                'model_type': 'ensemble',
                'config': {}
            }
            
            # Mettre à jour avec la configuration fournie
            final_config = {**default_config, **config}
            
            # Valider la configuration complète
            ForecastConfig(**final_config)
            
            return {
                'valid': True,
                'errors': [],
                'validated_config': final_config
            }
        except Exception as e:
            error_report = ErrorHandler.handle_error(
                e,
                {'config': config}
            )
            raise ConfigurationError(
                f"Configuration invalide: {error_report['message']}",
                {'config': config}
            )

class ErrorHandler:
    """Gestionnaire d'erreurs pour l'application"""
    
    @staticmethod
    def handle_error(error: Exception, context: Dict = None) -> Dict:
        """
        Gère une erreur et retourne un rapport détaillé
        
        Args:
            error (Exception): L'erreur à gérer
            context (Dict, optional): Contexte supplémentaire
            
        Returns:
            Dict: Rapport d'erreur
        """
        return {
            'type': type(error).__name__,
            'message': str(error),
            'context': context or {}
        }
    
    @staticmethod
    def get_suggestion(error_type: str, error_message: str) -> str:
        """
        Retourne une suggestion basée sur le type d'erreur
        
        Args:
            error_type (str): Type d'erreur
            error_message (str): Message d'erreur
            
        Returns:
            str: Suggestion pour résoudre l'erreur
        """
        suggestions = {
            'DataValidationError': "Vérifiez le format de vos données et assurez-vous qu'elles sont conformes au schéma attendu.",
            'ConfigurationError': "Vérifiez les paramètres de configuration et assurez-vous qu'ils sont valides.",
            'ValueError': "Vérifiez les valeurs fournies et assurez-vous qu'elles sont dans les plages acceptables."
        }
        return suggestions.get(error_type, "Une erreur inattendue s'est produite. Veuillez vérifier vos données et votre configuration.")

def validate_data(df: pd.DataFrame) -> Dict:
    """
    Valide les données de trésorerie
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données à valider
        
    Returns:
        Dict: Rapport de validation
        
    Raises:
        DataValidationError: Si les données ne sont pas valides
    """
    return DataValidator.validate_dataframe(df)
