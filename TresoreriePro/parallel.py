"""
Module de gestion des tâches parallèles pour l'accélération des calculs
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any
import logging
import functools
import pandas as pd

class ParallelManager:
    """Gestionnaire de tâches parallèles"""
    def __init__(self, max_workers=None):
        """
        Initialise le gestionnaire avec un nombre maximum de workers
        
        Args:
            max_workers (int): Nombre maximum de processus parallèles
                Si None, utilise le nombre de CPU disponibles
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
        
    def parallel_map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        Applique une fonction à une liste d'éléments de manière parallèle
        
        Args:
            func (Callable): Fonction à appliquer
            items (List): Liste des éléments à traiter
            **kwargs: Arguments supplémentaires à passer à la fonction
            
        Returns:
            List: Liste des résultats
        """
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Créer les futures avec les arguments
            futures = [
                executor.submit(functools.partial(func, **kwargs), item)
                for item in items
            ]
            
            # Récupérer les résultats
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'exécution parallèle: {e}")
                    results.append(None)
        
        return results
    
    def parallel_forecast(self, forecast_func: Callable, 
                         dataframes: List[pd.DataFrame], 
                         n_mois: int, 
                         model_types: List[str] = None) -> Dict[str, Any]:
        """
        Effectue des prédictions parallèles sur plusieurs modèles et/ou données
        
        Args:
            forecast_func (Callable): Fonction de prévision
            dataframes (List[pd.DataFrame]): Liste des DataFrames à prévoir
            n_mois (int): Nombre de mois à prévoir
            model_types (List[str]): Types de modèles à utiliser
            
        Returns:
            Dict: Dictionnaire des prédictions par modèle et/ou données
        """
        if model_types is None:
            model_types = ['prophet', 'arima', 'ensemble']
        
        # Créer les tâches
        tasks = []
        for df in dataframes:
            for model_type in model_types:
                tasks.append((df, model_type))
        
        # Exécuter les tâches en parallèle
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(forecast_func, df, n_mois, model_type)
                for df, model_type in tasks
            ]
            
            # Collecter les résultats
            results = {}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Construire la clé unique pour le résultat
                    df_id = id(tasks[futures.index(future)][0])
                    model_type = tasks[futures.index(future)][1]
                    key = f"df_{df_id}_{model_type}"
                    results[key] = result
                except Exception as e:
                    self.logger.error(f"Erreur lors de la prévision: {e}")
                    
        return results
    
    def parallel_analyze(self, analyze_func: Callable, 
                        dataframes: List[pd.DataFrame], 
                        **kwargs) -> Dict[str, Any]:
        """
        Effectue des analyses parallèles sur plusieurs ensembles de données
        
        Args:
            analyze_func (Callable): Fonction d'analyse
            dataframes (List[pd.DataFrame]): Liste des DataFrames à analyser
            **kwargs: Arguments supplémentaires à passer à la fonction
            
        Returns:
            Dict: Dictionnaire des résultats d'analyse
        """
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Créer les tâches
            futures = [
                executor.submit(analyze_func, df, **kwargs)
                for df in dataframes
            ]
            
            # Collecter les résultats
            for future in as_completed(futures):
                try:
                    result = future.result()
                    df_id = id(dataframes[futures.index(future)])
                    results[f"df_{df_id}"] = result
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'analyse: {e}")
                    
        return results

def run_parallel(func: Callable, items: List[Any], max_workers: int = None, **kwargs) -> List[Any]:
    """
    Exécute une fonction en parallèle sur une liste d'éléments
    
    Args:
        func (Callable): Fonction à exécuter
        items (List[Any]): Liste des éléments à traiter
        max_workers (int, optional): Nombre maximum de processus parallèles
        **kwargs: Arguments supplémentaires à passer à la fonction
        
    Returns:
        List[Any]: Liste des résultats
        
    Example:
        >>> results = run_parallel(process_data, data_list, max_workers=4)
    """
    manager = ParallelManager(max_workers=max_workers)
    return manager.parallel_map(func, items, **kwargs)
