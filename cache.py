"""
Module de gestion du cache pour les calculs lourds
"""
import os
import pickle
import hashlib
from pathlib import Path
from functools import wraps

class CacheManager:
    """Gestionnaire de cache pour les calculs lourds"""
    def __init__(self, cache_dir='.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, func, *args, **kwargs):
        """Génère une clé unique pour un appel de fonction"""
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        key = f"{func.__name__}_{args_str}_{kwargs_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """Retourne le chemin du fichier cache pour une clé donnée"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key):
        """Récupère une valeur du cache"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Erreur lors de la lecture du cache: {e}")
        return None
    
    def set(self, key, value):
        """Stocke une valeur dans le cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Erreur lors de l'écriture du cache: {e}")
    
    def clear(self):
        """Efface tout le cache"""
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Erreur lors de la suppression du cache: {e}")

def cached(cache_manager=None, timeout=None):
    """Décorateur pour mettre en cache les résultats d'une fonction"""
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Vérifier si une clé de cache est fournie dans les kwargs
            cache_key = kwargs.pop('cache_key', None)
            
            if cache_key is None:
                # Générer une clé basée sur les arguments de la fonction
                key = cache_manager._get_cache_key(func, *args, **kwargs)
            else:
                key = cache_key
            
            # Vérifier si le résultat est déjà en cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                print(f"Résultat récupéré du cache pour {func.__name__} avec la clé {key}")
                return cached_result
                
            # Calculer le résultat
            result = func(*args, **kwargs)
            
            # Stocker le résultat en cache
            cache_manager.set(key, result)
            print(f"Résultat mis en cache pour {func.__name__} avec la clé {key}")
            
            return result
        
        return wrapper
    
    return decorator

# Instance globale du gestionnaire de cache
_cache_manager = CacheManager()

def cache_data(data, key=None):
    """
    Stocke des données dans le cache
    
    Args:
        data: Les données à stocker
        key (str, optional): Clé personnalisée pour le cache. Si None, une clé sera générée.
        
    Returns:
        str: La clé utilisée pour stocker les données
    """
    if key is None:
        # Générer une clé basée sur les données
        key = hashlib.md5(str(data).encode()).hexdigest()
    
    _cache_manager.set(key, data)
    return key

def get_cached_data(key):
    """
    Récupère des données du cache
    
    Args:
        key (str): La clé des données à récupérer
        
    Returns:
        Les données en cache ou None si non trouvées
    """
    return _cache_manager.get(key)
