# 📊 TresoreriePro - Outil Professionnel de Prévision de Trésorerie

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

## 📝 Description
TresoreriePro est une application web interactive développée avec Streamlit, conçue pour l'analyse avancée et la prévision des flux de trésorerie. Cette solution complète intègre plusieurs algorithmes de machine learning et d'intelligence artificielle pour fournir des prévisions précises et des analyses financières détaillées.

## ✨ Fonctionnalités clés
- **Prévision multi-modèles** : Intègre plusieurs modèles (Prophet, ARIMA, LSTM, XGBoost, Random Forest)
- **Sélection automatique** du meilleur modèle basée sur les métriques de performance
- **Simulation de scénarios** : Scénarios prédéfinis et personnalisables
- **Analyse financière avancée** : Calcul de ratios clés et recommandations personnalisées
- **Tableaux de bord interactifs** : Visualisations dynamiques et paramétrables
- **Export des données** : Génération de rapports au format Excel
- **Interface intuitive** : Navigation simple et conviviale

## 🚀 Démarrage rapide

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation
1. Clonez le dépôt :
```bash
git clone https://github.com/votre-utilisateur/tresorerie-pro.git
cd tresorerie-pro
```

2. Créez et activez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancez l'application :
```bash
streamlit run app.py
```

## 🏗 Structure du projet
```
TresoreriePro/
├── app.py                  # Application principale Streamlit
├── utils.py                # Fonctions utilitaires pour le traitement des données
├── models.py               # Implémentation des modèles de prévision
├── visualizations.py       # Fonctions de visualisation des données
├── widgets.py              # Composants d'interface utilisateur personnalisés
├── validation.py           # Validation des données d'entrée
├── parallel.py             # Traitement parallèle
├── requirements.txt        # Dépendances du projet
├── .gitignore             # Fichiers à ignorer par Git
└── README.md              # Documentation du projet
```

## 📊 Format des données
L'application attend un fichier Excel contenant deux feuilles spécifiques :

1. **Feuille des flux** :
   - Colonne des dates
   - Colonnes d'encours
   - Colonnes de décaissements

2. **Feuille TGR** :
   - Détails des opérations
   - Montants et dates

## 📄 Licence
Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contribution
Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## 📧 Contact
Pour toute question ou suggestion, veuillez ouvrir une issue sur le dépôt GitHub.

## Auteur
Développé par l'équipe Outils_Prediction

## Licence
Tous droits réservés
