# TresoreriePro - Outil Professionnel de Prévision de Trésorerie

## Description
TresoreriePro est une application Streamlit avancée pour l'analyse et la prévision des flux de trésorerie. Elle utilise plusieurs modèles de machine learning et d'intelligence artificielle pour générer des prévisions précises et offre des fonctionnalités d'analyse financière complètes.

## Fonctionnalités
- **Prévision multi-modèles** : Prophet, ARIMA, LSTM, XGBoost, Random Forest
- **Sélection automatique** du meilleur modèle basée sur les métriques de performance
- **Simulation de scénarios** prédéfinis et personnalisés
- **Analyse financière** avec calcul de ratios et recommandations
- **Visualisations interactives** des flux et prévisions
- **Export des données** au format Excel

## Structure du projet
```
TresoreriePro/
├── app.py                 # Application principale Streamlit
├── utils.py               # Fonctions utilitaires pour le traitement des données
├── models.py              # Modèles de prévision
├── visualizations.py      # Fonctions de visualisation
└── README.md              # Documentation
```

## Prérequis
- Python 3.8+
- Bibliothèques Python : streamlit, pandas, numpy, plotly, prophet, xgboost, tensorflow, scikit-learn, statsmodels

## Installation
1. Installez les dépendances requises :
```bash
pip install streamlit pandas numpy plotly prophet xgboost tensorflow scikit-learn statsmodels
```

2. Lancez l'application :
```bash
cd TresoreriePro
streamlit run app.py
```

## Utilisation
1. Chargez votre fichier Excel contenant les données de trésorerie
2. Configurez les paramètres de prévision dans la barre latérale
3. Cliquez sur "Générer Prévisions"
4. Explorez les différents onglets pour analyser les résultats
5. Exportez les prévisions au format Excel si nécessaire

## Format du fichier Excel
Le fichier Excel doit contenir deux feuilles :
- **31-12-24** : Données de flux avec les encaissements et décaissements
- **TGR 31-12-2024** : Données TGR avec les opérations détaillées

## Auteur
Développé par l'équipe Outils_Prediction

## Licence
Tous droits réservés
