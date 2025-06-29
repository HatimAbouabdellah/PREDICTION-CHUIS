"""
Fonctions de visualisation pour l'analyse de trésorerie
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import calendar
from datetime import datetime, timedelta

class TresorerieVisualizer:
    """Classe pour la création de visualisations des données de trésorerie"""
    
    def __init__(self, config=None):
        """
        Initialise le visualiseur avec la configuration spécifiée.
        
        Args:
            config (dict): Configuration des options d'affichage
        """
        self.config = config or {}
        self.color_palette = {
            'encaissement': 'rgba(16, 152, 84, 1)',       # Vert plus vif
            'decaissement': 'rgba(220, 57, 18, 1)',       # Rouge plus distinct
            'solde': 'rgba(102, 51, 153, 0.85)',          # Violet plus visible
            'optimiste': 'rgba(51, 187, 102, 0.9)',       # Vert clair amélioré
            'pessimiste': 'rgba(230, 85, 13, 0.9)',      # Orange-rouge
            'neutre': 'rgba(51, 102, 153, 0.85)',        # Bleu-gris
            'anomalie': 'rgba(255, 153, 0, 1)',          # Orange vif
            'saisonnier': 'rgba(0, 153, 198, 0.9)',      # Bleu ciel amélioré
            'tendance': 'rgba(153, 51, 102, 0.9)',       # Magenta amélioré
            'residuel': 'rgba(119, 119, 119, 0.7)',      # Gris amélioré
            'enc_prev': 'rgba(16, 152, 84, 0.7)',        # Vert transparent pour prévisions
            'dec_prev': 'rgba(220, 57, 18, 0.7)',        # Rouge transparent pour prévisions
            'grid': 'rgba(204, 204, 204, 0.3)',          # Gris clair pour grille
            'background': 'rgba(250, 250, 250, 1)'        # Fond légèrement gris
        }
        
    def create_seasonal_analysis_chart(self, seasonal_patterns, title="Analyse Saisonnière"):
        """
        Crée un graphique d'analyse saisonnière des données.
        
        Args:
            seasonal_patterns (dict): Dictionnaire des composantes saisonnières
            title (str): Titre du graphique
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Vérifier si nous avons des données saisonnières
        if not seasonal_patterns or not seasonal_patterns.get('has_seasonality', False):
            # Créer un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Pas de saisonnalité significative détectée dans les données",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=title,
                height=400
            )
            return fig
        
        # Créer un graphique avec sous-graphiques pour les composantes
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=('Tendance', 'Saisonnalité', 'Résidus'),
                           shared_xaxes=True,
                           vertical_spacing=0.1)
        
        # Ajouter les composantes
        # Tendance
        fig.add_trace(
            go.Scatter(
                y=seasonal_patterns['trend'],
                mode='lines',
                name='Tendance',
                line=dict(color=self.color_palette['tendance'], width=2)
            ),
            row=1, col=1
        )
        
        # Saisonnalité
        fig.add_trace(
            go.Scatter(
                y=seasonal_patterns['seasonal'],
                mode='lines',
                name='Saisonnalité',
                line=dict(color=self.color_palette['saisonnier'], width=2)
            ),
            row=2, col=1
        )
        
        # Résidus
        fig.add_trace(
            go.Scatter(
                y=seasonal_patterns['resid'],
                mode='lines',
                name='Résidus',
                line=dict(color=self.color_palette['residuel'], width=1)
            ),
            row=3, col=1
        )
        
        # Ajouter une ligne horizontale à zéro pour les résidus et la saisonnalité
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="gray"),
            x0=0, x1=1, y0=0, y1=0,
            xref="paper", yref="y2"
        )
        
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="gray"),
            x0=0, x1=1, y0=0, y1=0,
            xref="paper", yref="y3"
        )
        
        # Ajouter des annotations pour les statistiques
        period = seasonal_patterns.get('dominant_period', 'Non détectée')
        strength = seasonal_patterns.get('seasonal_strength', 0)
        
        fig.add_annotation(
            text=f"Période dominante: {period} mois<br>Force de la saisonnalité: {strength:.2f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        )
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_monthly_pattern_chart(self, df, value_column, title="Patterns Mensuels"):
        """
        Crée un graphique des patterns mensuels pour visualiser les tendances saisonnières.
        
        Args:
            df (DataFrame): Données avec une colonne 'ds' (date) et une colonne de valeurs
            value_column (str): Nom de la colonne contenant les valeurs
            title (str): Titre du graphique
            
        Returns:
            Figure: Objet Figure Plotly
        """
        if df is None or len(df) < 12:
            # Pas assez de données pour une analyse mensuelle
            fig = go.Figure()
            fig.add_annotation(
                text="Pas assez de données pour analyser les patterns mensuels (minimum 12 mois requis)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=title,
                height=400
            )
            return fig
        
        # Ajouter le mois et l'année aux données
        df_copy = df.copy()
        df_copy['month'] = df_copy['ds'].dt.month
        df_copy['year'] = df_copy['ds'].dt.year
        df_copy['month_name'] = df_copy['ds'].dt.month.apply(lambda x: calendar.month_name[x])
        
        # Calculer les moyennes mensuelles
        monthly_avg = df_copy.groupby('month')[value_column].mean().reset_index()
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: calendar.month_name[x])
        
        # Trier par mois
        monthly_avg = monthly_avg.sort_values('month')
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les barres pour les moyennes mensuelles
        fig.add_trace(go.Bar(
            x=monthly_avg['month_name'],
            y=monthly_avg[value_column],
            name='Moyenne Mensuelle',
            marker_color=self.color_palette['saisonnier']
        ))
        
        # Ajouter une ligne pour la tendance
        fig.add_trace(go.Scatter(
            x=monthly_avg['month_name'],
            y=monthly_avg[value_column],
            mode='lines+markers',
            name='Tendance',
            line=dict(color=self.color_palette['tendance'], width=2)
        ))
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_all_models_chart(self, df_enc, df_dec, forecasts, best_model, future_dates):
        """
        Crée un graphique affichant tous les modèles de prévision activés.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            best_model (str): Nom du meilleur modèle
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Création du graphique principal
        fig_all_models = go.Figure()
        
        # Options d'affichage
        show_historical = self.config.get('show_historical', True)
        show_predictions = self.config.get('show_predictions', True)
        
        # Données historiques
        if show_historical:
            fig_all_models.add_trace(go.Scatter(
                x=df_enc['ds'], 
                y=df_enc['y_enc'], 
                mode='lines+markers', 
                name='Encaissements Historiques',
                line=dict(color='rgba(0, 128, 0, 0.8)', width=3),
                marker=dict(size=6)
            ))
            fig_all_models.add_trace(go.Scatter(
                x=df_dec['ds'], 
                y=df_dec['y_dec'], 
                mode='lines+markers', 
                name='Décaissements Historiques',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
                marker=dict(size=6)
            ))
        
        # Palette de couleurs pour les modèles
        enc_colors = ['rgba(0, 128, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(128, 0, 128, 0.5)', 
                      'rgba(0, 128, 128, 0.5)', 'rgba(128, 128, 0, 0.5)', 'rgba(70, 130, 180, 0.5)']
        dec_colors = ['rgba(255, 0, 0, 0.5)', 'rgba(255, 165, 0, 0.5)', 'rgba(255, 69, 0, 0.5)', 
                      'rgba(178, 34, 34, 0.5)', 'rgba(220, 20, 60, 0.5)', 'rgba(139, 0, 0, 0.5)']
        
        # Prévisions pour tous les modèles activés
        if show_predictions:
            # Filtrer les modèles d'encaissement
            enc_models = [m for m in forecasts.keys() if 'enc' in m]
            
            # Ajouter les prévisions de chaque modèle d'encaissement
            for i, model in enumerate(enc_models):
                # Style spécial pour le meilleur modèle
                if model == best_model:
                    line_style = dict(color=enc_colors[i % len(enc_colors)], width=4, dash='solid')
                    marker_style = dict(size=8, symbol='star')
                    name = f'{model} (Meilleur modèle)'
                else:
                    line_style = dict(color=enc_colors[i % len(enc_colors)], width=2, dash='dash')
                    marker_style = dict(size=6)
                    name = f'{model}'
                
                fig_all_models.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasts[model], 
                    mode='lines+markers', 
                    name=name,
                    line=line_style,
                    marker=marker_style
                ))
            
            # Filtrer les modèles de décaissement
            dec_models = [m for m in forecasts.keys() if 'dec' in m]
            
            # Ajouter les prévisions de chaque modèle de décaissement
            for i, model in enumerate(dec_models):
                # Style spécial pour le meilleur modèle
                best_dec_model = best_model.replace('enc', 'dec') if best_model else ''
                if model == best_dec_model:
                    line_style = dict(color=dec_colors[i % len(dec_colors)], width=4, dash='solid')
                    marker_style = dict(size=8, symbol='star')
                    name = f'{model} (Meilleur modèle)'
                else:
                    line_style = dict(color=dec_colors[i % len(dec_colors)], width=2, dash='dash')
                    marker_style = dict(size=6)
                    name = f'{model}'
                
                fig_all_models.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasts[model], 
                    mode='lines+markers', 
                    name=name,
                    line=line_style,
                    marker=marker_style
                ))
        
        # Amélioration du design
        fig_all_models.update_layout(
            title=dict(
                text="Comparaison de Tous les Modèles de Prévision",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig_all_models
    
    def create_flux_chart(self, df_enc, df_dec, forecasts, best_model, future_dates):
        """
        Crée un graphique des flux de trésorerie historiques et prévisionnels.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            best_model (str): Nom du meilleur modèle
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Création du graphique principal avec un fond personnalisé
        fig_flux = go.Figure()
        
        # Options d'affichage
        show_historical = self.config.get('show_historical', True)
        show_predictions = self.config.get('show_predictions', True)
        show_grid = self.config.get('show_grid', True)
        
        # Style du titre avec un break après
        fig_flux.update_layout(
            title={
                'text': "<br>Analyse des Flux de Trésorerie et Prévisions",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='#333', family='Arial'),
                'pad': dict(t=50)
            }
        )
        
        # Ajouter une ligne de référence à zéro avec un style plus subtil
        fig_flux.add_shape(
            type="line",
            x0=min(df_enc['ds'].min(), df_dec['ds'].min()),
            x1=max(future_dates[-1] if len(future_dates) > 0 else df_enc['ds'].max(), 
                   df_enc['ds'].max(), df_dec['ds'].max()),
            y0=0,
            y1=0,
            line=dict(color="rgba(128, 128, 128, 0.2)", width=1, dash="dot")
        )
        
        # Configuration générale du graphique
        fig_flux.update_layout(
            height=600,
            margin=dict(l=70, r=40, t=120, b=60),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Style des axes
        fig_flux.update_xaxes(
            title_text="Date",
            title_font=dict(size=14, color='#666'),
            tickfont=dict(size=12, color='#666'),
            gridcolor='rgba(220,220,220,0.3)',
            showgrid=show_grid,
            zeroline=False,
            tickformat='%b %Y'
        )
        
        fig_flux.update_yaxes(
            title_text="Montant (DH)",
            title_font=dict(size=14, color='#666'),
            tickfont=dict(size=12, color='#666'),
            gridcolor='rgba(220,220,220,0.3)',
            showgrid=show_grid,
            zeroline=False,
            tickformat=".2s"
        )
        
        # Style de la légende
        fig_flux.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color='#666'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(220,220,220,0.3)',
                borderwidth=1
            )
        )
        
        # Données historiques avec des lignes plus épaisses et des marqueurs plus visibles
        if show_historical:
            # Encaissements historiques
            fig_flux.add_trace(go.Scatter(
                x=df_enc['ds'], 
                y=df_enc['y_enc'], 
                mode='lines+markers', 
                name='Encaissements Historiques',
                line=dict(color=self.color_palette['encaissement'], width=2.5),
                marker=dict(size=7, line=dict(width=1, color='white')),
                hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Montant</b>: %{y:,.2f} DH<extra></extra>'
            ))
            
            # Décaissements historiques
            fig_flux.add_trace(go.Scatter(
                x=df_dec['ds'], 
                y=df_dec['y_dec'], 
                mode='lines+markers', 
                name='Décaissements Historiques',
                line=dict(color=self.color_palette['decaissement'], width=2.5),
                marker=dict(size=7, line=dict(width=1, color='white')),
                hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Montant</b>: %{y:,.2f} DH<extra></extra>'
            ))
        
        # Prévisions avec un style distinct et des marqueurs plus visibles
        if show_predictions and best_model in forecasts:
            # Prévisions d'encaissements
            fig_flux.add_trace(go.Scatter(
                x=future_dates, 
                y=forecasts[best_model], 
                mode='lines+markers', 
                name='Prévision Encaissements',
                line=dict(color=self.color_palette['enc_prev'], width=3, dash='dot'),
                marker=dict(size=9, symbol='diamond', line=dict(width=1, color=self.color_palette['encaissement'])),
                hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Prévision</b>: %{y:,.2f} DH<extra></extra>'
            ))
        
            # Trouver le meilleur modèle pour les décaissements
            best_dec_model = best_model.replace('enc', 'dec')
            if best_dec_model in forecasts:
                # Prévisions de décaissements
                fig_flux.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasts[best_dec_model], 
                    mode='lines+markers', 
                    name='Prévision Décaissements',
                    line=dict(color=self.color_palette['dec_prev'], width=3, dash='dot'),
                    marker=dict(size=9, symbol='diamond', line=dict(width=1, color=self.color_palette['decaissement'])),
                    hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Prévision</b>: %{y:,.2f} DH<extra></extra>'
                ))
                
                # Solde prévisionnel avec un remplissage plus esthétique
                if best_model in forecasts and best_dec_model in forecasts:
                    solde_previsionnel = forecasts[best_model] - forecasts[best_dec_model]
                    fig_flux.add_trace(go.Scatter(
                        x=future_dates, 
                        y=solde_previsionnel, 
                        mode='lines', 
                        name='Solde Prévisionnel',
                        line=dict(color=self.color_palette['solde'], width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102, 51, 153, 0.2)',
                        hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Solde</b>: %{y:,.2f} DH<extra></extra>'
                    ))
        
        # Amélioration du design global
        fig_flux.update_layout(
            title=dict(
                text="Analyse des Flux de Trésorerie et Prévisions",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        # Ajouter des annotations pour les périodes (historique vs prévision)
        if show_historical and show_predictions and len(future_dates) > 0:
            separation_date = future_dates[0]
            
            # Ligne de séparation entre historique et prévision
            fig_flux.add_shape(
                type="line",
                x0=separation_date,
                x1=separation_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
            )
            
            # Annotations pour les périodes
            fig_flux.add_annotation(
                x=df_enc['ds'].min() + (separation_date - df_enc['ds'].min())/2,
                y=1.05,
                yref="paper",
                text="Données Historiques",
                showarrow=False,
                font=dict(size=12, color="#555")
            )
            
            fig_flux.add_annotation(
                x=separation_date + (future_dates[-1] - separation_date)/2,
                y=1.05,
                yref="paper",
                text="Prévisions",
                showarrow=False,
                font=dict(size=12, color="#555")
            )
        
        return fig_flux
    
    def create_anomaly_detection_chart(self, anomalies_data, title="Détection d'Anomalies"):
        """
        Crée un graphique pour visualiser les anomalies détectées dans les données.
        
        Args:
            anomalies_data (dict): Dictionnaire contenant les résultats de la détection d'anomalies
            title (str): Titre du graphique
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Vérifier si nous avons des données d'anomalies
        if not anomalies_data or not anomalies_data.get('all_data') is not None:
            # Créer un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Pas de données disponibles pour la détection d'anomalies",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=title,
                height=400
            )
            return fig
        
        # Récupérer les données
        df = anomalies_data['all_data']
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter toutes les données
        if 'ds' in df.columns:
            x_values = df['ds']
        else:
            x_values = np.arange(len(df))
            
        # Déterminer la colonne de valeur (y_enc, y_dec ou autre)
        value_cols = [col for col in df.columns if col.startswith('y_') or col == 'y']
        if value_cols:
            value_col = value_cols[0]
        else:
            value_col = df.columns[1]  # Prendre la deuxième colonne par défaut
        
        # Tracer toutes les données
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[value_col],
            mode='lines+markers',
            name='Données',
            line=dict(color='rgba(100, 100, 100, 0.8)', width=1),
            marker=dict(size=6)
        ))
        
        # Ajouter les anomalies si détectées
        if anomalies_data.get('anomalies_detected', False) and 'anomaly_data' in anomalies_data:
            anomaly_df = anomalies_data['anomaly_data']
            
            if 'ds' in anomaly_df.columns:
                anomaly_x = anomaly_df['ds']
            else:
                anomaly_x = anomaly_df.index
                
            fig.add_trace(go.Scatter(
                x=anomaly_x,
                y=anomaly_df[value_col],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=self.color_palette['anomalie'],
                    size=12,
                    symbol='circle-open',
                    line=dict(width=2, color=self.color_palette['anomalie'])
                )
            ))
            
            # Ajouter des annotations pour les anomalies
            for i, row in anomaly_df.iterrows():
                if 'ds' in anomaly_df.columns:
                    x_val = row['ds']
                else:
                    x_val = i
                    
                y_val = row[value_col]
                score = row.get('anomaly_score', 0)
                
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=f"Score: {score:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.color_palette['anomalie'],
                    ax=0,
                    ay=-40
                )
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        # Ajouter des statistiques sur les anomalies
        if anomalies_data.get('anomalies_detected', False):
            anomaly_count = anomalies_data.get('anomaly_count', 0)
            anomaly_percent = anomalies_data.get('anomaly_percent', 0)
            
            fig.add_annotation(
                text=f"Anomalies détectées: {anomaly_count} ({anomaly_percent:.1f}% des données)",
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=12)
            )
        
        return fig
    
    def create_comparative_analysis_chart(self, df, value_column, periods=None, title="Analyse Comparative"):
        """
        Crée un graphique d'analyse comparative entre différentes périodes.
        
        Args:
            df (DataFrame): Données avec une colonne 'ds' (date) et une colonne de valeurs
            value_column (str): Nom de la colonne contenant les valeurs
            periods (list): Liste des périodes à comparer (années ou trimestres)
            title (str): Titre du graphique
            
        Returns:
            Figure: Objet Figure Plotly
        """
        if df is None or len(df) < 12:
            # Pas assez de données pour une analyse comparative
            fig = go.Figure()
            fig.add_annotation(
                text="Pas assez de données pour une analyse comparative (minimum 12 mois requis)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=title,
                height=400
            )
            return fig
        
        # Ajouter l'année, le trimestre et le mois aux données
        df_copy = df.copy()
        df_copy['year'] = df_copy['ds'].dt.year
        df_copy['quarter'] = df_copy['ds'].dt.quarter
        df_copy['month'] = df_copy['ds'].dt.month
        df_copy['month_name'] = df_copy['ds'].dt.month.apply(lambda x: calendar.month_abbr[x])
        
        # Déterminer le type de période (année ou trimestre)
        unique_years = df_copy['year'].unique()
        
        if periods is None:
            if len(unique_years) >= 2:
                # Comparer les deux dernières années par défaut
                periods = sorted(unique_years)[-2:]
                period_type = 'year'
            else:
                # Comparer les trimestres de la dernière année
                periods = df_copy['quarter'].unique()
                period_type = 'quarter'
        else:
            # Déterminer le type de période en fonction du format des périodes fournies
            if all(isinstance(p, int) and p > 1900 for p in periods):
                period_type = 'year'
            else:
                period_type = 'quarter'
        
        # Créer le graphique
        fig = go.Figure()
        
        # Couleurs pour les différentes périodes
        colors = px.colors.qualitative.Plotly
        
        # Tracer les données pour chaque période
        for i, period in enumerate(periods):
            if period_type == 'year':
                # Filtrer par année
                period_data = df_copy[df_copy['year'] == period]
                period_label = f"Année {period}"
            else:
                # Filtrer par trimestre
                period_data = df_copy[df_copy['quarter'] == period]
                period_label = f"Trimestre {period}"
            
            if len(period_data) > 0:
                # Agréger par mois pour une comparaison cohérente
                monthly_data = period_data.groupby('month')[value_column].mean().reset_index()
                monthly_data['month_name'] = monthly_data['month'].apply(lambda x: calendar.month_abbr[x])
                monthly_data = monthly_data.sort_values('month')
                
                fig.add_trace(go.Scatter(
                    x=monthly_data['month_name'],
                    y=monthly_data[value_column],
                    mode='lines+markers',
                    name=period_label,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8)
                ))
        
        # Calculer et afficher la variation entre les périodes
        if len(periods) >= 2 and period_type == 'year':
            # Calculer la variation d'une année à l'autre
            year1_data = df_copy[df_copy['year'] == periods[0]].groupby('month')[value_column].mean()
            year2_data = df_copy[df_copy['year'] == periods[1]].groupby('month')[value_column].mean()
            
            # Calculer la variation en pourcentage
            if not year1_data.empty and not year2_data.empty:
                common_months = set(year1_data.index).intersection(set(year2_data.index))
                if common_months:
                    year1_total = year1_data.loc[list(common_months)].sum()
                    year2_total = year2_data.loc[list(common_months)].sum()
                    
                    if year1_total > 0:
                        variation_percent = ((year2_total - year1_total) / year1_total) * 100
                        
                        fig.add_annotation(
                            text=f"Variation {periods[0]} à {periods[1]}: {variation_percent:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98, showarrow=False,
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=12)
                        )
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_year_over_year_chart(self, df, value_column, title="Évolution Année par Année"):
        """
        Crée un graphique d'évolution année par année.
        
        Args:
            df (DataFrame): Données avec une colonne 'ds' (date) et une colonne de valeurs
            value_column (str): Nom de la colonne contenant les valeurs
            title (str): Titre du graphique
            
        Returns:
            Figure: Objet Figure Plotly
        """
        if df is None or len(df) < 12:
            # Pas assez de données pour une analyse année par année
            fig = go.Figure()
            fig.add_annotation(
                text="Pas assez de données pour une analyse année par année (minimum 12 mois requis)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#555")
            )
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=22, color='#333')
                },
                height=500,
                paper_bgcolor='white',
                plot_bgcolor=self.color_palette['background']
            )
            return fig
        
        # Options d'affichage
        show_grid = self.config.get('show_grid', True)
        
        # Ajouter l'année aux données
        df_copy = df.copy()
        df_copy['year'] = df_copy['ds'].dt.year
        
        # Calculer les totaux annuels
        yearly_totals = df_copy.groupby('year')[value_column].sum().reset_index()
        
        # Calculer les variations d'une année à l'autre
        yearly_totals['previous_year'] = yearly_totals['year'] - 1
        yearly_totals = yearly_totals.merge(
            yearly_totals[['year', value_column]].rename(columns={value_column: 'previous_value', 'year': 'previous_year'}),
            on='previous_year', how='left'
        )
        yearly_totals['variation'] = yearly_totals[value_column] - yearly_totals['previous_value']
        yearly_totals['variation_percent'] = (yearly_totals['variation'] / yearly_totals['previous_value']) * 100
        
        # Créer le graphique
        fig = go.Figure()
        
        # Définir un gradient de couleurs pour les barres (du plus clair au plus foncé)
        num_years = len(yearly_totals)
        if num_years <= 3:
            bar_colors = px.colors.sequential.Blues[-num_years:]
        elif num_years <= 5:
            bar_colors = px.colors.sequential.Viridis[-num_years:]
        else:
            bar_colors = px.colors.sequential.Plasma[-num_years:]
        
        # Ajouter les barres pour les totaux annuels avec des couleurs graduées
        fig.add_trace(go.Bar(
            x=yearly_totals['year'].astype(str),
            y=yearly_totals[value_column],
            name='Total Annuel',
            marker=dict(
                color=bar_colors,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Année</b>: %{x}<br><b>Total</b>: %{y:,.2f} DH<extra></extra>'
        ))
        
        # Ajouter une ligne pour la tendance avec un style amélioré
        fig.add_trace(go.Scatter(
            x=yearly_totals['year'].astype(str),
            y=yearly_totals[value_column],
            mode='lines+markers',
            name='Tendance',
            line=dict(color=self.color_palette['tendance'], width=3),
            marker=dict(size=10, line=dict(width=2, color='white')),
            hovertemplate='<b>Année</b>: %{x}<br><b>Total</b>: %{y:,.2f} DH<extra></extra>'
        ))
        
        # Ajouter les variations en pourcentage comme annotations avec un style amélioré
        for i, row in yearly_totals.iterrows():
            if pd.notna(row['variation_percent']):
                # Couleurs plus vives et contrastées
                color = 'rgba(16, 152, 84, 1)' if row['variation_percent'] >= 0 else 'rgba(220, 57, 18, 1)'
                symbol = '▲' if row['variation_percent'] >= 0 else '▼'
                
                # Ajouter une annotation avec un fond pour une meilleure lisibilité
                fig.add_annotation(
                    x=str(row['year']),
                    y=row[value_column],
                    text=f"{symbol} {abs(row['variation_percent']):.1f}%",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=13, color='white', family='Arial'),
                    bgcolor=color,
                    bordercolor='rgba(255,255,255,0.8)',
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.9
                )
        
        # Amélioration du design global
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        # Ajouter une annotation explicative
        fig.add_annotation(
            text="Les pourcentages indiquent la variation par rapport à l'année précédente",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="#555"),
            align="center"
        )
        
        return fig
    
    def create_forecast_chart(self, df_enc, df_dec, forecasts, best_model, show_historical=True, show_confidence=True):
        """
        Crée un graphique de prévision combinant les données historiques et les prédictions
        
        Args:
            df_enc (pd.DataFrame): Données historiques d'encaissements
            df_dec (pd.DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire contenant les prédictions pour chaque modèle
            best_model (str): Nom du meilleur modèle
            show_historical (bool): Si True, affiche les données historiques
            show_confidence (bool): Si True, affiche les intervalles de confiance
            
        Returns:
            plotly.graph_objects.Figure: Graphique Plotly
        """
        # Préparer les données pour le graphique
        future_dates = pd.date_range(
            start=df_enc['ds'].max(),
            periods=len(forecasts[next(iter(forecasts))]),
            freq='M'
        )
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les données historiques si demandé
        if show_historical:
            fig.add_trace(go.Scatter(
                x=df_enc['ds'],
                y=df_enc['y'],
                mode='lines+markers',
                name='Données Historiques',
                line=dict(color='#10b981', width=2),
                marker=dict(size=8, color='#10b981'),
                hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Valeur</b>: %{y:,.2f} DH<extra></extra>'
            ))
        
        # Ajouter les prédictions pour chaque modèle
        for model_name, model_forecast in forecasts.items():
            line_style = 'solid' if model_name == best_model else 'dot'
            line_width = 3 if model_name == best_model else 2
            line_color = '#10b981' if model_name == best_model else '#6b7280'
            
            # Ajouter la ligne de prévision
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=model_forecast,
                mode='lines',
                name=f'{model_name} (Meilleur)' if model_name == best_model else model_name,
                line=dict(color=line_color, width=line_width, dash=line_style),
                hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Prévision</b>: %{y:,.2f} DH<extra></extra>'
            ))
            
            # Ajouter l'intervalle de confiance si demandé
            if show_confidence and model_name == best_model:
                # Calculer l'intervalle de confiance (ex: ±10%)
                lower_bound = model_forecast * 0.9
                upper_bound = model_forecast * 1.1
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(16, 185, 129, 0.1)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Ajouter les séparateurs visuels
        separation_date = df_enc['ds'].max()
        fig.add_vline(
            x=separation_date,
            line=dict(color='#e5e7eb', width=2, dash='dash'),
            annotation_text="Prévision",
            annotation_position="top right"
        )
        
        # Ajouter les annotations
        fig.add_annotation(
            x=df_enc['ds'].min() + (separation_date - df_enc['ds'].min())/2,
            y=1.05,
            yref="paper",
            text="Données Historiques",
            showarrow=False,
            font=dict(size=12, color="#555")
        )
        
        fig.add_annotation(
            x=separation_date + (future_dates[-1] - separation_date)/2,
            y=1.05,
            yref="paper",
            text="Prévisions",
            showarrow=False,
            font=dict(size=12, color="#555")
        )
        
        # Mettre en forme le graphique
        fig.update_layout(
            title=dict(
                text="Prévisions des Flux de Trésorerie",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_model_comparison_chart(self, df_enc, df_dec, forecasts, best_model, future_dates):
        """
        Crée des graphiques de comparaison des différents modèles de prévision.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            best_model (str): Nom du meilleur modèle
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            tuple: (fig_enc, fig_dec, fig_ecarts) - Figures Plotly
        """
        # Création du graphique pour les encaissements
        fig_enc = go.Figure()
        
        # Données historiques
        fig_enc.add_trace(go.Scatter(
            x=df_enc['ds'], 
            y=df_enc['y_enc'], 
            mode='lines', 
            name='Données Historiques',
            line=dict(color='black', width=2),
            opacity=0.7
        ))
        
        # Palette de couleurs pour les modèles
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Filtrer les modèles d'encaissement
        enc_models = [m for m in forecasts.keys() if 'enc' in m]
        
        # Ajouter les prévisions de chaque modèle
        i = 0
        for model in enc_models:
            fig_enc.add_trace(go.Scatter(
                x=future_dates, 
                y=forecasts[model], 
                mode='lines', 
                name=model,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
            i += 1
        
        # Mise en évidence du meilleur modèle si disponible
        if best_model in forecasts:
            fig_enc.add_trace(go.Scatter(
                x=future_dates, 
                y=forecasts[best_model], 
                mode='lines+markers', 
                name=f'Meilleur Modèle: {best_model}',
                line=dict(color='green', width=4),
                marker=dict(size=8, symbol='star')
            ))
        
        # Amélioration du design
        fig_enc.update_layout(
            title=dict(
                text="Comparaison des Modèles de Prévision d'Encaissements",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        # Création du graphique pour les décaissements
        fig_dec = go.Figure()
        
        # Données historiques
        fig_dec.add_trace(go.Scatter(
            x=df_dec['ds'], 
            y=df_dec['y_dec'], 
            mode='lines', 
            name='Données Historiques',
            line=dict(color='black', width=2),
            opacity=0.7
        ))
        
        # Ajouter les prévisions de chaque modèle
        i = 0
        for model, forecast in forecasts.items():
            if 'dec' in model:
                fig_dec.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecast, 
                    mode='lines', 
                    name=model,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
                i += 1
        
        # Mise en évidence du meilleur modèle pour les décaissements
        best_dec_model = best_model.replace('enc', 'dec')
        if best_dec_model in forecasts:
            fig_dec.add_trace(go.Scatter(
                x=future_dates, 
                y=forecasts[best_dec_model], 
                mode='lines+markers', 
                name=f'Meilleur Modèle: {best_dec_model}',
                line=dict(color='red', width=4),
                marker=dict(size=8, symbol='star')
            ))
        
        # Amélioration du design
        fig_dec.update_layout(
            title=dict(
                text="Comparaison des Modèles de Prévision de Décaissements",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        # Calcul des écarts moyens par rapport au meilleur modèle
        ecarts = {}
        for model, forecast in forecasts.items():
            if 'enc' in model and model != best_model:
                ecart = np.mean(np.abs(forecast - forecasts[best_model])) / np.mean(forecasts[best_model]) * 100
                ecarts[model] = ecart
        
        # Affichage des écarts
        if ecarts:
            ecarts_df = pd.DataFrame.from_dict(ecarts, orient='index', columns=['Écart Moyen (%)'])
            ecarts_df = ecarts_df.sort_values('Écart Moyen (%)')
            
            # Création d'un graphique à barres pour les écarts
            fig_ecarts = px.bar(
                ecarts_df, 
                y=ecarts_df.index, 
                x='Écart Moyen (%)',
                orientation='h',
                title="Écarts Moyens par Rapport au Meilleur Modèle",
                labels={'index': 'Modèle'},
                color='Écart Moyen (%)',
                color_continuous_scale='Reds'
            )
        else:
            fig_ecarts = go.Figure()
            fig_ecarts.update_layout(
                title="Aucun écart à afficher",
                xaxis_title="Écart Moyen (%)",
                yaxis_title="Modèle"
            )
        
        return fig_enc, fig_dec, fig_ecarts
    
    def create_confidence_interval_chart(self, df_enc, df_dec, forecasts, scenarios, future_dates, confidence_levels=None):
        """
        Crée un graphique avec des intervalles de confiance personnalisables.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            scenarios (dict): Dictionnaire des scénarios avec intervalles de confiance
            future_dates (array): Dates futures pour les prévisions
            confidence_levels (list): Liste des niveaux de confiance à afficher
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Créer le graphique principal
        fig = go.Figure()
        
        # Déterminer les niveaux de confiance disponibles
        if confidence_levels is None:
            # Chercher les scénarios avec des niveaux de confiance
            available_levels = []
            for scenario_name in scenarios.keys():
                if '_' in scenario_name:
                    parts = scenario_name.split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        level = int(parts[1])
                        if level not in available_levels:
                            available_levels.append(level)
            
            confidence_levels = sorted(available_levels)
        
        # Afficher les données historiques
        fig.add_trace(go.Scatter(
            x=df_enc['ds'], 
            y=df_enc['y_enc'], 
            mode='lines+markers', 
            name='Encaissements Historiques',
            line=dict(color=self.color_palette['encaissement'], width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_dec['ds'], 
            y=df_dec['y_dec'], 
            mode='lines+markers', 
            name='Décaissements Historiques',
            line=dict(color=self.color_palette['decaissement'], width=2),
            marker=dict(size=6)
        ))
        
        # Couleurs pour les différents niveaux de confiance
        colors = px.colors.sequential.Viridis
        
        # Afficher les prévisions avec intervalles de confiance
        if 'neutre' in scenarios:
            # Prévisions de base
            fig.add_trace(go.Scatter(
                x=future_dates, 
                y=scenarios['neutre']['encaissement'], 
                mode='lines', 
                name='Encaissements Prévus',
                line=dict(color=self.color_palette['encaissement'], width=3, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates, 
                y=scenarios['neutre']['decaissement'], 
                mode='lines', 
                name='Décaissements Prévus',
                line=dict(color=self.color_palette['decaissement'], width=3, dash='dash')
            ))
        
        # Ajouter les intervalles de confiance
        for i, level in enumerate(confidence_levels):
            level_str = str(level)
            
            # Scénario optimiste pour ce niveau de confiance
            opt_key = f'optimiste_{level_str}'
            pess_key = f'pessimiste_{level_str}'
            
            if opt_key in scenarios and pess_key in scenarios:
                # Intervalle de confiance pour les encaissements
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=scenarios[opt_key]['encaissement'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=scenarios[pess_key]['encaissement'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=f'rgba(0, 128, 0, {0.1 + 0.1 * i})',
                    name=f'IC Encaissements {level}%'
                ))
                
                # Intervalle de confiance pour les décaissements
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=scenarios[opt_key]['decaissement'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=scenarios[pess_key]['decaissement'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=f'rgba(255, 0, 0, {0.1 + 0.1 * i})',
                    name=f'IC Décaissements {level}%'
                ))
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text="Prévisions avec Intervalles de Confiance Personnalisables",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_monte_carlo_chart(self, df_enc, df_dec, monte_carlo_results, future_dates):
        """
        Crée un graphique pour visualiser les résultats de la simulation Monte Carlo.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            monte_carlo_results (dict): Résultats de la simulation Monte Carlo
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            Figure: Objet Figure Plotly
        """
        if not monte_carlo_results:
            # Créer un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Pas de résultats de simulation Monte Carlo disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Simulation Monte Carlo",
                height=400
            )
            return fig
        
        # Créer un graphique avec sous-graphiques
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Prévisions avec Intervalles de Confiance', 'Probabilité de Solde Négatif'),
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])
        
        # Afficher les données historiques
        fig.add_trace(
            go.Scatter(
                x=df_enc['ds'], 
                y=df_enc['y_enc'] - df_dec['y_dec'], 
                mode='lines+markers', 
                name='Solde Historique',
                line=dict(color=self.color_palette['solde'], width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Afficher la moyenne des simulations
        fig.add_trace(
            go.Scatter(
                x=future_dates, 
                y=monte_carlo_results['solde_mean'], 
                mode='lines', 
                name='Solde Prévisionnel Moyen',
                line=dict(color=self.color_palette['solde'], width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        # Afficher l'intervalle de confiance à 95%
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=monte_carlo_results['solde_upper_95'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=monte_carlo_results['solde_lower_95'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(75, 0, 130, 0.2)',
                name='IC Solde 95%'
            ),
            row=1, col=1
        )
        
        # Ajouter une ligne horizontale à zéro
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="gray"),
            x0=df_enc['ds'].min(), x1=future_dates[-1], y0=0, y1=0,
            row=1, col=1
        )
        
        # Afficher la probabilité de solde négatif
        fig.add_trace(
            go.Bar(
                x=future_dates,
                y=monte_carlo_results['prob_negative_solde'],
                name='Probabilité de Solde Négatif (%)',
                marker_color='rgba(255, 0, 0, 0.7)'
            ),
            row=2, col=1
        )
        
        # Ajouter une annotation pour le nombre de simulations
        n_simulations = monte_carlo_results.get('n_simulations', 1000)
        fig.add_annotation(
            text=f"Basé sur {n_simulations} simulations",
            xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        )
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text="Simulation Monte Carlo - Évaluation des Risques",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_sensitivity_analysis_chart(self, sensitivity_results):
        """
        Crée un graphique pour visualiser les résultats de l'analyse de sensibilité.
        
        Args:
            sensitivity_results (dict): Résultats de l'analyse de sensibilité
            
        Returns:
            Figure: Objet Figure Plotly
        """
        if not sensitivity_results or 'factor_importance' not in sensitivity_results:
            # Créer un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Pas de résultats d'analyse de sensibilité disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Analyse de Sensibilité",
                height=400
            )
            return fig
        
        # Créer un graphique pour l'importance des facteurs
        importance = sensitivity_results['factor_importance']
        factors = list(importance.keys())
        values = list(importance.values())
        
        # Trier par importance décroissante
        sorted_indices = np.argsort(values)[::-1]
        factors = [factors[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les barres pour l'importance des facteurs
        fig.add_trace(go.Bar(
            y=factors,  # Inverser pour avoir les barres horizontales
            x=values,
            orientation='h',
            marker_color=px.colors.sequential.Viridis,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        ))
        
        # Amélioration du design
        fig.update_layout(
            title=dict(
                text="Analyse de Sensibilité - Importance Relative des Facteurs",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def create_scenario_chart(self, df_enc, df_dec, scenario, future_dates):
        """
        Crée un graphique pour visualiser un scénario de prévision.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            scenario (dict): Dictionnaire du scénario (encaissement, décaissement)
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Création du graphique pour le scénario
        fig_scenario = go.Figure()
        
        # Données historiques
        fig_scenario.add_trace(go.Scatter(
            x=df_enc['ds'], 
            y=df_enc['y_enc'], 
            mode='lines', 
            name='Encaissements Historiques',
            line=dict(color='rgba(0, 128, 0, 0.5)', width=2)
        ))
        fig_scenario.add_trace(go.Scatter(
            x=df_dec['ds'], 
            y=df_dec['y_dec'], 
            mode='lines', 
            name='Décaissements Historiques',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=2)
        ))
        
        # Prévisions du scénario
        fig_scenario.add_trace(go.Scatter(
            x=future_dates, 
            y=scenario['encaissement'], 
            mode='lines', 
            name='Encaissements Simulés',
            line=dict(color='rgba(0, 128, 0, 0.5)', width=3, dash='dash')
        ))
        fig_scenario.add_trace(go.Scatter(
            x=future_dates, 
            y=scenario['decaissement'], 
            mode='lines', 
            name='Décaissements Simulés',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=3, dash='dash')
        ))
        
        # Solde prévisionnel
        solde = scenario['encaissement'] - scenario['decaissement']
        fig_scenario.add_trace(go.Scatter(
            x=future_dates, 
            y=solde, 
            mode='lines', 
            name='Solde Prévisionnel',
            line=dict(color='rgba(75, 0, 130, 0.7)', width=3),
            fill='tozeroy'
        ))
        
        # Amélioration du design
        fig_scenario.update_layout(
            title=dict(
                text="Simulation de Scénario",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig_scenario
    
    def create_all_models_chart(self, df_enc, df_dec, forecasts, best_model, future_dates):
        """
        Crée un graphique affichant tous les modèles de prévision activés.
        
        Args:
            df_enc (DataFrame): Données historiques d'encaissements
            df_dec (DataFrame): Données historiques de décaissements
            forecasts (dict): Dictionnaire des prévisions par modèle
            best_model (str): Nom du meilleur modèle
            future_dates (array): Dates futures pour les prévisions
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Création du graphique avec un fond amélioré
        fig_models = go.Figure()
        
        # Options d'affichage
        show_grid = self.config.get('show_grid', True)
        
        # Ajouter une ligne de référence à zéro
        fig_models.add_shape(
            type="line",
            x0=min(df_enc['ds'].min(), df_dec['ds'].min()),
            x1=max(future_dates[-1] if len(future_dates) > 0 else df_enc['ds'].max(), 
                   df_enc['ds'].max(), df_dec['ds'].max()),
            y0=0,
            y1=0,
            line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot")
        )
        
        # Données historiques avec un style amélioré
        fig_models.add_trace(go.Scatter(
            x=df_enc['ds'], 
            y=df_enc['y_enc'], 
            mode='lines+markers', 
            name='Encaissements Historiques',
            line=dict(color=self.color_palette['encaissement'], width=2.5),
            marker=dict(size=7, line=dict(width=1, color='white')),
            hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Montant</b>: %{y:,.2f} DH<extra></extra>'
        ))
        
        # Filtrer les modèles d'encaissements
        enc_models = [m for m in forecasts.keys() if 'enc' in m]
        
        # Utiliser une palette de couleurs plus harmonieuse et distincte
        colors = px.colors.qualitative.Bold
        if len(enc_models) > len(colors):
            # Si nous avons plus de modèles que de couleurs, utiliser une palette étendue
            colors = px.colors.qualitative.Bold + px.colors.qualitative.Safe
        
        # Ajouter une ligne de séparation entre historique et prévision
        if len(future_dates) > 0:
            separation_date = future_dates[0]
            fig_models.add_shape(
                type="line",
                x0=separation_date,
                x1=separation_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
            )
            
            # Annotations pour les périodes
            fig_models.add_annotation(
                x=df_enc['ds'].min() + (separation_date - df_enc['ds'].min())/2,
                y=1.05,
                yref="paper",
                text="Données Historiques",
                showarrow=False,
                font=dict(size=12, color="#555")
            )
            
            fig_models.add_annotation(
                x=separation_date + (future_dates[-1] - separation_date)/2,
                y=1.05,
                yref="paper",
                text="Prévisions",
                showarrow=False,
                font=dict(size=12, color="#555")
            )
        
        # Ajouter chaque modèle avec un style amélioré
        for i, model_name in enumerate(enc_models):
            model_color = colors[i % len(colors)]
            model_display_name = model_name.replace('_enc', '').replace('_', ' ').title()
            
            if model_name == best_model:
                # Mettre en évidence le meilleur modèle avec un style distinctif
                fig_models.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasts[model_name], 
                    mode='lines+markers', 
                    name=f'{model_display_name} ⭐',
                    line=dict(color=self.color_palette['enc_prev'], width=3.5, dash='solid'),
                    marker=dict(size=8, symbol='star', color=self.color_palette['encaissement']),
                    hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Prévision</b>: %{y:,.2f} DH<br><b>Modèle</b>: ' + model_display_name + ' (Meilleur)<extra></extra>'
                ))
            else:
                # Autres modèles avec des styles différenciés
                fig_models.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasts[model_name], 
                    mode='lines', 
                    name=model_display_name,
                    line=dict(color=model_color, width=2, dash='dot'),
                    opacity=0.8,
                    hovertemplate='<b>Date</b>: %{x|%d-%m-%Y}<br><b>Prévision</b>: %{y:,.2f} DH<br><b>Modèle</b>: ' + model_display_name + '<extra></extra>'
                ))
        
        # Amélioration du design global
        fig_models.update_layout(
            title=dict(
                text="Comparaison des Modèles de Prévision",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig_models
    
    def create_financial_indicators_chart(self, metrics):
        """
        Crée un graphique radar des indicateurs financiers.
        
        Args:
            metrics (dict): Dictionnaire des métriques financières
            
        Returns:
            Figure: Objet Figure Plotly
        """
        # Normalisation des ratios pour le graphique radar
        radar_ratios = {
            'Couverture des Dépenses': min(metrics['Ratio de Couverture'] * 50, 100),
            'Croissance des Revenus': min(max(metrics['Taux de Croissance Encaissements'] + 50, 0), 100),
            'Stabilité des Flux': min(metrics['Indice de Stabilité'] * 100, 100),
            'Marge de Sécurité': min(max(metrics['Marge de Sécurité (%)'] / 2 + 50, 0), 100),
            'Faible Volatilité': min(max(100 - metrics['Volatilité Encaissements (%)'], 0), 100)
        }
        
        # Création du graphique radar
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(radar_ratios.values()),
            theta=list(radar_ratios.keys()),
            fill='toself',
            name='Profil Financier'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=dict(
                text="Profil Financier de l'Entreprise",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            )
        )
        
        return fig_radar
    
    def create_metrics_chart(self, model_metrics, scenarios=None):
        """
        Crée un graphique des métriques de performance des modèles.
        
        Args:
            model_metrics (dict): Dictionnaire des métriques de performance pour chaque modèle
            scenarios (list, optional): Liste des scénarios analysés
            
        Returns:
            plotly.graph_objects.Figure: Graphique Plotly avec plusieurs onglets pour les différentes métriques
        """
        if not model_metrics:
            # Retourner un graphique vide si aucune métrique n'est disponible
            fig = go.Figure()
            fig.update_layout(
                title="Aucune métrique disponible",
                xaxis_title="Erreur",
                yaxis_title="Modèle"
            )
            return fig
        
        # Convertir le dictionnaire en DataFrame
        metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        
        # Normaliser les noms de colonnes (majuscules/minuscules)
        metrics_df.columns = [col.upper() for col in metrics_df.columns]
        
        # Vérifier les colonnes disponibles
        available_metrics = []
        if 'MAE' in metrics_df.columns:
            available_metrics.append('MAE')
        if 'RMSE' in metrics_df.columns:
            available_metrics.append('RMSE')
        if 'MAPE' in metrics_df.columns:
            available_metrics.append('MAPE')
        
        # Créer une figure avec plusieurs sous-graphiques (un par métrique)
        fig = make_subplots(rows=len(available_metrics), cols=1, 
                           subplot_titles=[f"Métrique de performance: {metric}" for metric in available_metrics],
                           vertical_spacing=0.1)
        
        # Couleurs pour les barres
        colors = px.colors.qualitative.Plotly
        
        # Ajouter un graphique pour chaque métrique disponible
        for i, metric in enumerate(available_metrics):
            # Trier les données par la métrique actuelle
            sorted_df = metrics_df.sort_values(metric)
            
            # Déterminer le titre de l'axe X en fonction de la métrique
            if metric == 'MAE':
                xaxis_title = "Erreur Absolue Moyenne (MAE)"
            elif metric == 'RMSE':
                xaxis_title = "Racine de l'Erreur Quadratique Moyenne (RMSE)"
            elif metric == 'MAPE':
                xaxis_title = "Erreur Absolue Moyenne en Pourcentage (MAPE %)"
            else:
                xaxis_title = metric
            
            # Ajouter le graphique à barres horizontales
            fig.add_trace(
                go.Bar(
                    y=sorted_df.index,
                    x=sorted_df[metric],
                    orientation='h',
                    name=metric,
                    text=sorted_df[metric].round(2),
                    textposition='auto',
                    marker_color=colors[i % len(colors)]
                ),
                row=i+1, col=1
            )
            
            # Mettre à jour les titres des axes
            fig.update_xaxes(title_text=xaxis_title, row=i+1, col=1)
            fig.update_yaxes(title_text="Modèle", row=i+1, col=1)
        
        # Mise à jour de la mise en page générale
        fig.update_layout(
            title=dict(
                text="Comparaison des Métriques de Performance par Modèle",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#1e40af')
            ),
            margin=dict(t=100, l=10, r=10, b=10),
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
        
        return fig
    
    def generate_financial_recommendations(self, metrics):
        """
        Génère des recommandations financières basées sur les métriques.
        
        Args:
            metrics (dict): Dictionnaire des métriques financières
            
        Returns:
            list: Liste des recommandations
        """
        recommendations = []
        
        if metrics['Ratio de Couverture'] < 1:
            recommendations.append("⚠️ **Attention au déficit de trésorerie.** Envisagez d'augmenter les encaissements ou de réduire les décaissements.")
        elif metrics['Ratio de Couverture'] > 1.5:
            recommendations.append("✅ **Bonne couverture des dépenses.** Envisagez d'investir l'excédent de trésorerie.")
        
        if metrics['Taux de Croissance Encaissements'] < metrics['Taux de Croissance Décaissements']:
            recommendations.append("⚠️ **Les dépenses augmentent plus vite que les revenus.** Surveillez cette tendance pour éviter des problèmes de trésorerie à long terme.")
        
        if metrics['Indice de Stabilité'] < 0.5:
            recommendations.append("⚠️ **Flux de trésorerie instables.** Envisagez des stratégies pour régulariser les encaissements et décaissements.")
        
        if metrics['Marge de Sécurité (%)'] < 0:
            recommendations.append("⚠️ **Marge de sécurité négative.** Constituez une réserve de trésorerie pour faire face aux imprévus.")
        
        if not recommendations:
            recommendations.append("✅ **Situation financière saine.** Continuez à surveiller régulièrement vos flux de trésorerie.")
        
        return recommendations
