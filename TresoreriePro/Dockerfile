# Utilisation d'une image Python officielle comme image de base
FROM python:3.9-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances système requises par certaines bibliothèques
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source de l'application
COPY . .

# Exposition du port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'application
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande pour lancer l'application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
