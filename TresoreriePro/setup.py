from setuptools import setup, find_packages
import pathlib

# Le répertoire contenant ce fichier
HERE = pathlib.Path(__file__).parent

# Le texte du README
README = (HERE / "README.md").read_text(encoding='utf-8')

# Appel de la fonction setup
setup(
    name="tresorerie-pro",
    version="0.1.0",
    description="Application professionnelle de prévision de trésorerie",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-utilisateur/tresorerie-pro",
    author="Votre Nom",
    author_email="votre.email@example.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.32.0",
        "pandas>=2.0.0,<3.0.0",
        "prophet>=1.1.5",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "optuna>=3.3.0",
        "xgboost>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "statsmodels>=0.14.0",
        "joblib>=1.3.0",
        "plotly>=5.15.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
        "pytz>=2022.7.1",
        "openpyxl>=3.1.0",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.2.2",
        ],
        "test": [
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.2.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "tresorerie-pro=app:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/votre-utilisateur/tresorerie-pro/issues",
        "Source": "https://github.com/votre-utilisateur/tresorerie-pro",
    },
)
