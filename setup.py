# setup.py
from setuptools import setup, find_packages

setup(
    name='pynar',  # Nom du package
    version='1',  # Version actuelle du package
    packages=find_packages(),  # Trouve automatiquement tous les packages et sous-packages
    install_requires=[  # Liste des dépendances nécessaires
        'pandas',  # Bibliothèque pour la manipulation de données
        'xarray',  # Bibliothèque pour le traitement de données multidimensionnelles
        'numpy',   # Bibliothèque pour le calcul numérique
    ],
    description='Un package pour calculer des indicateurs climatiques',  # Brève description du package
    author='Renan LE ROUX',  # Nom de l'auteur
    author_email='renan.le-roux@inrae.fr',  # Email de l'auteur
    url='https://forgemia.inra.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar',  # URL du projet
    classifiers=[  # Classificateurs pour aider à la recherche de votre package
        'Programming Language :: Python :: 3',  # Indique que le package est compatible avec Python 3
        'License :: OSI Approved :: MIT License',  # Indique la licence du package
        'Operating System :: OS Independent',  # Indique que le package est indépendant du système d'exploitation
    ],
    python_requires='>=3.6',  # Indique la version minimale de Python requise
)