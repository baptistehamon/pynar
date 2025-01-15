# Mon Package

Ce package permet de calculer des indicateurs climatiques à partir de données NetCDF. Il fournit des outils pour analyser et extraire des informations pertinentes à partir de fichiers de données climatiques.

## Installation

Pour installer le package, vous pouvez cloner le dépôt et utiliser `pip` :

```sh
git clone https://forgemia.inra.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar.git
```


## Fonctionnalités

- **Calcul d'indicateurs climatiques** : Le package permet de calculer divers indicateurs à partir de données NetCDF.
- **Support des formats de données courants** : Compatible avec les fichiers NetCDF, ce qui est standard dans les données climatiques.

## Documentation de la fonction `compute_indicator_netcdf`
## Description

    Cette fonction calcule un indicateur climatique spécifique à partir de données NetCDF.

    Parameters:
    - `climate_data` (xr.Dataset): Le jeu de données NetCDF contenant les données climatiques.
    - `NameIndicator` (str): Le nom de l'indicateur climatique à calculer.
    - `varName` (str): Le nom de la variable climatique à utiliser pour le calcul de l'indicateur.
    - `threshold` (float, optional): Le seuil à utiliser pour les calculs impliquant un seuil. Par défaut, None.
    - `two_years_culture` (bool, optional): Indique si l'indicateur doit être calculé sur deux années de culture. Par défaut, True.
    - `start_stage` (int, float, xr.Dataset, optional): La date de début de la période à considérer pour le calcul de l'indicateur. Peut être un entier ou un flottant représentant le jour de l'année, ou un xr.Dataset contenant les données de début de période. Par défaut, None.
    - `end_stage` (int, float, xr.Dataset, optional): La date de fin de la période à considérer pour le calcul de l'indicateur. Peut être un entier ou un flottant représentant le jour de l'année, ou un xr.Dataset contenant les données de fin de période. Par défaut, None.

    Returns:
    - `xr.Dataset`: Un jeu de données xarray contenant le résultat du calcul de l'indicateur climatique.
    """


## Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet.
2. Créez une branche pour votre fonctionnalité (`git checkout -b ma-fonctionnalite`).
3. Effectuez vos modifications et validez (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`).
4. Poussez vers la branche (`git push origin ma-fonctionnalite`).
5. Ouvrez une demande de tirage (merge request) sur GitLab.

## Auteurs

- Renan LE ROUX - 

## License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.