from setuptools import setup, find_packages

setup(
    name='mon_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'xarray',
        'numpy',
    ],
    description='Un package pour calculer des indicateurs climatiques',
    author='Votre Nom',
    author_email='votre.email@example.com',
    url='https://gitlab.com/votre_utilisateur/mon_package',
)