from setuptools import setup, find_packages

setup(
    name='pynar',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'xarray',
        'numpy',
    ],
    description='Un package pour calculer des indicateurs climatiques',
    author='Renan LE ROUX',
    author_email='renan.le-roux@inrae.fr',
    url='https://forgemia.inra.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar',
)