from apogee import __version__
from setuptools import find_packages, setup


setup(name="apogee",
      version=__version__,
      description="Apogee",
      author="Mark Douthwaite",
      author_email="mark.douthwaite@peak.ai",
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=["numpy", "tornado", "jupyter", "networkx", "matplotlib", "scipy"],
      include_package_data=True,
      )

