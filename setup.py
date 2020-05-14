import os
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

from apogee import __version__


PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/markdouthwaite/apogee/issues',
    'Source Code': 'https://github.com/markdouthwaite/apogee'
}

DOWNLOAD_URL = 'https://pypi.org/project/apogee/#files'

DESCRIPTION = 'A set of tools for quickly building and ' \
              'deploying Probabilistic Graphical Models.'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()


class build_numpy_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


class CythonHelper:
    def __init__(
        self, *extensions, include_dirs=None, language="c", file_ext=".pyx", **kwargs
    ):
        self._extensions = extensions
        self._language = language
        self._file_ext = file_ext
        self._include_dirs = include_dirs or ["."]
        self._kwargs = kwargs

    @property
    def numpy_dirs(self):
        def _dirs():
            import numpy as np

            for dir in np.get_include():
                yield dir

        return _dirs()

    def build(self, *args, **kwargs):
        from Cython.Build import build_ext, cythonize

        cythonize(self.extensions)
        return build_ext(*args, **kwargs)

    @property
    def extensions(self):
        extensions = []
        for extension in self._extensions:
            definition = Extension(
                extension,
                [os_path(extension, self._file_ext)],
                language=self._language,
                **self._kwargs,
                include_dirs=self._include_dirs,
            )
            extensions.append(definition)
        return extensions


def os_path(import_path: str, ext: str) -> str:
    """
    Build the path to a module from it's import path.
    """

    return os.path.join(*import_path.split(".")) + ext


# list your extensions here
cython_helper = CythonHelper(
    "apogee.factors.discrete.operations.fast.arithmetic", "apogee.core.fast.arrays",
)


setup(
    name="apogee",
    version=__version__,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    download_url=DOWNLOAD_URL,
    licence="MIT",
    author="Mark Douthwaite",
    author_email="mark.douthwaite@peak.ai",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    setup_requires=["numpy", "Cython"],
    install_requires=[
        "numpy",
        "jupyter",
        "networkx",
        "scipy",
        "networkx",
        "pyyaml",
        "Cython",
        "scikit-learn",
        "pandas",
    ],
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Cython',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 ('Programming Language :: Python :: '
                  'Implementation :: CPython'),
                 ('Programming Language :: Python :: '
                  'Implementation :: PyPy')
                 ],
    include_package_data=True,
    cmdclass={"build_ext": build_numpy_ext},
    ext_modules=cython_helper.extensions,
    build_ext=cython_helper.build,
)
