import os
import numpy as np
from Cython.Build import cythonize, build_ext
from setuptools import Extension, setup, find_packages
from apogee import __version__


def os_path(import_path: str, ext: str) -> str:
    """
    Build the path to a module from it's import path.
    """

    return os.path.join(*import_path.split(".")) + ext


def define_cython_extensions(
    *extensions: str,
    link_args: list = None,
    compile_args: list = None,
    language: str = "c",
    file_ext: str = ".pyx"
) -> list:
    """
    Build a list of Cython extension modules.
    """

    modules = []
    for extension in extensions:
        modules.append(
            Extension(
                extension,
                [os_path(extension, file_ext)],
                language=language,
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                include_dirs=["."],
            )
        )

    cythonize(modules)

    return modules


# list your extensions here
ext_modules = define_cython_extensions("apogee.factors.discrete.operations.fast.arithmetic")


setup(
    name="apogee",
    version=__version__,
    description="Apogee",
    author="Mark Douthwaite",
    author_email="mark.douthwaite@peak.ai",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "jupyter",
        "networkx",
        "scipy",
        "networkx",
        "pyyaml",
        "Cython"
    ],
    include_package_data=True,
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,
    build_ext=build_ext,
)
