from setuptools import setup

setup(
    name="jeometric",
    version="0.0.1",
    description="Graph Neural Networks in JAX",
    # url="",
    author="Daniele Paliotta",
    author_email="daniele.paliotta@unige.ch",
    # license="BSD 2-clause",
    packages=["jeometric"],
    install_requires=[
        "pandas",
        "numpy",
    ],
)
