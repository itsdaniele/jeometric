from setuptools import setup


def _get_version():
    with open("jeometric/__init__.py") as fp:
        for line in fp:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `jeometric/__init__.py`")


setup(
    name="jeometric",
    version=_get_version(),
    description="Graph Neural Networks in JAX",
    author="Daniele Paliotta",
    author_email="daniele.paliotta@unige.ch",
    license="Apache 2.0",
    packages=["jeometric"],
    install_requires=["jax>=0.4.20", "numpy>=1.18.5", "flax>=0.7.5", "pandas>=1.1.5"],
)
