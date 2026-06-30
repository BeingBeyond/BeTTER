from setuptools import find_packages, setup


setup(
    name="better",
    version="0.1.0",
    description="BeTTER simulation and retrieval toolkit",
    packages=find_packages(include=["src", "src.*", "services", "services.*"]),
    include_package_data=True,
)