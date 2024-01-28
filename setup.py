from setuptools import setup

setup(
    name="lvd",
    version="0.1",
    packages=['lvd'],  # Specify only the 'lvd' package
    package_dir={'lvd': 'lvd'},  # Map the 'lvd' package to the 'lvd' directory
)