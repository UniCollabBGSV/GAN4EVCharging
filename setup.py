from setuptools import setup, find_packages

def get_requirements(path):
    with open(path) as f:
        requirements = f.read().strip().split("\n")
    return requirements

setup(
    name="GAN_IDS",
    version="0.0.0",
    description="IDS V2G using GAN to generate synthetic attacks",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    python_requires=">=3.9",  # Specify the minimum Python version required
)
