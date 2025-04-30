from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
    try:
        with open(file_path,'r') as file_obj:
            requirements = file_obj.read().splitlines()
    
    except FileNotFoundError:
        print(f"Error: Requirements file not found in {file_path}")
        return[]
    
    return requirements

setup (
    name = "Customer segmentation project",
    version = "0.0.0.1,",
    author = "Ramguhan",
    author_email = "ramguhan99@gmail.com",
    packages = find_packages(),
    packages_dir = {"":"src"},
    install_requires = get_requirements("requirements.txt"),
    description = "A Machine Learning Project"
)