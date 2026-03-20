from setuptools import find_packages , setup
from typing import List

#this func return list of requirements
def get_requirements(file_path:str)->List[str]: #-> means return type
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace("\n" , "") for req in requirements]
    return requirements












setup(
    name = "mlproject",
    version = "0.0.1",
    author= "mahak" , 
    author_email= "mahaksolanki49@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)