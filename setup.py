from typing import List
from setuptools import setup, find_packages

def get_requirements() -> List[str]:
    requirements_list = []

    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:

                # remove newline character
                line = line.strip()

                # ignore comments
                if line.startswith('#') or line == '':
                    continue

                # ignore -e .[dev] requirements
                if line.startswith('-e'):
                    continue

                requirements_list.append(line)

    except FileNotFoundError:
        print("requirements.txt not found")

    return requirements_list


setup(
    name='rag_app',
    version='0.1',
    author='Vikram Kare',
    packages=find_packages(),
    install_requires=get_requirements(),
)