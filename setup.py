from setuptools import setup, find_packages

setup(
    name='abalytics',
    version='2.0.0',
    author='Janis Berneker',
    packages=find_packages(),
    license='LICENSE',
    description='Advanced A/B Testing Statistical Analytics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas>=1.5.2',
        'scipy>=1.10.1',
        'statsmodels>=0.14.1',
        'scikit-posthocs>=0.8.1',
        'pingouin>=0.5.4'
    ],
    python_requires='>=3.8',
    project_urls={
        'Repository': 'https://github.com/BigBerny/abalytics',
    },
)