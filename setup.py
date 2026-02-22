from setuptools import setup, find_packages

setup(
    name='abalytics',
    version='4.0.1',
    author='Janis Berneker',
    packages=find_packages(),
    license='LICENSE',
    description='Advanced A/B Testing Statistical Analytics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas>=2.2.0',
        'scipy>=1.12.0',
        'statsmodels>=0.14.6',
        'scikit-posthocs>=0.12.0',
        'pingouin>=0.5.5',
        'tabulate>=0.9.0'
    ],
    python_requires='>=3.8',
    project_urls={
        'Repository': 'https://github.com/BigBerny/abalytics',
    },
)