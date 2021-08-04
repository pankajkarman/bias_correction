from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
     name='bias_correction',
     version='0.3.1',
     description='python library for bias_correction',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/pankajkarman/bias_correction',
     author='Pankaj Kumar',
     author_email='pankaj.kmr1990@gmail.com',
     license='MIT',
     py_modules=['bias_correction'],
     install_requires=[
     "numpy", 'scipy', 'pandas', 'statsmodels', 'xarray'
     ],
     python_requires=">=3.6",
     setup_requires=['setuptools'],
)
