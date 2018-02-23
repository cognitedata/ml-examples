from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='LSTM keras GMLE CMHE',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)
