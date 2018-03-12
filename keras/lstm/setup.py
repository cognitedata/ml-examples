from setuptools import setup, find_packages

setup(name='blueprint',
      version='0.1',
      packages=find_packages(),
      description='LSTM for CMHE',
      install_requires=[
          'keras',
          'h5py'
      ],
      package_data={'blueprint': ['processing_requirements.txt']},
      zip_safe=False)


