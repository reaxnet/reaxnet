from setuptools import setup, find_packages

setup(name='reaxnet',
      version="1.0",
      install_requires=[
                        'jax-md>=0.2.8',
                        'jax>=0.4.20',
                        'ase',
                        'e3nn_jax>=0.20.2',
                        'flax>=0.7.5',
                        'jraph>=0.0.6.dev0',
                        'ml_collections>=0.1.1',
                        'numpy>=1.26.3',
                        'PyYAML>=6.0.1',
                        'matscipy>=1.0.0',
                        'optax>=0.1.8',

],
      author='Rongzhi Gao',
      packages=find_packages(),
      package_data={
        '':['dftd3_params.npz'],
           },
)
