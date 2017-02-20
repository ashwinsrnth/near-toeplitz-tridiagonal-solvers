from setuptools import setup, find_packages

setup(name='neato',
        version=0.1,
        description='near-Toeplitz tridiagonal system solver',
        url='http://github.com/shwina/neato',
        author='Ashwin Srinath',
        author_email='atrikut@clemson.edu',
        license='MIT',
        packages=find_packages(exclude=['*test*', 'python']),
        dependency_links=['https://mathema.tician.de/software/pycuda/'],
        zip_safe=False)
