from setuptools import setup

setup(
    name='sfer',
    version='0.1.0',
    description='A example Python package',
    url='https://github.com/Saritus/sfer',
    author='Sebastian Mischke',
    author_email='sebastianmischke@gmx.de',
    license='BSD 2-clause',
    packages=['sfer'],
    install_requires=[
        'Keras~=2.2.5',
        'tensorflow~=1.15.0',
        'protobuf~=3.20.0',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],
)
