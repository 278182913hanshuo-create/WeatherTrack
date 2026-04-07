"""
WeatherTrack 包配置
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='weather-track',
    version='1.0.0',
    description='Advanced Multi-Object Tracking System for Adverse Weather Conditions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='278182913hanshuo-create',
    url='https://github.com/278182913hanshuo-create/WeatherTrack',
    license='MIT',
    
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'torchaudio>=0.12.0',
        'opencv-python>=4.6.0',
        'numpy>=1.21.0',
        'pyyaml>=6.0',
        'matplotlib>=3.5.0',
        'tensorboard>=2.10.0',
        'scipy>=1.7.0',
        'tqdm>=4.62.0',
    ],
    
    python_requires='>=3.9',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    keywords='object-tracking weather adverse-conditions video-analysis',
)
