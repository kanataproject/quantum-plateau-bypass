from setuptools import setup

setup(
    name='quantum_classifier',
    version='1.0.0',
    py_modules=['quantum_classifier'], # モジュールのファイル名（.py抜き）
    install_requires=[
        'pennylane',
        'scikit-learn'
    ],
)
