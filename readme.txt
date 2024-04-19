conda create --name AUDIOLIBROSA

conda activate AUDIOLIBROSA

conda install -c conda-forge librosa
pip install --upgrade pip
pip install xgboost notebook scipy matplotlib scikit-learn tensorflow

jupyter notebook
import matplotlib.pyplot as plt
%matplotlib inline 

conda deactivate