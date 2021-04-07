# VLKG_VCR


## 1.Setting up Environment   
```
conda create -n vlkg_vcr python=3.6
conda activate vlkg_vcr
git clone --recursive https://github.com/jaeyun95/VLKG_VCR.git
cd vlkg_vcr/code
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
python setup.py develop
conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d
pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm
```
cuda version9.0 is also possible.

## 2.extract knowledge first.
[click here](https://github.com/jaeyun95/VLKG_VCR/tree/master/reference) and make your knowledge data.