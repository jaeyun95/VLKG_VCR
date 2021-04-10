# VLKG_VCR

Thanks for [TAB-VCR](https://github.com/Deanplayerljx/tab-vcr) and [VilBERT](https://github.com/facebookresearch/vilbert-multi-task)
## 1. Setting up Environment   
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

## 2. Download Dataset First.
VCR is original Dataset. TAB-VCR is a dataset with attribute or new tags added.   
[VCR](https://visualcommonsense.com/download/)   
[TAB-VCR](https://github.com/Deanplayerljx/tab-vcr/tree/master/data)   

## 3. Extract Knowledge.
[click here](https://github.com/jaeyun95/VLKG_VCR/tree/master/reference) and make your knowledge data.

## 4. Extract Embedded Feature.(image and language).
(1) Image Feature   
select version and extract object feature.   
```
# original
python vcr_extract_image_origin.py

# attribute
python vcr_extract_image_attr.py

# new tag
python vcr_extract_image_new_tag.py
```

(2) Language Feature   
[click here](https://github.com/rowanz/r2c/tree/master/data/get_bert_embeddings) and download embedded language data or extract own your language.   

## 5. Get Co-embedding Feature.   
I use Co-Transformer from [vilbert](https://github.com/facebookresearch/vilbert-multi-task)   
```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <pretrained_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1-2-4-7-8-9-10-11-12-13-15-17 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name multi_task_model
```

## 6. Training. 
```
python my_train.py -params {path_to_your_model_config} -folder
{path_to_save_model_checkpoints} -plot {plot name}
```

## 7. Evaluation.
```
python eval_from_preds.py -preds {path_to_prediction_file} 
```