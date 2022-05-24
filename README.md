## Usage
```
mkdir log
mkdir save
mkdir save/pretrained_lm // download
```
download the pretrained model params (GPT) from [here](https://github.com/openai/finetune-transformer-lm/tree/master/model). Put the files into `save/pretrained_lm`


download the pretrained fasttext model using `wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/fasttext_empathetic_dialogues.mdl  # fastText classifier used for EmoPrepend-1` 

### Train
Run the command (GPU will be used if available, make sure CUDA is installed):
```
python train.py --model_type [trans|moel|adde|adm|mime|kemp|cem]
```

### Interact with model
```
python play.py --model_path [checkpoint_dir] --turns 2
```


### Requirements
* PyTorch (version >=1.4)
* tqdm
* sklearn
* spacy (version < 3)
* ftfy
* pandas
* tensorboardX
