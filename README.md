# GA-Reader
Implementation of the paper [Gated Attention Reader for Text Comprehension](https://arxiv.org/abs/1606.01549).

## Prerequisites
- Python 2.7
- Tensorflow (tested with 1.11.0)
- Keras (tested with 2.2.4)
- Numpy (>=1.12)
- Maybe more, just use `pip install` if you get an error


## Preprocessed Data
You can get the preprocessed data files from [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing). Extract the tar files to the `data/` directory. Ensure that the symbolic links point to folders with `training/`, `validation/` and `test/` directories for each dataset.

You can also get the pretrained Glove vectors from the above link. Place this file in the `data/` directory as well.

## To run
Issue the command:
```
python run.py --dataset cnn --gating_fn Tsum
```

