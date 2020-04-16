# SWM_Final
CSE 573 Final submission

## Install Dependencies

```bash
pip install -r requirements.txt
```
## Instructions to run code
- Create ```save/``` directory inside the ``./Code`` folder in order to save checkpoints
- Run ```python Code/train.py  --max_seq_len 200 --num_layers 2 ``` for the Transformer Model
- Run ```python Code/trainRNN.py  --max_seq_len 200 --num_layers 2``` for the RNN Model

## Tensorboard
```tensorboard --logdir=runs```
