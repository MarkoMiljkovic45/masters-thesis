#!/bin/bash

python3 train.py -m \
  datamodule=synthetic \
  model=small,medium,large \
  loss=mse,nll,combined \
  trainer=fast,slow