#!/bin/bash

python3 train.py -m \
  datamodule=real \
  model=large \
  model.learning_rate=0.001,0.0001,0.00001 \
  trainer.max_epochs=100,200