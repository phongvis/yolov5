#!/bin/bash

python download.py "$@"
python -m torch.distributed.run train.py "$@"
python logo.py "$@"