#!/bin/bash

# Back up
args=("$@")

# https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi

  shift
done

python download.py "${args[@]}"

# Train the number of models from nummodels argument
for i in `seq 1 $nummodels`
do
    python -m torch.distributed.run train.py "${args[@]}"
done

python logo.py "${args[@]}"