#!/bin/bash

path=egs/debug/tr
if [[ ! -e $path ]]
then
    mkdir -p $path
fi
python -m svoice.data.audio dataset/debug/mix > $path/mix.json
python -m svoice.data.audio dataset/debug/s1 > $path/s1.json
python -m svoice.data.audio dataset/debug/s2 > $path/s2.json
