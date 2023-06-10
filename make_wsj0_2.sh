#!/bin/bash

path=egs/wsj0_2/cv
if [[ ! -e $path ]]
then
    mkdir -p $path
fi
python -m svoice.data.audio dataset/wsj0_2/min/cv/mix > $path/mix.json
python -m svoice.data.audio dataset/wsj0_2/min/cv/s1 > $path/s1.json
python -m svoice.data.audio dataset/wsj0_2/min/cv/s2 > $path/s2.json

path=egs/wsj0_2/tr
if [[ ! -e $path ]]
then
    mkdir -p $path
fi
python -m svoice.data.audio dataset/wsj0_2/min/tr/mix > $path/mix.json
python -m svoice.data.audio dataset/wsj0_2/min/tr/s1 > $path/s1.json
python -m svoice.data.audio dataset/wsj0_2/min/tr/s2 > $path/s2.json

path=egs/wsj0_2/tt
if [[ ! -e $path ]]
then
    mkdir -p $path
fi
python -m svoice.data.audio dataset/wsj0_2/min/tt/mix > $path/mix.json
python -m svoice.data.audio dataset/wsj0_2/min/tt/s1 > $path/s1.json
python -m svoice.data.audio dataset/wsj0_2/min/tt/s2 > $path/s2.json