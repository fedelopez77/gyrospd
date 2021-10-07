#!/bin/bash
# Uncompress data files
tar -xvzf data.tar.gz

# create necessary dirs
mkdir -p ckpt
mkdir -p out
mkdir -p tensorboard
