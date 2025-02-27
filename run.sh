#!/bin/bash

srun --exclusive --gres=gpu:4 \
	./main -v -w -n 16384 $@