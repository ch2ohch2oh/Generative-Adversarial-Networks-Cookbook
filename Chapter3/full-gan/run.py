#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 28
WIDTH   = 28
CHANNEL = 1
LATENT_SPACE_SIZE = 100
EPOCHS = 10000
BATCH = 32
CHECKPOINT = 1000
# MODEL_TYLE:
#   -1: Train on all numbers from 0 to 9
#    x: Train only for digit x. This will make the convergence faster.
MODEL_TYPE = -1

trainer = Trainer(height=HEIGHT,\
                 width=WIDTH,\
                 channels=CHANNEL,\
                 latent_size=LATENT_SPACE_SIZE,\
                 epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,
                 model_type=MODEL_TYPE)
if input("Enter y to continue training: ") == 'y':
    trainer.train()
else:
    print("Bye")