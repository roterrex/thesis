#from PlanGen.util import util
from util import util
import tensorflow as tf
import os
import pathlib
import time
import datetime
import yaml

from matplotlib import pyplot as plt

from testTrain import testTrain
from pixToPix import pix2Pix
from postProcessor import postProcessor
from dataloader import dataloader
#from IPython import display
import time

conf = yaml.load(open("PlanGen/settings.yaml", 'r'), Loader=yaml.Loader)

PATH = pathlib.Path(conf['images']['Dataset'])
print(list(PATH.parent.iterdir()))

dataloader = dataloader(conf, PATH)
train_ds, test_ds, val_ds = dataloader.load_ds()

p2p = pix2Pix(conf) 
p2p.load_checkpoint(conf['Checkpoint']['load_from_checkpoint']) #only load if

if conf['images']['Epochs'] > 0:
  tt = testTrain(p2p, None, conf)
  tt.fit(train_ds, test_ds, steps=conf['images']['Epochs'])

pstpro = postProcessor(conf, p2p)
pstpro.process(val_ds)

while conf['Misc']['PauseOnFinish']:
  time.sleep(500000)
