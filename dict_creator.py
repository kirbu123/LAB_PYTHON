from os import listdir
from os.path import isfile, join

simpson_dict = {}
counter = 0
for file in listdir('train/simpsons_dataset'):
  counter += 1
  simpson_dict[file] = counter