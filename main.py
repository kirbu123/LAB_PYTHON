import start
import imageio.v3 as iio
import torch
from dict_creator import simpson_dict
from neiron_architecture import SimpleConvNet
from fit_predict import fit, predict

import torch.nn as nn

from tqdm import tqdm_notebook

from os import listdir
from os.path import isfile, join

net = SimpleConvNet()

epochs = 1
for epoch in range(epochs):
  for file in listdir('train/simpsons_dataset'):
    print(file)
    counter = 0
    for image in listdir('train/simpsons_dataset/' + str(file)):
      if counter > 50:
        break
      counter += 1
      im = iio.imread('train/simpsons_dataset/' + str(file) + '/' + str(image))
      tensor = torch.from_numpy(im)
      elem = [tensor.transpose(0, 2).float(), torch.Tensor([simpson_dict[file]])]
      fit(net, nn.CrossEntropyLoss(), elem[0], elem[1].long(), 3e-4)

print(predict(net, elem[0]))