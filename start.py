import torch
import torchvision
from torchvision import transforms
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt  # для отрисовки картиночек

import imageio.v3 as iio

import glob

from os import listdir
from os.path import isfile, join

print('start')

simpson_dict = {'abraham_grampa_simpson': 1, 'agnes_skinner': 2, 'apu_nahasapeemapetilon': 3,
                'barney_gumble': 4, 'bart_simpson': 5, 'carl_carlson': 6, 'charles_montgomery_burns': 7,
                'chief_wiggum': 8, 'cletus_spuckler': 9, 'comic_book_guy': 10, 'disco_stu': 11, 'edna_krabappel': 12,
                'fat_tony': 13, 'gil': 14, 'groundskeeper_willie': 15, 'homer_simpson': 16, 'kent_brockman': 17,
                'krusty_the_clown': 18, 'lenny_leonard': 19, 'lionel_hutz': 20, 'lisa_simpson': 21, 'maggie_simpson': 22,
                'marge_simpson': 23, 'martin_prince': 24, 'mayor_quimby': 25, 'milhouse_van_houten': 26, 'miss_hoover': 27,
                'moe_szyslak': 28, 'ned_flanders': 29, 'nelson_muntz': 30, 'otto_mann': 31, 'patty_bouvier': 32,
                'principal_skinner': 33, 'professor_john_frink': 34, 'rainier_wolfcastle': 35, 'ralph_wiggum': 36,
                'selma_bouvier': 37, 'sideshow_bob': 38, 'sideshow_mel': 39, 'snake_jailbird': 40, 'troy_mcclure': 41, 'waylon_smithers': 42}