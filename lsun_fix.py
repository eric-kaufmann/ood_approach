# full credits to https://github.com/alinlab/CSI/blob/master/datasets/lsun_fix_preprocess.py 
# Interesting paper! May take a look! -> https://arxiv.org/abs/2007.08176

import os
import time
import random
from tqdm import tqdm

import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

check = time.time()

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

orig_lsun_path = '../data/lsun'

set_random_seed(0)

LSUN_class_list = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                   'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']

total_test_image_all_class = []
for LSUN_class in tqdm(LSUN_class_list):
    LSUN_set = datasets.LSUN(orig_lsun_path, classes=[LSUN_class + "_train"], transform=transform)
    LSUN_loader = DataLoader(LSUN_set, batch_size=100, shuffle=True, pin_memory=False)

    total_test_image = None
    for n, (test_image, _) in enumerate(LSUN_loader):

        if n == 0:
            total_test_image = test_image
        else:
            total_test_image = torch.cat((total_test_image, test_image), dim=0).cpu()

        if total_test_image.size(0) >= 1000:
            break

    total_test_image_all_class.append(total_test_image)

total_test_image_all_class = torch.cat(total_test_image_all_class, dim=0)

print (f'Preprocessing time {time.time()-check}')

if not os.path.exists("../data/lsun_fix"):
    os.mkdir("../data/lsun_fix")

check = time.time()
for i in range(10000):
    save_image(total_test_image_all_class[i], f'../data/lsun_fix/correct_resize_{i}.png')
print (f'Saving time {time.time()-check}')