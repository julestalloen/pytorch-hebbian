import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from pytorch_hebbian import config


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('mnist_digits-examples.png', bbox_inches='tight', pad_inches=0, transparent=True, quality=95)
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)

num = 20
images = torch.ByteTensor()
for i in range(10):
    idx = dataset.targets == i
    data = dataset.data[idx][:num]
    images = torch.cat((images, data))

images = torch.unsqueeze(images, dim=1)
grid = torchvision.utils.make_grid(images, nrow=num)
imshow(grid)
