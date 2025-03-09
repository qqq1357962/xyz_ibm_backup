
import torch
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plotter')
parser.add_argument('--cmd', default='plot')
parser.add_argument('--path', required=True)
parser.add_argument('--fig', default='tmp.png')
parser.add_argument('--key', default=' ')
args = parser.parse_args()
import numpy as np

print('Reading {}...'.format(args.path))
data = torch.load(args.path, map_location=torch.device('cpu'))
# print(data)
import imageio
# data = torch.load('tmp.zip', map_location=torch.device('cpu'))

if (args.cmd == 'imshow'):
    # plt.figure()
    # im = plt.imshow(data.detach().cpu().numpy())
    # rb=plt.colorbar(im)
    # plt.savefig(args.fig)
    # plt.close()
    array = data.detach().cpu().numpy()

    plt.plot(array)
    plt.title('Tensor')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    plt.savefig(args.fig)
    plt.show()

# if (args.cmd == 'plot'):
#     plt.figure()
#     x = list(range(len(data)))
#     plt.plot(x, data.detach().cpu().numpy(), label=args.key)
#     plt.legend()
#     plt.savefig(args.fig)
#     plt.close()
# data0=data[0,:,:]
# # data = np.transpose(data, (1, 2, 0))
# print(data0.min())
# print(data0.max())
# data0-=data0.min()

# data0/=data0.max()
# print(data0)
# imageio.imsave("0.png",data0.detach().cpu().numpy())

# data1=data[1,:,:]
# # data = np.transpose(data, (1, 2, 0))
# data1-=data1.min()
# print(data1.min())
# print(data1.max())
# data1/=data1.max()
# print(data1)
# imageio.imsave("1.png",data1.detach().cpu().numpy())

# data[0]-=data[0].min()
# data[0]/=data[0].max()
# data[1]-=data[1].min()
# data[1]/=data[1].max()
# data = np.transpose(data, (1, 2, 0))
# imageio.imsave("2.png",data.detach().cpu().numpy())

# # os.system("rm tmp.zip")
