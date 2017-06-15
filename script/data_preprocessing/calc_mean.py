import numpy as np
import sys

if len(sys.argv) != 3:
    print "Usage: python calc_mean.py mean.npy chanelwise_mean.txt"
    sys.exit()

fname_rgb_mean = open(sys.argv[1], 'r')
fname_cw_mean = open(sys.argv[2], 'w')

#fname_rgb_mean = '/home/yuliu/Documents/0RSNet/data/PD/ndsm512_mean.npy'

arr_rgb = np.load(fname_rgb_mean)

print arr_rgb.shape
vec_rgb = np.mean(arr_rgb, axis=1)
vec_rgb = np.mean(vec_rgb, axis=1)
print vec_rgb.shape
print vec_rgb
fname_cw_mean.write(str(vec_rgb))
