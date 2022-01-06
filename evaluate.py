import numpy as np
import cv2
from math import log,sqrt

def MSE(src,result):
    r,c,channel=src.shape
    err=0
    for i in range(channel):
        target=src[:,:,i]
        test=result[:,:,i]
        errC=(target-test) ** 2
        errC=sum(sum(errC))
        err+=errC

    mse=err/(src.size) # total_err/(row*col*n_channel)
    return mse


def PSNR(src,result):
    r, c, channel = src.shape

    err = 0
    for i in range(channel):
        target = src[:, :, i]
        test = result[:, :, i]
        errC = (target - test) ** 2
        errC = sum(sum(errC))
        err += errC
    mse=err/(src.size) # total_err/(row*col*n_channel)
    psnr=10 * log( 255**2 / mse)

    return psnr

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


if __name__=='__main__':


    print('table 1')
    t=[0.1,0.5,1,5,10]
    for scale in t:

        q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
        q = q * scale
        min = q.min()
        max = q.max()
        print(f'lum min={min}, max= {max}')

        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
        q = q * scale
        min = q.min()
        max = q.max()
        print(f'chr min={min}, max= {max}')

        print('table 2')
        q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 4, 5, 7, 9],
                      [2, 2, 2, 4, 5, 7, 9, 12],
                      [3, 3, 4, 5, 8, 10, 12, 12],
                      [4, 4, 5, 7, 10, 12, 12, 12],
                      [5, 5, 7, 9, 12, 12, 12, 12],
                      [6, 6, 9, 12, 12, 12, 12, 12]])
        q = q * scale
        min=q.min()
        max=q.max()
        print(f'lum min={min}, max= {max}')

        q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                      [3, 4, 6, 11, 14, 12, 12, 12],
                      [5, 6, 9, 14, 12, 12, 12, 12],
                      [9, 11, 14, 12, 12, 12, 12, 12],
                      [13, 14, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12]])
        q = q * scale
        min = q.min()
        max = q.max()
        print(f'chr min={min}, max= {max}\n')