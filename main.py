import encoder,decoder
import evaluate
import cv2
import os
import numpy as np

def cut(image,x1,y1,x2,y2):
    cutimage=image[x1:x2,y1:y2,:]
    markimg=cv2.rectangle(image, (y1,x1), (y2,x2), (0,0,255), 5)
    return cutimage,markimg

if __name__ == "__main__":

    filename = 'pets'
    block_size =8
    quantize_type = 0

    directory=f'./output/{filename}'
    if not os.path.exists(directory):
        os.makedirs(directory)



    srcPath = f'./img/{filename}.jpg'

    srcImg = cv2.imread(srcPath)
    ori_w, ori_h,channel = srcImg.shape

    row, col, channel = srcImg.shape
    # rescale image if in need, zero padding
    if row % block_size != 0:
        temp=np.zeros((block_size-row % block_size,srcImg.shape[1],channel))
        srcImg=np.append(srcImg,temp,axis=0)

    if col % block_size != 0:
        temp = np.zeros((srcImg.shape[0],block_size - col % block_size , channel))
        srcImg = np.append(srcImg, temp,axis=1)
    if [row, col, channel] != srcImg.shape:
        print('create new image for process')
        srcPath = f'{directory}/newSrc.jpg'
        cv2.imwrite(srcPath, srcImg)
    print(srcImg.shape)

    resultName = f'{filename}_{block_size}'
    codePath=f'./{directory}/{resultName}.txt'
    resultPath=f'{directory}/{resultName}.jpg'

    # encode and decode
    encoder.encode(srcPath,codePath,imageSize=(row,col),block_size=block_size,quantize_type=quantize_type)
    decoder.decode(codePath,resultPath,block_size=block_size,quantize_type=quantize_type)

    # evaluate
    srcImg = cv2.imread(f'./img/{filename}.jpg')
    resultImg=cv2.imread(resultPath)

    psnr=evaluate.PSNR(srcImg, resultImg)
    print("PSNR= ", psnr)
    mse=evaluate.MSE(srcImg, resultImg)
    print("MSE= ",mse )

    # record
    f=open(f'{directory}/record.txt', 'a')
    f.write(f"{resultName}\n")
    f.write(f"MSE= {mse}\n")
    f.write(f"PSNR= {psnr}\n\n")
    f.close()


    # create detail
    (x, y) = (200, 510)
    box = 150

    src_cut,src_mark=cut(srcImg,x,y,x+box,y+box)
    cv2.imwrite(f'{directory}/{filename}_cut.jpg', src_cut)
    cv2.imwrite(f'{directory}/{filename}_mark.jpg', src_mark)

    res_cut,res_mark=cut(resultImg,x,y,x+box,y+box)
    cv2.imwrite(f'{directory}/{resultName}_cut.jpg', res_cut)
    cv2.imwrite(f'{directory}/{resultName}_mark.jpg', res_mark)