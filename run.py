import numpy as np
import cv2 as cv
import argparse, os, sys
from util import *

def main(args):
    source = readSource(args.s)
    f = readMask(args.f)
    b = readMask(args.b)

    lowerLimit = 150000

    image = source.copy()
    iDown = [image]
    fDown = [f]
    bDown = [b]

    levels = 0
    imgSize = image.shape[0] * image.shape[1]

    while imgSize > lowerLimit:
        image = cv.pyrDown(iDown[-1])
        iDown.append(image)
        f = cv.pyrDown(fDown[-1])
        fDown.append(f)
        b = cv.pyrDown(bDown[-1])
        bDown.append(b)
        imgSize = image.shape[0] * image.shape[1]
        levels += 1

    dImage = iDown[-1]
    dF = fDown[-1]
    dB = bDown[-1]


    mask = np.ones(dImage.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
    bgdModel = np.zeros((1, 65), dtype=np.float64)
    fgdModel = np.zeros((1, 65), dtype=np.float64)

    mask[dF!=255] = cv.GC_FGD
    mask[dB!=255] = cv.GC_BGD

    cv.grabCut(dImage,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = dImage*mask[:,:,np.newaxis]

    # writeImage(args.o, img)

    mask *= 255
    iUp = [mask]
    for i in xrange(levels):
        if iUp[-1].shape[0] != iDown[-1 * (i + 1)].shape[0]:
            iUp[-1] = iUp[-1][:-1,:]
        if iUp[-1].shape[1] != iDown[-1 * (i + 1)].shape[1]:
            iUp[-1] = iUp[-1][:,:-1]
        mask = cv.pyrUp(iUp[-1])
        iUp.append(mask)


    guidedBoundary = np.ones_like(mask, dtype=np.uint8) * 255
    boundary = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = boundary[0][0]
    for p in points:
        guidedBoundary[p[0][1], p[0][0]] = 0

    mOut = source.copy()
    writeImage(args.o, guidedBoundary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        type=str,
                        help='Path to source image',
                        required=True)
    parser.add_argument('-f',
                        type=str,
                        help='Path to foreground mask image',
                        required=True)
    parser.add_argument('-b',
                        type=str,
                        help='Path to background mask image',
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        help='Path to output image',
                        required=True)
    args = parser.parse_args()

    t1 = t2 = 0
    t1 = cv.getTickCount()
    main(args)
    t2 = cv.getTickCount()
    print('Completed in %g seconds'%((t2-t1)/cv.getTickFrequency()))
