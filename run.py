import numpy as np
import cv2 as cv
import argparse, os, sys
from util import *

class GraphCut:
    def __init__(self, source, foregroundSeed, backgroundSeed):
        self.source = source.copy()
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed
        self.minImageSize = 150000
        self.imageSize = self.source.shape[0] * self.source.shape[1]

        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        self.timesDownsampled = 0
        self.boundary = None

    def cut(self):
        self.downSample()
        self.boundary = self.getGuidedBoundary()
        writeImage("boundary.png", self.boundary)

    def getGuidedBoundary(self):
        downSizedImage = self.iDown[-1]
        downSizedForeground = self.fDown[-1]
        downSizedBackground = self.bDown[-1]

        mask = np.ones(downSizedImage.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
        mask[downSizedForeground!=255] = cv.GC_FGD
        mask[downSizedBackground!=255] = cv.GC_BGD

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)

        cv.grabCut(downSizedImage,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask *= 255

        for i in xrange(self.timesDownsampled):
            if mask.shape[0] != self.iDown[-1 * (i + 1)].shape[0]:
                mask = mask[:-1,:]
            if mask.shape[1] != self.iDown[-1 * (i + 1)].shape[1]:
                mask = mask[:,:-1]
            mask = cv.pyrUp(mask)

        guidedBoundary = np.ones_like(mask, dtype=np.uint8) * 255
        boundary = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        points = boundary[0][0]
        for p in points:
            guidedBoundary[p[0][1], p[0][0]] = 0

        return guidedBoundary


    def downSample(self): # Builds image pyramids
        while self.imageSize > self.minImageSize:
            image = cv.pyrDown(self.iDown[-1])
            self.iDown.append(image)

            foreground = cv.pyrDown(self.fDown[-1])
            self.fDown.append(foreground)

            background = cv.pyrDown(self.bDown[-1])
            self.bDown.append(background)

            self.imageSize = image.shape[0] * image.shape[1]
            self.timesDownsampled += 1

def main(args):
    source = readSource(args.s)
    foregroundSeed = readMask(args.f)
    backgroundSeed = readMask(args.b)

    assert source is not None
    assert foregroundSeed is not None
    assert backgroundSeed is not None

    graphCut = GraphCut(source, foregroundSeed, backgroundSeed)
    graphCut.cut()

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
