import numpy as np
import cv2 as cv
import argparse, os, sys
from util import *

class GraphCut:
    def __init__(self, source, foregroundSeed, backgroundSeed, numThreads):
        self.source = source.copy()
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed
        self.minImageSize = 150000
        self.imageSize = self.source.shape[0] * self.source.shape[1]

        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        self.timesDownsampled = 0
        self.numThreads = numThreads

        self.boundary = None
        self.eroded = None
        self.dilated = None


    def run(self):
        self.downSample()
        self.boundary, self.eroded, self.dilated = self.getBoundaries()
        self.generateBoundaryImages(self.boundary, self.eroded, self.dilated, verbose=1)
        # self.result = self.cut(self.source, self.foregroundSeed, self.backgroundSeed)[0]
        # writeImage("result.png", self.result)

    def cut(self, image, foreground, background):
        mask = np.ones(image.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
        mask[foreground!=255] = cv.GC_FGD
        mask[background!=255] = cv.GC_BGD

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)

        cv.grabCut(image,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        return image * mask[:,:,np.newaxis], mask

    def getBoundaries(self):
        downSizedImage = self.iDown[-1]
        downSizedForeground = self.fDown[-1]
        downSizedBackground = self.bDown[-1]

        mask = self.cut(downSizedImage, downSizedForeground, downSizedBackground)[1]
        mask *= 255

        for i in xrange(self.timesDownsampled):
            if mask.shape[0] != self.iDown[-1 * (i + 1)].shape[0]:
                mask = mask[:-1,:]
            if mask.shape[1] != self.iDown[-1 * (i + 1)].shape[1]:
                mask = mask[:,:-1]
            mask = cv.pyrUp(mask)

        # Deal with multiple segmented objects
        kernel = np.ones((5,5), np.uint8)
        erodedMask = cv.erode(mask, kernel, iterations=15)
        dilatedMask = cv.dilate(mask, kernel, iterations=5)

        boundary = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        erodedBoundary = cv.findContours(erodedMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        dilatedBoundary = cv.findContours(dilatedMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        points = boundary[0][0]
        erodedPoints = erodedBoundary[0][0]
        dilatedPoints = dilatedBoundary[0][0]

        return points, erodedPoints, dilatedPoints


    def splitJobs(self):
        # offset = points.shape[0] // (self.numThreads * 4)
        # jobs = []
        #
        # for i in range(offset // 2, points.shape[0], offset):
        #     jobs.append((points[i][0][1], points[i][0][0]))
        #
        # for i in jobs:
        #     combined[i[0], i[1], 0] = 0
        #     combined[i[0], i[1], 2] = 0
        pass

    def generateBoundaryImages(self, points, erodedPoints, dilatedPoints, verbose):
        if verbose:
            combined = np.ones(self.source.shape, dtype=np.uint8) * 255

        boundary = np.ones(self.source.shape[:2], dtype=np.uint8) * 255
        foreground = np.ones(self.source.shape[:2], dtype=np.uint8) * 255
        background = np.ones(self.source.shape[:2], dtype=np.uint8) * 255

        for p in points:
            # Red
            if verbose:
                combined[p[0][1], p[0][0], 0] = 0
                combined[p[0][1], p[0][0], 1] = 0
            boundary[p[0][1], p[0][0]] = 0

        for p in erodedPoints:
            # Green
            if verbose:
                combined[p[0][1], p[0][0], 0] = 0
                combined[p[0][1], p[0][0], 2] = 0
            foreground[p[0][1], p[0][0]] = 0

        for p in dilatedPoints:
            # Blue
            if verbose:
                combined[p[0][1], p[0][0], 1] = 0
                combined[p[0][1], p[0][0], 2] = 0
            background[p[0][1], p[0][0]] = 0

        writeImage("_boundary.png", boundary)
        writeImage("_foreground.png", foreground)
        writeImage("_background.png", background)

        if verbose:
            writeImage("_combined.png", combined)

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

    graphCut = GraphCut(source, foregroundSeed, backgroundSeed, 8)
    graphCut.run()

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
