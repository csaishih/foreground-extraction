import numpy as np
import cv2 as cv
import argparse, os, sys
from threading import Thread
from util import *

class GraphCut:
    def __init__(self, source, foregroundSeed, backgroundSeed):
        self.source = source.copy()
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed

        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        self.mask = None
        self.cuts = None
        self.foreground = None
        self.background = None
        self.results = [None, None, None, None]

        self._combined = None
        self._combinedWithImage = None
        self._boundary = None

        self.imageSize = self.source.shape[0] * self.source.shape[1]
        self.minImageSize = 500000
        self.timesDownsampled = 0
        self.erodeIterations = 1
        self.dilateIterations = 1
        self.hyperOffset = 5

    def run(self):
        # return self.cut(self.source, self.foregroundSeed, self.backgroundSeed)[0]
        self.downSample()
        self.cuts = self.getBoundaries()
        self.generateBoundaryImages(self.cuts)
        self.refineBoundary(self.cuts)
        return self.source, self._boundary, self._combined, self._combinedWithImage


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
        self.mask = mask.copy()

        # TODO: Deal with multiple segmented objects
        kernel = np.ones((5,5), np.uint8)
        erodedMask = cv.erode(mask, kernel, iterations=self.erodeIterations)
        dilatedMask = cv.dilate(mask, kernel, iterations=self.dilateIterations)

        boundary = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        erodedBoundary = cv.findContours(erodedMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        dilatedBoundary = cv.findContours(dilatedMask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        points = boundary[0][0]
        erodedPoints = erodedBoundary[0][0]
        dilatedPoints = dilatedBoundary[0][0]

        cuts = []
        numCuts = min((len(boundary[0]), len(erodedBoundary[0]), len(dilatedBoundary[0])))
        for i in range(numCuts):
            points = boundary[0][i]
            erodedPoints = erodedBoundary[0][i]
            dilatedPoints = dilatedBoundary[0][i]
            cuts.append((points, erodedPoints, dilatedPoints))

        # return points, erodedPoints, dilatedPoints
        return cuts

    def generateBoundaryImages(self, cuts):
        combined = np.ones(self.source.shape, dtype=np.uint8) * 255
        combinedWithImage = self.source.copy()

        boundary = np.ones(self.source.shape[:2], dtype=np.uint8) * 255
        foreground = np.ones(self.source.shape[:2], dtype=np.uint8) * 255
        background = np.ones(self.source.shape[:2], dtype=np.uint8) * 255

        for cut in cuts:
            points = cut[0]
            erodedPoints = cut[1]
            dilatedPoints = cut[2]

            for p in points: # Red
                boundary[p[0][1], p[0][0]] = 0
                combined[p[0][1], p[0][0]] = [0, 0, 255]
                combinedWithImage[p[0][1], p[0][0]] = [0, 0, 255]

            for p in erodedPoints: # Green
                foreground[p[0][1], p[0][0]] = 0
                combined[p[0][1], p[0][0]] = [0, 255, 0]
                combinedWithImage[p[0][1], p[0][0]] = [0, 255, 0]

            for p in dilatedPoints: # Blue
                background[p[0][1], p[0][0]] = 0
                combined[p[0][1], p[0][0]] = [255, 0, 0]
                combinedWithImage[p[0][1], p[0][0]] = [255, 0, 0]

        self.foreground = foreground
        self.background = background

        self._combined = combined
        self._combinedWithImage = combinedWithImage
        self._boundary = boundary

    def refineBoundary(self, cuts):
        offset = self.hyperOffset
        numRows = self.source.shape[0]
        numCols= self.source.shape[1]

        jobs = []

        for cut in cuts:
            points = cut[0]

            for i in range(offset // 2, points.shape[0], offset):
                row = points[i][0][1]
                col = points[i][0][0]

                patch = self.source[max(0, row-offset):min(numRows, row+offset),max(0, col-offset):min(numCols, col+offset),:]
                foregroundPatch = self.foreground[max(0, row-offset):min(numRows, row+offset),max(0, col-offset):min(numCols, col+offset)]
                backgroundPatch = self.background[max(0, row-offset):min(numRows, row+offset),max(0, col-offset):min(numCols, col+offset)]

                jobs.append((patch, foregroundPatch, backgroundPatch))

        numTasks = len(jobs) // 4
        t0 = Thread(target=self.threadCut, args=(0, jobs[:numTasks]))
        t1 = Thread(target=self.threadCut, args=(1, jobs[numTasks:2*numTasks]))
        t2 = Thread(target=self.threadCut, args=(2, jobs[2*numTasks:3*numTasks]))
        t3 = Thread(target=self.threadCut, args=(3, jobs[3*numTasks:]))

        t0.start()
        t1.start()
        t2.start()
        t3.start()

        t0.join()
        t1.join()
        t2.join()
        t3.join()

        tID = 0
        jobID = 0

        for cut in cuts:
            points = cut[0]

            for i in range(offset // 2, points.shape[0], offset):
                row = points[i][0][1]
                col = points[i][0][0]

                if tID < 3 and jobID == numTasks:
                    tID += 1
                    jobID = 0
                # print((tID, jobID))
                result = self.results[tID][jobID]
                jobID += 1

                g = np.uint8(result>0) * 255
                g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)

                self.mask[max(0, row-offset):min(numRows, row+offset),max(0, col-offset):min(numCols, col+offset)] = g

        writeImage("outman.png", self.mask)
        # boundary = cv.findContours(self.mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # contourIndex = np.argmax([len(x) for x in boundary[0]])
        # points = boundary[0][contourIndex]
        #
        # for p in points:
        #     self.source[p[0][1], p[0][0]] = [0, 0, 255]

    def threadCut(self, tID, jobs):
        self.results[tID] = []
        for job in jobs:
            patch = job[0]
            foregroundPatch = job[1]
            backgroundPatch = job[2]
            self.results[tID].append(self.cut(patch, foregroundPatch, backgroundPatch)[0])

    def cut(self, image, foreground, background):
        mask = np.ones(image.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
        mask[background!=255] = cv.GC_BGD
        mask[foreground!=255] = cv.GC_FGD

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)

        cv.grabCut(image,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        return image * mask[:,:,np.newaxis], mask


def main(args):
    source = readSource(args.s)
    foregroundSeed = readMask(args.f)
    backgroundSeed = readMask(args.b)

    assert source is not None
    assert foregroundSeed is not None
    assert backgroundSeed is not None

    verbose = 1

    graphCut = GraphCut(source, foregroundSeed, backgroundSeed)
    result, _boundary, _combined, _combinedWithImage = graphCut.run()

    writeImage(args.o, result)

    if verbose:
        splitString = args.o.split('.')
        _boundaryPath = str(splitString[0]) + '_boundary.'
        _combinedPath = str(splitString[0]) + '_combined.'
        _combinedWithImagePath = str(splitString[0]) + '_combinedWithImage.'

        for i in range(1, len(splitString)):
            _boundaryPath += str(splitString[i])
            _combinedPath += str(splitString[i])
            _combinedWithImagePath += str(splitString[i])

        writeImage(_boundaryPath, _boundary)
        writeImage(_combinedPath, _combined)
        writeImage(_combinedWithImagePath, _combinedWithImage)

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
