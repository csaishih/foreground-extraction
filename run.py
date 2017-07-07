import numpy as np
import cv2 as cv
import argparse, os, sys
from threading import Thread
from util import *

class GraphCut:
    def __init__(self, source, foregroundSeed, backgroundSeed):
        self.source = source.copy()
        self.cutOut = source.copy()
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed

        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        self.kernel = np.ones((5,5), np.uint8)

        self.cuts = None
        self.numCuts = None
        self.foreground = None
        self.background = None
        self.results = [None, None, None, None]

        self._combined = None
        self._initialMask = None
        self._finalMask = np.zeros(self.source.shape[:2], dtype=np.uint8)

        self.numRows = source.shape[0]
        self.numCols = source.shape[1]
        self.minImageSize = 500000
        self.timesDownsampled = 0

        self.erodeIterations = 8
        self.dilateIterations = 8
        self.patchRadius = 50

    def run(self):
        self.downSample()
        self.cuts, self.foreground, self.background, self._combined = self.getBoundaries()
        self.refineBoundary(self.cuts)
        self.cutOut[self._finalMask==0] = 0
        return self.cutOut, self._finalMask, self._combined, self._initialMask


    def downSample(self):
        '''
        Constructs the image pyramid
        '''
        imageSize = self.numRows * self.numCols
        while imageSize > self.minImageSize:
            image = cv.pyrDown(self.iDown[-1])
            self.iDown.append(image)

            foreground = cv.pyrDown(self.fDown[-1])
            self.fDown.append(foreground)

            background = cv.pyrDown(self.bDown[-1])
            self.bDown.append(background)

            imageSize = image.shape[0] * image.shape[1]
            self.timesDownsampled += 1

    def getBoundaries(self):
        downSizedImage = self.iDown[-1]
        downSizedForeground = self.fDown[-1]
        downSizedBackground = self.bDown[-1]

        mask = self.cut(downSizedImage, downSizedForeground, downSizedBackground)

        for i in xrange(self.timesDownsampled):
            if mask.shape[0] != self.iDown[-1 * (i + 1)].shape[0]:
                mask = mask[:-1,:]
            if mask.shape[1] != self.iDown[-1 * (i + 1)].shape[1]:
                mask = mask[:,:-1]
            mask = cv.pyrUp(mask)
        self._initialMask = mask.copy()

        mask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        erodedMask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        dilatedMask = cv.dilate(mask, self.kernel, iterations=self.dilateIterations)

        boundary = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        erodedBoundary = cv.findContours(erodedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        dilatedBoundary = cv.findContours(dilatedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        combined = np.ones(self.source.shape, dtype=np.uint8) * 255
        foreground = np.zeros(self.source.shape[:2], dtype=np.uint8)
        background = np.zeros(self.source.shape[:2], dtype=np.uint8)

        cuts = []
        self.numCuts = min((len(boundary[0]), len(erodedBoundary[0]), len(dilatedBoundary[0])))
        for i in range(1, self.numCuts + 1):
            points = boundary[0][-i]
            erodedPoints = erodedBoundary[0][-i]
            dilatedPoints = dilatedBoundary[0][-i]

            for p in points: # Red
                combined[p[0][1], p[0][0]] = [0, 0, 255]

            for p in erodedPoints: # Green
                foreground[p[0][1], p[0][0]] = 255
                combined[p[0][1], p[0][0]] = [0, 255, 0]

            for p in dilatedPoints: # Blue
                background[p[0][1], p[0][0]] = 255
                combined[p[0][1], p[0][0]] = [255, 0, 0]

            cuts.append((points, erodedPoints, dilatedPoints))

        return cuts, foreground, background, combined


    def refineBoundary(self, cuts):
        jobs = []
        for cut in cuts:
            points = cut[0]

            for i in range(self.patchRadius // 2, points.shape[0], self.patchRadius):
                row = points[i][0][1]
                col = points[i][0][0]
                patch = self.source[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius),:]
                foregroundPatch = self.foreground[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)]
                backgroundPatch = self.background[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)]
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
        ccc = []
        for cut in cuts:
            points = cut[0]
            contours = np.zeros(self.source.shape[:2], dtype=np.uint8)

            for i in range(self.patchRadius // 2, points.shape[0], self.patchRadius):
                if tID < 3 and jobID == numTasks:
                    tID += 1
                    jobID = 0
                result = self.results[tID][jobID]
                jobID += 1

                row = points[i][0][1]
                col = points[i][0][0]
                contours[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)] = result
            ccc.append(contours)
            writeImage("test.png", contours)
            boundary = cv.findContours(contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(self._finalMask, boundary[0], -1, 255, thickness=-1)
            # contourIndex = np.argmax([len(x) for x in boundary[0]])
            # cv.drawContours(self._finalMask, boundary[0], contourIndex, 255, thickness=-1)


    def threadCut(self, tID, jobs):
        self.results[tID] = []
        for job in jobs:
            patch = job[0]
            foregroundPatch = job[1]
            backgroundPatch = job[2]

            _mask = self.test(patch, foregroundPatch, backgroundPatch)
            self.results[tID].append(_mask)

    def test(self, image, foreground, background):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        f = image.copy()
        b = image.copy()
        u = image.copy()
        f[foreground==0] = 0
        b[background==0] = 0

        fAverageColor = np.sum(f, axis=(0,1))
        bAverageColor = np.sum(b, axis=(0,1))

        if np.count_nonzero(np.sum(f, axis=2)) != 0:
            fAverageColor /= np.count_nonzero(np.sum(f, axis=2))
        if np.count_nonzero(np.sum(b, axis=2)) != 0:
            bAverageColor /= np.count_nonzero(np.sum(b, axis=2))

        tt = np.zeros(image.shape, dtype=np.uint8)
        tt[:,:] = fAverageColor
        yy = np.zeros(image.shape, dtype=np.uint8)
        yy[:,:] = bAverageColor

        ff = np.sum(np.abs((u - fAverageColor).astype(np.int64)),axis=2)
        bb = np.sum(np.abs((u - bAverageColor).astype(np.int64)),axis=2)

        dd = ff - bb

        closerToForeground = np.array(dd < -10)
        mask[closerToForeground==1] = 255

        return mask


    def cut(self, image, foreground, background):
        '''
        Performs a graph cut given the image, foreground mask and background mask
        '''
        mask = np.ones(image.shape[:2], dtype=np.uint8) * cv.GC_PR_BGD
        mask[background!=255] = cv.GC_BGD
        mask[foreground!=255] = cv.GC_FGD

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)

        cv.grabCut(image,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask *= 255
        return mask


def main(args):
    source = readSource(args.s)
    foregroundSeed = readMask(args.f)
    backgroundSeed = readMask(args.b)

    assert source is not None
    assert foregroundSeed is not None
    assert backgroundSeed is not None

    graphCut = GraphCut(source, foregroundSeed, backgroundSeed)
    result, _finalMask, _combined, _initialMask = graphCut.run()

    splitString = args.o.split('.')
    _finalMaskPath = str(splitString[0]) + '_finalMask.'
    _combinedPath = str(splitString[0]) + '_combined.'
    _initialMaskPath = str(splitString[0]) + '_initialMask.'

    for i in range(1, len(splitString)):
        _finalMaskPath += str(splitString[i])
        _combinedPath += str(splitString[i])
        _initialMaskPath += str(splitString[i])

    writeImage(args.o, result)
    writeImage(_finalMaskPath, _finalMask)
    writeImage(_combinedPath, _combined)
    writeImage(_initialMaskPath, _initialMask)

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
