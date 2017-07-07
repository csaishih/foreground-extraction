# Shihan Ai

import numpy as np
import cv2 as cv
import argparse, os, sys
from threading import Thread
from util import *

class GraphCut:
    def __init__(self, source, foregroundSeed, backgroundSeed):
        # Keep a copy of source
        self.source = source.copy()

        # This is where the output is stored
        self.cutOut = source.copy()

        # Store the user marked foreground and background seeds
        self.foregroundSeed = foregroundSeed
        self.backgroundSeed = backgroundSeed

        # Used to create an image pyramid
        self.iDown = [self.source]
        self.fDown = [self.foregroundSeed]
        self.bDown = [self.backgroundSeed]

        # Kernel used for convolution
        self.kernel = np.ones((5,5), np.uint8)

        # Keep a list of contour points for items being cut out
        self.cuts = None

        # Number of objects being cut out
        self.numCuts = None

        # Refined foreground and background
        self.foreground = None
        self.background = None

        # Store the results of each thread
        self.results = [None, None, None, None]

        # Images stored for debugging purposes
        self._combined = None
        self._initialMask = None
        self._finalMask = np.zeros(self.source.shape[:2], dtype=np.uint8)

        self.numRows = source.shape[0]
        self.numCols = source.shape[1]
        self.minImageSize = 500000
        self.timesDownsampled = 0

        # Hyper parameters to play with
        self.erodeIterations = 5
        self.dilateIterations = 5
        self.patchRadius = 25
        self.colorWeight = 0.5
        self.locationWeight = 0.5
        self.contourSizeThreshold = 500

    def run(self):
        # We down sample the large image to a much more managable size
        # and we perform Grabcut on the downsampled image to get a rough
        # esimate of the cutout mask. Using this estimate, we can obtain a more
        # refined foreground and background seed.
        self.downSample()
        self.cuts, self.foreground, self.background, self._combined = self.getBoundaries()

        # Using the refined foreground and background seeds, we take the estimated
        # cutout mask and refine the contour by examining patches that lie on the contour
        # and cut out pixels with respect to the refined foreground and background seeds
        self.refineBoundary(self.cuts)

        # Cut out the original image using the refined mask
        self.cutOut[self._finalMask==0] = 0
        return self.cutOut, self._finalMask, self._combined, self._initialMask

    # Constructs an image pyramid by scaling down the image and foreground and background seeds
    def downSample(self):
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

    # Get a rough estimate of the cutout mask
    def getBoundaries(self):

        # Perform the cut
        mask = self.cut(self.iDown[-1], self.fDown[-1], self.bDown[-1])

        # Upsample the mask back to the size of the original image
        for i in xrange(self.timesDownsampled):
            if mask.shape[0] != self.iDown[-1 * (i + 1)].shape[0]:
                mask = mask[:-1,:]
            if mask.shape[1] != self.iDown[-1 * (i + 1)].shape[1]:
                mask = mask[:,:-1]
            mask = cv.pyrUp(mask)

        # Store a copy of the upsampled unrefined mask
        self._initialMask = mask.copy()

        # We know pixels inside the contour of the unrefined mask
        # should be a part of the foreground and that pixels outside of the
        # contour should be a part of the background so we can generate a better
        # foreground and background seed by eroded and dilating the unrefined mask
        mask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        erodedMask = cv.erode(mask, self.kernel, iterations=self.erodeIterations)
        dilatedMask = cv.dilate(mask, self.kernel, iterations=self.dilateIterations)

        boundary = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        erodedBoundary = cv.findContours(erodedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        dilatedBoundary = cv.findContours(dilatedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Create image representations of the new foreground and background seeds
        combined = np.ones(self.source.shape, dtype=np.uint8) * 255
        foreground = np.zeros(self.source.shape[:2], dtype=np.uint8)
        background = np.zeros(self.source.shape[:2], dtype=np.uint8)

        cuts = []
        self.numCuts = min((len(boundary[0]), len(erodedBoundary[0]), len(dilatedBoundary[0])))
        for i in range(1, self.numCuts + 1):
            points = boundary[0][-i]
            erodedPoints = erodedBoundary[0][-i]
            dilatedPoints = dilatedBoundary[0][-i]

            # Red
            for p in points:
                combined[p[0][1], p[0][0]] = [0, 0, 255]

            # Green
            for p in erodedPoints:
                foreground[p[0][1], p[0][0]] = 255
                combined[p[0][1], p[0][0]] = [0, 255, 0]

            # Blue
            for p in dilatedPoints:
                background[p[0][1], p[0][0]] = 255
                combined[p[0][1], p[0][0]] = [255, 0, 0]

            cuts.append((points, erodedPoints, dilatedPoints))

        return cuts, foreground, background, combined

    # Refines the contour by examining patches on the contour
    def refineBoundary(self, cuts):
        patches = []

        # For each object that needs to be cut out we look at its contour and
        # create a tuple like representation of the patch and its corresponding
        # foreground and background seeds
        for cut in cuts:
            points = cut[0]

            for i in range(self.patchRadius // 2, points.shape[0], self.patchRadius):
                row = points[i][0][1]
                col = points[i][0][0]
                patch = self.source[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius),:]
                foregroundPatch = self.foreground[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)]
                backgroundPatch = self.background[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)]
                patches.append((patch, foregroundPatch, backgroundPatch))

        # To speed up processing, we create four threads to examine and refine
        # these patches concurrently
        numTasks = len(patches) // 4
        t0 = Thread(target=self.threadCut, args=(0, patches[:numTasks]))
        t1 = Thread(target=self.threadCut, args=(1, patches[numTasks:2*numTasks]))
        t2 = Thread(target=self.threadCut, args=(2, patches[2*numTasks:3*numTasks]))
        t3 = Thread(target=self.threadCut, args=(3, patches[3*numTasks:]))

        t0.start()
        t1.start()
        t2.start()
        t3.start()

        t0.join()
        t1.join()
        t2.join()
        t3.join()

        # Take the result of the refinement process and copy the data onto
        # the initial mask. This should result in a mask with a sharpened contour edge
        tID = 0
        patchID = 0
        for cut in cuts:
            points = cut[0]
            contours = cv.erode(self._initialMask, self.kernel, iterations=10)
            for i in range(self.patchRadius // 2, points.shape[0], self.patchRadius):
                if tID < 3 and patchID == numTasks:
                    tID += 1
                    patchID = 0
                result = self.results[tID][patchID]
                patchID += 1

                row = points[i][0][1]
                col = points[i][0][0]
                contours[max(0, row-self.patchRadius):min(self.numRows, row+self.patchRadius),max(0, col-self.patchRadius):min(self.numCols, col+self.patchRadius)] = result

            # Draw the result of all cuts onto image
            # Filter out noise by ignoring contours that are too small
            boundary = cv.findContours(contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for i in range(len(boundary[0])):
                if cv.contourArea(boundary[0][i]) > self.contourSizeThreshold:
                    cv.drawContours(self._finalMask, boundary[0], i, 255, thickness=-1)

    # Function called by each thread to process the patches
    def threadCut(self, tID, patches):
        self.results[tID] = []
        for patch in patches:
            refinedPatch = self.refine(patch[0], patch[1], patch[2])
            self.results[tID].append(refinedPatch)

    # Refines a patch given a foreground and background seed by masking out pixels
    # based on color of the pixel and location of the pixel
    def refine(self, patch, foreground, background):

        # Placeholder for the resulting mask
        mask = np.zeros(patch.shape[:2], dtype=np.uint8)

        # Placeholder for the foreground pixels in the patch as marked by the foreground seed
        f = patch.copy()
        f[foreground==0] = 0

        # Placeholder for the background pixels in the patch as marked by the background seed
        b = patch.copy()
        b[background==0] = 0

        # Placeholder for the uncertain pixels in the patch
        u = patch.copy()

        # Calculate the average color of the foreground and background pixels in the patch
        fAverageColor = np.sum(f, axis=(0,1))
        if np.count_nonzero(np.sum(f, axis=2)) != 0:
            fAverageColor /= np.count_nonzero(np.sum(f, axis=2))

        bAverageColor = np.sum(b, axis=(0,1))
        if np.count_nonzero(np.sum(b, axis=2)) != 0:
            bAverageColor /= np.count_nonzero(np.sum(b, axis=2))

        fColor = np.sum(np.abs((u - fAverageColor).astype(np.int64)),axis=2)
        bColor = np.sum(np.abs((u - bAverageColor).astype(np.int64)),axis=2)

        # Normalize the values
        fColorNormal = fColor / (fColor + bColor + np.finfo(np.float64).eps)
        bColorNormal = bColor / (fColor + bColor + np.finfo(np.float64).eps)

        # Calculate the center of the cluster of foreground and background pixels
        fIndices = np.nonzero(f)
        fAverageLocation = (np.average(fIndices[0]).astype(np.int), np.average(fIndices[1]).astype(np.int))

        bIndices = np.nonzero(b)
        bAverageLocation = (np.average(bIndices[0]).astype(np.int), np.average(bIndices[1]).astype(np.int))

        # Calculate the distance between each pixel and the foreground and background cluster centers
        uIndices = np.indices(u.shape[:2])
        fRow = uIndices[0] - fAverageLocation[0]
        fCol = uIndices[1] - fAverageLocation[1]
        fDist = np.sqrt(np.square(fRow) + np.square(fCol))

        bRow = uIndices[0] - bAverageLocation[0]
        bCol = uIndices[1] - bAverageLocation[1]
        bDist = np.sqrt(np.square(bRow) + np.square(bCol))

        # Normalize the values
        fDistNormal = fDist / (fDist + bDist + np.finfo(np.float64).eps)
        bDistNormal = bDist / (fDist + bDist + np.finfo(np.float64).eps)

        # Calculate the cost of pixels being marked either foreground or background
        fCost = fColorNormal * self.colorWeight + fDistNormal * self.locationWeight
        bCost = bColorNormal * self.colorWeight + bDistNormal * self.locationWeight

        # Get the indices of the pixels that have a smaller cost when labelled foreground
        closerToForeground = fCost < bCost

        # Mark those pixels on the mask
        mask[closerToForeground==1] = 255

        return mask

    # Performs a basic Grabcut on an image given a foreground and background seed
    def cut(self, image, foreground, background):
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
    # Read images
    source = readSource(args.s)
    foregroundSeed = readMask(args.f)
    backgroundSeed = readMask(args.b)

    assert source is not None
    assert foregroundSeed is not None
    assert backgroundSeed is not None

    # Run algorithm
    graphCut = GraphCut(source, foregroundSeed, backgroundSeed)
    result, _finalMask, _combined, _initialMask = graphCut.run()

    # Figure out the paths for the debugging images
    splitString = args.o.split('.')
    _finalMaskPath = str(splitString[0]) + '_finalMask.'
    _combinedPath = str(splitString[0]) + '_combined.'
    _initialMaskPath = str(splitString[0]) + '_initialMask.'

    for i in range(1, len(splitString)):
        _finalMaskPath += str(splitString[i])
        _combinedPath += str(splitString[i])
        _initialMaskPath += str(splitString[i])

    # Write the images
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
