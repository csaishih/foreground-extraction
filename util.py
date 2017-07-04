import cv2 as cv

def readSource(fileName):
    try:
        source = cv.imread(fileName, 1)
    except:
        print("[ERROR] Source must be a color uint8 image")
        return None
    return source

def readMask(fileName):
    try:
        mask = cv.imread(fileName, 0)
    except:
        print("[ERROR] Alpha must be a grayscale uint8 image")
        return None
    return mask

def writeImage(fileName, image):
    try:
        cv.imwrite(fileName, image)
        success = True
    except:
        success = False
    return success

def debug(image):
    # Display the image
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
