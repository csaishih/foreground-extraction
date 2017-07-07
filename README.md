# Object Segmentation For Large Images
Object segmentation has a wide number of uses in Computer Vision.

As technology advances, the resolution of your everyday photo also increases.

This is problematic for older traditional object segmentation methods as the computation time increases tremendously when working with high resolution images.

This repo proposes an object segmentation algorithm that reduces the time needed to process high resolution images while still achieving a decent result.

Inspiration of algorithm came from [here](../master/paper.pdf)


## How it works
How it works

## Running the script
###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -f [path to foreground seed image] -b [path to background seed image] -o [path to output image]`
###### Example usage
`python run.py -s test_images/test2/source.jpg -f test_images/test2/foreground.png -b test_images/test2/background.png -o test_images/test2/out.png`

## Results
###### Test 2 source (3024 × 4032) Taken with iPhone 7Plus
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/source.jpg "Test 2 source")

###### Test 2 foreground seed
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/foreground.png "Test 2 foreground seed")

###### Test 2 background seed
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/background.png "Test 2 background seed")

###### Test 2 initial mask
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_initialMask.png "Test 2 result")

###### Test 2 refined mask
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_refinedMask.png "Test 2 result")

###### Test 2 contours (Red = Mask contour, Blue = Background pixels, Green = Foreground pixels)
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_contour.png "Test 2 contours")

###### Test 2 result
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out.png "Test 2 result")
