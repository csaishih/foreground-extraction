# Object Segmentation For Large Images
Object segmentation has a wide number of uses in computer vision. As technology advances, the resolution of your everyday photo also increases. This is problematic for older traditional object segmentation methods as the computation time increases tremendously when working with high resolution images.  

This repo proposes an object segmentation algorithm that reduces the time needed to process high resolution images while still achieving a decent result. Inspiration of algorithm came from [here](http://graphicsinterface.org/wp-content/uploads/gi2015-11.pdf).


## How it works
The algorithm takes in three images as input, the source image, the foreground seed image and the background seed image. The seed images give the algorithm a very basic guideline when decided whether a pixel is part of the foreground or background.  

To handle the resolution of the image in a reasonable time, we construct an image pyramid by downsampling the source image to create a low resolution version of the image. We use the traditional [grabcut](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) algorithm to get a rough foreground mask of the source image and upsample the mask back to the source image's original resolution. This mask is labelled the initial mask. More information regarding grabcut [here](http://docs.opencv.org/3.1.0/d8/d83/tutorial_py_grabcut.html).  

Using the initial mask, we can generate improved foreground and background seeds using the contour of the mask. We proceed to examine patches along the contour to get a more refined cut. Each patch will contain pixels that are already marked by the improved foreground and background seeds and the cut performed will be based off of the color and location of each pixel within the patch. The algorithm assumes that if a pixel has a similar color intensity as the cluster of seeded foreground pixels within the patch, or if it is located close to the cluster, then it should considered as a foreground pixel. The same logic applies for background pixels. We speed up processing by spawning four threads process multiple patches concurrently.  

Once each patch is refined, its intensity values are copied back onto an eroded version of the initial mask to create the refined mask. The source image is then cut out using the refined mask.  

## Running the script
###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -f [path to foreground seed image] -b [path to background seed image] -o [path to output image]`
###### Example usage
`python run.py -s test_images/test2/source.jpg -f test_images/test2/foreground.png -b test_images/test2/background.png -o test_images/test2/out.png`

## Results
#### Tests done on 2.7 GHz Quad Core i7 MacBook
###### Test 2 source (3024 × 4032) Taken with iPhone 7
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/source.jpg "Test 2 source")

###### Test 2 unoptimized result (run time: 74.0391 seconds)
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out.png "Test 2 unoptimized result")

###### Test 2 optimized result (run time: 3.6541 seconds)
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out.png "Test 2 optimized result")

###### Test 2 foreground seed
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/foreground.png "Test 2 foreground seed")

###### Test 2 background seed
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/background.png "Test 2 background seed")

###### Test 2 initial mask
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_initialMask.png "Test 2 result")

###### Test 2 refined mask
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_refinedMask.png "Test 2 result")

###### Test 2 contours (Red = Initial mask contour, Blue = Background pixels, Green = Foreground pixels)
![alt text](https://github.com/g3aishih/object-segmentation/blob/master/test_images/test2/out_contour.png "Test 2 contours")
