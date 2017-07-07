# Object Segmentation For Large Images
Image inpainting can remove objects in a photo and replace them with believable textures.  
The research behind the algorithm can be found [here](../master/criminisi_tip2004.pdf)


## Visualization of the algorithm
The red square depicts the patch of the image that the algorithm is currently filling.  
The green square depicts the patch that will be used to fill the red square.  
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/algo_animation.gif "Inpainting visualization")

## Running the script
###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -f [path to foreground seed image] -b [path to background seed image] -o [path to output image]`
###### Example usage
`python run.py -s test_images/test2/source.jpg -f test_images/test2/foreground.png -b test_images/test2/background.png -o test_images/test2/out.png`

## Results
###### Test 2 source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/source.png "Test 2 source")

###### Test 2 mask
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/mask.png "Test 2 mask")

###### Test 2 result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/out.png "Test 2 result")


###### Test 5 source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/source.png "Test 5 source")

###### Test 5 mask
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/mask.png "Test 5 mask")

###### Test 5 result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/out.png "Test 5 result")
