# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:

Import the necessary libraries and read the original image and save it as a image variable.
<br>

### Step2:

Translate the image using M=np.float32([[1,0,20],[0,1,50],[0,0,1]]) translated_img=cv2.warpPerspective(input_img,M,(cols,rows)) 
<br>

### Step3:

Scale the image using M=np.float32([[1.5,0,0],[0,2,0],[0,0,1]]) scaled_img=cv2.warpPerspective(input_img,M,(cols,rows)) 
<br>

### Step4:

Shear the image using M_x=np.float32([[1,0.2,0],[0,1,0],[0,0,1]]) sheared_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows)) 
<br>

### Step5:

Reflection of image can be achieved through the code M_x=np.float32([[1,0,0],[0,-1,rows],[0,0,1]]) reflected_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows)) 
<br>

### Step6:

Rotate the image using angle=np.radians(45) M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]) rotated_img=cv2.warpPerspective(input_img,M,(cols,rows)) 
<br>

### Step7:

Crop the image using cropped_img=input_img[20:150,60:230] 
<br>

### Step8:

Display all the Transformed images and end the program. 
<br>

## Program:
```python
Developed By: Palamakula Deepika
Register Number: 212221240035
i)Image Translation

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("monkey.png")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M= np.float32([[1, 0, 100],
                [0, 1, 200],
                 [0, 0, 1]])
translatedImage =cv2.warpPerspective (inputImage, M, (cols, rows))
plt.imshow(translatedImage)
plt.show()

ii) Image Scaling

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("monkey.png")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M = np. float32 ([[1.5, 0 ,0],
                 [0, 1.8, 0],
                  [0, 0, 1]])
scaledImage=cv2.warpPerspective(inputImage, M, (cols * 2, rows * 2))
plt.imshow(scaledImage)
plt.show()

iii)Image shearing

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("monkey.png")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixX = np.float32([[1, 0.5, 0],
                      [0, 1 ,0],
                      [0, 0, 1]])

matrixY = np.float32([[1, 0, 0],
                      [0.5, 1, 0],
                      [0, 0, 1]])
shearedXaxis = cv2.warpPerspective (inputImage, matrixX, (int(cols * 1.5), int (rows * 1.5)))
shearedYaxis = cv2.warpPerspective (inputImage, matrixY, (int (cols * 1.5), int (rows * 1.5)))
plt.imshow(shearedXaxis)
plt.show()
plt.imshow(shearedYaxis)
plt.show()

iv)Image Reflection

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("monkey.png")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixx=np.float32([[1, 0, 0],
                    [0,-1,rows],
                    [0,0,1]])
matrixy=np.float32([[-1, 0, cols],
                    [0,1,0],
                    [0,0,1]])
reflectedX=cv2.warpPerspective(inputImage, matrixx, (cols, rows))
reflectedY=cv2.warpPerspective(inputImage, matrixy, (cols, rows))
plt.imshow(reflectedY)
plt.show()

v)Image Rotation

import numpy as np
import cv2
angle=np.radians(45)
inputImage=cv2.imread("monkey.png")
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.sin(angle),np.cos(angle),0],
               [0,0,1]])
rotatedImage = cv2.warpPerspective(inputImage,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotatedImage)
plt.show()

vi)Image Cropping

import numpy as np
import cv2
import matplotlib.pyplot as plt
angle=np.radians(45)
inputImage=cv2.imread("monkey.png")
CroppedImage= inputImage[20:150, 60:230]
plt.axis('off')
plt.imshow(CroppedImage)
plt.show()

```
## Output:
### i)Image Translation
<br>
<img width="219" alt="e1" src="https://user-images.githubusercontent.com/94154679/167078070-69b25ea7-9069-4364-879f-1688c37fd221.png">

<br>

### ii) Image Scaling
<br>
<img width="230" alt="e2" src="https://user-images.githubusercontent.com/94154679/167078086-2362df2e-2918-4f0d-9499-1d0616d7bad6.png">

<br>


### iii)Image shearing
<br>
<img width="191" alt="e3" src="https://user-images.githubusercontent.com/94154679/167078097-3ea9a2ec-2789-4c7a-b7ee-35b410a27cc4.png">

<br>
<br>
<br>


### iv)Image Reflection
<br>
<img width="230" alt="e4" src="https://user-images.githubusercontent.com/94154679/167078108-dfac7971-f2d0-4c75-91a4-abef7950d8c4.png">

<br>
<br>
<br>



### v)Image Rotation
<br>
<img width="200" alt="e5" src="https://user-images.githubusercontent.com/94154679/167078120-8ccec91c-5ac2-4318-8b6a-892eacf3c29c.png">

<br>
<br>
<br>



### vi)Image Cropping
<br>
<img width="299" alt="e6" src="https://user-images.githubusercontent.com/94154679/167078136-f9e735b8-a4ed-4a27-a58b-8ae66c8ab150.png">

<br>
<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
