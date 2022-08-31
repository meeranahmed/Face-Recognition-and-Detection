# Computer Vision Course - Group Task #5

---

| Member Name    | Sec     | BN  
| ------------- | ------------- | --------    |
| Ahmed khaled hilal | 1         |   3   |
| Dalia lotfy Abdulhay | 1        | 30   |
| Radwa Saeed Mohammady | 1        | 33   |
| Meeran Ahmed Mostafa | 2        | 34  |
| Yousef Samir | 2       | 49   |
| **Group NO.**    |  | 8   |
**Contact mail** : dalialotfy289@gmail.com
------
# Table of Content

| Requiered Part | Title |
| ----------- | ----------- |
| [#Part 1](#part-1) | Face Detection |
| [#Part 2](#part-2) | Face Recognition |
| [#Part 3](#part-3) | ROC |



---

# Part 1

## Face Detection
We used Open-Cv to detect faces in images (grayscale or rgb),by unsing pre-trained cascaded classifier
Results:
![](outputs/13.png)


---


# Part 2 

## Face Recognition

---
### introduction

The objective of this project is to highlight the importance of linear algebra in the field of computer vision and face recognition. Eigenface is the name of a set of eigenvectors computed from an image dataset. Eigenvectors is a set of features which characterize the global variation among face images.The basis of the eigenfaces method is the Principal Component Analysis (PCA).PCA is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

### Algorithm

* first if all we Obtain face image and represent every image in a n^2 x m matrix Image.
* Compute the Mean face vector(m).
* Then subtract each image with mean.
* Compute the eigen vectors(v).
* Select eigenvectors(k).
* Now project new image into (k).
* The new image will be represented using the eigenvectors(x).
* Face Detection
    * Subtract x with m.
    * If the difference is lower than a chosen threshold, the new image face is detected.
* Face Recognitio.
    * Each image is represented using the eigenvectors.
    * Each image is then subtracted with x.
    * If the difference is lower than a chosen threshold, the new image face is classified to a class.





Mean image.

   ![](outputs/7.png)

Each image when subtracted with the mean image.

  ![](outputs/8.png)

Each image represented with eigenvectors.

  ![](outputs/6.png)



---
# Part 3

## ROC
   ![](outputs/10.png)
   
   ![](outputs/11.png)
