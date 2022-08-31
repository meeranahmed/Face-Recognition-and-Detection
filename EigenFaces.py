from pickletools import uint8
import numpy as np 
from os import path, listdir
import cv2
from sklearn import preprocessing
import matplotlib.pyplot as plt
import PIL

THIS_FOLDER= path.dirname(path.abspath(__file__))

data_path = "./dataset"

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))



class FaceRecognition():

    def __init__(self, dataSetPath: str):
        self.get_dataset_images(dataSetPath)
        self.get_images_matrix()
        self.get_mean_image_matrix()
        self.get_eigen_vectors()
        #self.eigen_vectors_normalized = preprocessing.normalize(self.eigenvector_C , axis=0)
        self.images_weights = self.eigen_vectors.T @ self.zero_mean_images_matrix
        self.thresh = (np.array([euclidean_distance(self.images_weights[:,i], self.images_weights[:,i+1]) for i in range(self.images_weights.shape[1]-1)],type(float))).max()/2


    def get_dataset_images(self,data_path):
        images = []
        images_names = []
        self.labels = []
        for folder in listdir(data_path):
            folder_path = path.join(data_path,folder)
            for file in listdir(folder_path):
                    label = folder
                    self.labels.append(label)
                    images_names.append(file)
                    img = cv2.imread(path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)  #reading image
                    images.append(img)

        self.dataset_images = np.array(images)
        self.images_number = self.dataset_images.shape[0]
        self.images_names = images_names
        #print(self.images_number)

    def get_images_matrix(self):
        images_matrix = []   
        for img in self.dataset_images:
                flattened = img.flatten()
                if not len(images_matrix):
                    images_matrix = flattened

                else:
                    images_matrix = np.column_stack((images_matrix,flattened))

        self.images_matrix = images_matrix
        


    def get_mean_image_matrix(self):
        self.mean_vector = np.sum(self.images_matrix, axis=1, dtype='float')/(self.images_number)
        mean_vector_matrix = np.tile(self.mean_vector,(self.images_number,1))
        self.zero_mean_images_matrix = self.images_matrix - mean_vector_matrix.T


    def get_eigen_vectors(self):

        cov_mat = ((self.zero_mean_images_matrix.T).dot(self.zero_mean_images_matrix)) / (self.images_number - 1)  #m* n^2

        # Calculating the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        #sort eigen values descendingly 
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        c = np.argwhere(np.cumsum(eigenvalues) / eigenvalues.sum() >= 0.9)[0][0]
        #satisfying_eigen_values = eigenvalues[0:c+1]
        # Sort the eigenvectors according to the highest eigenvalues
        eigenvectors = eigenvectors[:, idx]
        #print(eigenvectors.shape)
        satisfying_eigen_vectors = eigenvectors[:,0:c+1]
        #print(satisfying_eigen_vectors.shape)

        eigenvector_C = self.zero_mean_images_matrix @ satisfying_eigen_vectors    #dimensionality trick
        self.eigen_vectors = preprocessing.normalize(eigenvector_C , axis=0)

    def recognize_face(self,image,weights_thresh):
        zero_mean_image = image.flatten() - self.mean_vector
        weights = self.eigen_vectors.T @ zero_mean_image 
        if weights_thresh > weights.shape[0]:
            weights_thresh = weights.shape[0]
        #print(weights_thresh)
        min_distance = None
        distances = [euclidean_distance(weights[0:weights_thresh,], self.images_weights [0:weights_thresh,i]) for i in range(self.images_weights .shape[1])]
        for i in range(len(distances)):
            if (distances[i] < self.thresh) and (min_distance == None or min_distance > distances[i]):
                min_distance = distances[i]
                min_distance_idx = i
        
        #return self.images_matrix[:,min_distance_idx].reshape(image.shape).astype(np.uint8)
        return self.dataset_images[min_distance_idx], self.labels[min_distance_idx]


def get_images_grid(images,number_of_row_images,number_of_col_images):
    first_image = PIL.Image.fromarray(images[0])
    contact_sheet = PIL.Image.new(
        first_image.mode, (first_image.width*number_of_row_images, first_image.height*number_of_col_images))
    x = 0
    y = 0

    for img in images:
        # paste the current image into the contact sheet
        img = PIL.Image.fromarray(img)
        contact_sheet.paste(img, (x, y))
        # Now we update our X position. If it is going to be the width of the image, then we set it to 0
        # and update Y as well to point to the next "line" of the contact sheet.
        if x+first_image.width == contact_sheet.width:
            x = 0
            y = y+first_image.height
        else:
            x = x+first_image.width
    # resize and display the contact sheet
    contact_sheet = contact_sheet.resize(
        (int(contact_sheet.width/2), int(contact_sheet.height/2)))
    return np.asarray(contact_sheet)




# test_img = cv2.imread("4.png", cv2.IMREAD_GRAYSCALE)
# reco = FaceRecognition(data_path)

# #imageGrid = get_images_grid(reco.dataset_images,20,20)
# # plt.imshow(imageGrid, cmap=plt.cm.gray)
# # plt.tight_layout()
# # plt.show()

# face = reco.recognize_face(test_img,3)


# # create figure
# fig = plt.figure(figsize=(5, 3))
  
# # setting values to rows and column variables
# rows = 1
# columns = 2

# fig.add_subplot(rows, columns, 1)
# # showing image
# plt.imshow(test_img,cmap=plt.cm.gray)
# plt.axis('off')
# plt.title("Test Image")

# fig.add_subplot(rows, columns, 2)
  
# # showing image
# plt.imshow(face,cmap=plt.cm.gray)
# plt.axis('off')
# plt.title("Predicted")

# plt.show()

