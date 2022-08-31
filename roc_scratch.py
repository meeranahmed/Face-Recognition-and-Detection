from cProfile import label
import numpy as np 
from os import path, listdir
from sklearn import preprocessing
import cv2
import matplotlib.pyplot as plt
import PIL
from EigenFaces import *
import pandas as pd

test_data_path = "./testset"
train_data_path = "./trainset"
weightThresh =50


d = []
def load_test_set(test_data_path):
    test_images = []
    true_labels = []
    for folder in listdir(test_data_path):
        d.append(folder)
        folder_path = path.join(test_data_path,folder)
        for file in listdir(folder_path):
            label = folder
            true_labels.append(label)
            img = cv2.imread(path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)  #reading image
            test_images.append(img)

    return test_images, true_labels

test_images, true_labels = load_test_set(test_data_path)
#print(true_labels)

predicted_labels = []
reco = FaceRecognition(train_data_path)
predicted_images =[]
for im in test_images:
    predicted_image, label = reco.recognize_face(im,weightThresh)      
    predicted_labels.append(label)
    predicted_images.append(predicted_image)

#print(true_labels)
#print(predicted_labels)


df = pd.DataFrame([], columns=["index"]+d)
df["index"] = d
df = df.set_index('index')
df = df.fillna(0)
#print(df)

for k in range(len(true_labels)):
    df.at[true_labels[k],predicted_labels[k]] += 1

df['sum'] = df.sum(axis=1)

#print(df)    
#print(precision)


def confusion_mat(class_name):
    tp = df.at[class_name,class_name]
    tn = df.drop(class_name, axis=1).drop(class_name, axis=0)["sum"].sum()
    fp = df.at[class_name,"sum"] - tp
    fn = df[class_name].sum() - tp

    # print("TP = ",tp)
    # print("TN = ",tn)
    # print("FP = ",fp)
    # print("FN = ",fn)
    return tp,tn,fp,fn



tp_list = []
tn_list = []
fp_list = []
fn_list = []


for sample in true_labels:
    tp,tn,fp,fn = confusion_mat(sample)
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

total_tp = sum(tp_list)
total_tn = sum(tn_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
Precision = round(total_tp / (total_tp + total_fp),5)
Recall = round(total_tp / (total_tp + total_fn),5)
FalsePositiveRate = round(total_fp / (total_fp + total_tn),5)
MicroF1 = Precision


print("Total TP =",total_tp)  
print("Total TN = ",total_tn)  
print("Total FP = ",total_fp)  
print("Total FN = ",total_fn)  
print("Precision = ",Precision)
print("Recall = ",Recall)
print("Micro F1 = ",Recall)
print("True Positive Rate = ",Recall)
print("False Positive Rate = ",FalsePositiveRate)


# fp_rates = np.array(fp_rates)
# tp_rates = np.array(tp_rates)

# fp_rates = fp_rates[np.logical_not(np.isnan(fp_rates))]
# tp_rates = tp_rates[np.logical_not(np.isnan(tp_rates))]

# from statistics import mean
import matplotlib.pyplot as plt

x = [0,FalsePositiveRate,1]
y = [0,Recall,1]
 
plt.plot(x, y)
plt.title("ROC (weights threshold = {})".format(weightThresh))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()





# for i in range(len(test_images)):
#     # create figure
#     fig = plt.figure(figsize=(5, 3))
    
#     # setting values to rows and column variables
#     rows = 1
#     columns = 2

#     fig.add_subplot(rows, columns, 1)
#     # showing image
#     plt.imshow(test_images[i],cmap=plt.cm.gray)
#     plt.axis('off')
#     plt.title("Test Image {}".format(true_labels[i]))

#     fig.add_subplot(rows, columns, 2)
    
#     # showing image
#     plt.imshow(predicted_images[i],cmap=plt.cm.gray)
#     plt.axis('off')
#     plt.title("Predicted {}".format(predicted_labels[i]))

#     plt.show()