from mainwindow import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np
import sys
from FaceDetection import *
from EigenFaces import *

class App(QtWidgets.QMainWindow):
    
    Detect_path = ""
    Test_path = ""
 
    
    
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.ImportButton.clicked.connect(self.load_data)
        self.ui.TestImageButton.clicked.connect(self.load_data)
        self.ui.DetectionButton.clicked.connect(self.Apply_FaceDetection)
        self.ui.RecogintionButton.clicked.connect(self.Apply_FaceRecognition)
        

        
     
        
    def load_data(self):
        filepath = QFileDialog.getOpenFileName(self)
        if filepath[0]:
            self.Detect_path = filepath[0]
            self.Test_path = filepath[0]

        if self.ui.tabWidget.currentIndex() == 0 :
            img = cv2.imread(self.Detect_path)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            self.ui.Import.show()
            self.ui.Import.setImage(np.rot90(img,1))
        else:
            img = cv2.imread(self.Test_path)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            self.ui.TestImage.show()
            self.ui.TestImage.setImage(np.rot90(img,1))

    def Apply_FaceDetection (self):

        img = cv2.imread(self.Detect_path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detect_faces(img)


        img=draw_faces(img,faces)

        self.ui.Detection.show()
        self.ui.Detection.setImage(np.rot90(img,1))

    def Apply_FaceRecognition (self):

        test_img = cv2.imread(self.Test_path)
        test_img = cv2.cvtColor(test_img , cv2.COLOR_BGR2GRAY)

        THIS_FOLDER= path.dirname(path.abspath(__file__))
        data_path = "./trainset"
        weightsThresh = 100
        reco = FaceRecognition(data_path)
        face,_ = reco.recognize_face(test_img,weightsThresh)

        self.ui.Recognition.show()
        self.ui.Recognition.setImage(np.rot90(face,1))



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()
    


if __name__ == "__main__":
    main()
                                                                                                                                                                                                    