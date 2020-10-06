# Masked-Face-Detection

The main idea is to design a classifier which works on the face object detected. The transfer_classifier.py is to design the classifier which in this case is a VGG16 based transfer learning model.
The facedetector.py actually invokes the opencv-dnn based face detector which isolates the face data. Now on that extracted image section the classifier works to get the required prediction.
