import cv2
import numpy as np
import tensorflow as tf
import keras
import dlib

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.33)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
model = tf.keras.models.load_model("finalmodelvggnetindian3.h5")
modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile,modelFile)
face_casecade = cv2.CascadeClassifier("haarcascades/haarcascade_eye_tree_eyeglasses.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
test = []
while(True):
	try:
		ret,frame = cap.read()
		h,w = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
		net.setInput(blob)
		detections = net.forward()
		text = " "
		for i in range(0,detections.shape[2]):
			confidence = detections[0][0][i][2]
			if confidence > 0.5:
				box = detections[0][0][i][3:7] * np.array([w,h,w,h])
				(startX,startY,endX,endY) = box.astype(int)
				if (endX - startX)*(endY - startY) > 500:
					roi = frame[startY:endY,startX:endX]
					roi = cv2.resize(roi,(224,224))
					roi = roi.astype(float)
					roi = np.true_divide(roi,255)
					test.append(roi)
					if len(test) == 2:
						y_pred = np.argmax(model.predict(np.array(test)),axis =1 )
						if(np.sum(y_pred) == 0):
							text ="MASKED"
						else:
							text = "NOT MASKED"
						test = []
					cv2.imwrite("face.jpeg",roi)
					color = (255 , 0 , 0)
					stroke = 2
					cv2.rectangle(frame ,(startX,startY),(endX,endY),color,stroke)
					cv2.rectangle(frame ,(0,0),(325,75),(255,255,255),-1)
					font = cv2.FONT_HERSHEY_SIMPLEX 
					org = (50, 50)  
					fontScale = 1
					color = (0,0,0) 
					thickness = 2
					cv2.putText(frame,text,org,font,fontScale,color,thickness)
		cv2.imshow("frame",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except Exception as e:
		pass

cap.release()
cv2.destroyAllWindows()

	# gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# faces = face_casecade.detectMultiScale(gray_frame , scaleFactor = 1.5 , minNeighbors = 5)
	# for x,y,w,h in faces:
	# 	roi = frame[y:y+h,x:x+w]
	# 	# roi = cv2.resize(roi,(224,224))
	# 	# roi = roi.astype(float)
	# 	# roi = np.true_divide(roi,255)
	# 	# test.append(roi)
	# 	# if len(test) == 8:
	# 	# 	y_pred = np.argmax(model.predict(np.array(test)),axis =1 )
	# 	# 	if(np.sum(y_pred) > 0):
	# 	# 		print("MASKED")
	# 	# 	else:
	# 	# 		print("NOT MASKED")
	# 	# 	test = []
	# 	cv2.imwrite("face.jpeg",roi)
	# 	color = (255 , 0 , 0)
	# 	stroke = 2
	# 	cv2.rectangle(frame ,(x-10,y-10),(x+w+10,y+h+10),color,stroke)
	# cv2.imshow("frame",frame)
	