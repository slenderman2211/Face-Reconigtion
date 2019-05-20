import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="đường dẫn tới hình cần nhận diện")
ap.add_argument("-d", "--detector", required=True,
	help="Đường dẫn tới bộ phát hiện khuôn mặt của openCV")
ap.add_argument("-m", "--embedding-model", required=True,
	help="Đường dẫn tới mô hình nhúng khuôn mặt của OpenCV")
ap.add_argument("-r", "--recognizer", required=True,
	help="Đường dẫn tới mô hình đã được train nhận diện mặt người")
ap.add_argument("-l", "--le", required=True,
	help="Đường dẫn tới bộ mã hóa các nhãn tên")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="xác suất tối thiểu để loại ra các dự đoán yếu")
args = vars(ap.parse_args())

print("[INFO] tiến hành load mô hình phát hiện khuôn mặt...")
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "face_detection_model/deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "face_detection_model/opencv_face_detector_uint8.pb"
    configFile = "face_detection_model/opencv_face_detector.pbtxt"
    detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

print("[INFO] đang load mô hình nhúng nhận diện khuôn mặt......")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]


imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
	
	confidence = detections[0, 0, i, 2]

	
	if confidence > args["confidence"]:
	
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		if fW < 20 or fH < 20:
			continue

		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)