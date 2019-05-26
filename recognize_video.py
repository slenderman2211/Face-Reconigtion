# import các thư viện cần thiết
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import importlib

# tạo các parser và truyền biến vào
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="Đường dẫn tới bộ phát hiện khuôn mặt của openCV")
ap.add_argument("-m", "--embedding-model", required=True,
	help="Đường dẫn tới mô hình nhúng khuôn mặt của OpenCV")
ap.add_argument("-r", "--recognizer", required=True,
	help="Đường dẫn tới mô hình đã được train nhận diện mặt người")
ap.add_argument("-l", "--le", required=True,
	help="Đường dẫn tới bộ mã hóa các nhãn tên")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="xác suất tối thiểu để loại ra các dự đoán yếu")
args = vars(ap.parse_args())

# tiến hành load các mô hình phát hiện khuôn mặt 
print("[INFO] tiến hành load mô hình phát hiện khuôn mặt...")
DNN = "CAFFE"
if DNN == "CAFFE":
    modelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "face_detection_model/deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "face_detection_model/opencv_face_detector_uint8.pb"
    configFile = "face_detection_model/opencv_face_detector.pbtxt"
    detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# tiến hành load các mô hình nhúng nhận diện khuôn mặt
print("[INFO] đang load mô hình nhúng nhận diện khuôn mặt......")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load mô hình nhận diện khuôn mặt đã train từ trước, kèm với bộ mã hóa các nhản tên
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# khởi tạo video stream, kích hoạt webcam
print("[INFO] tiến hành kích hoạt webcam...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# kích hoạt bộ đếm FPS
fps = FPS().start()

# lặp từng khung hình trong video thu được từ webcam
while True:
	# giữ lại từng khung hình
	frame = vs.read()

	# thay đổi kích thường từng khung hình thành 600, rồi thu chiều của từng frame
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# tạo blob từ ảnh
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	#áp dụng bộ phát hiện mặt người deep learning của OpenCV, chuyển input là blob thu được từ bước trên vào trong bộ
	detector.setInput(imageBlob)
	detections = detector.forward()

	# tạo vòng lặp
	for i in range(0, detections.shape[2]):
		# thu chỉ số confidence (xác suất) từ các lần phát hiện
		confidence = detections[0, 0, i, 2]

		# lọc ra các phát hiện yếu
		if confidence > args["confidence"]:
			# tính tọa độ (x,y) của khung khoanh vùng khuôn mặt
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# trích xuất ROI (region of interest) và chiều của ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# đảm bảo chiều cao và rộng của khuôn mặt đủ lớn
			if fW < 20 or fH < 20:
				continue

			# Tạo một blob từ ROI, rồi chuyển nó qua mô hình nhúng khuôn mặt để thu được một vector 128 chiều của mặt
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# tiền hành phân loại để nhận diện khuôn mặt
			preds = recognizer.predict_proba(vec)[0]
			print (preds)
			if all (i < 0.7 for i in preds):
				name = 'unknow'
				proba = 0
			else:
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]

			# vẽ khung khoanh vùng khuôn mặt và hiển thị số xác suất đi kèm.
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# cập nhật lại bộ đếm FPS
	fps.update()

	# hiển thị output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Bấm nút q để dừng
	if key == ord("q"):
		break

# Dừng bộ đếm thời gian và hiển thị FPS 
fps.stop()
print("[INFO] Thời gian quay (tính bằng giây): {:.2f}".format(fps.elapsed()))
print("[INFO] FPS xấp xỉ: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
vs.stop()