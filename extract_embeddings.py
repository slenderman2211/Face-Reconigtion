# import các thư viện cần thiết
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# tạo các parser và truyến biến vào 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="dường dẫn tới dataset lưu hình")
ap.add_argument("-e", "--embeddings", required=True,
	help="đường dẫn tới nơi xuất output là các ảnh đã được xử lý nhúng (embedded)")
ap.add_argument("-d", "--detector", required=True,
	help="đường dẫn tới bộ nhận diện khuôn mặt của openCV")
ap.add_argument("-m", "--embedding-model", required=True,
	help="dường dẫn tới mô hình nhúng khuôn mặt của OpenCV")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="xác suất tối thiểu để loại ra các dự đoán không chắc chắn (mặc định là 0.5)")
args = vars(ap.parse_args())

# tiến hành load các mô hình phát hiện khuôn mặt 
print("[INFO] đang load mô hình phát hiện khuôn mặt...")
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "face_detection_model/deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "face_detection_model/opencv_face_detector_uint8.pb"
    configFile = "face_detection_model/opencv_face_detector.pbtxt"
    detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# tiến hành load các mô hình embedding khuôn mặt
print("[INFO] đang load mô hình embedding khuôn mặt...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# dường dẫn tới các file ảnh trong dataset
print("[INFO] Tiến hành xử lý các khuôn mặt...")
imagePaths = list(paths.list_images(args["dataset"]))

# Tạo 2 list chứa các khuôn mặt đã được nhúng và các nhãn tên đi kèm
knownEmbeddings = []
knownNames = []

# Khởi tạo tổng số mặt đã qua xử lý
total = 0

# tạo vòng lặp trong đường dẫn tới nơi chứa hình
for (i, imagePath) in enumerate(imagePaths):
	# lọc tên người ra từ tên file
	print("[INFO] xử lý {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load và thay đổi kích thước hình có độ rộng là 600 pixel (vẫn giữ nguyên tỷ lệ khung hình), 
	# sau đó lấy số chiều của hình
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# tạo một blob từ các hình
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# áp dụng bộ phát hiện mặt người deep learning của OpenCV, chuyển input là blob thu được từ bước trên vào trong detector
	detector.setInput(imageBlob)
	detections = detector.forward()

	# đảm bảo ít nhất phát hiện 1 khuôn mặt
	if len(detections) > 0:
		# giả sử mỗi hình chỉ có một khuôn mặt, tiến hành tìm khung khoanh vùng nào có xác suất cao nhất
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# đảm bảo phát hiện nào có xác suất cao nhất được chọn làm ngưỡng kiểm tra xác suất thấp nhất
		# để có thể lọc các phần phát hiện yếu
		if confidence > args["confidence"]:
			# tính tọa độ (x,y) của khung khoanh vùng khuôn mặt
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# trích xuất ROI (region of interest) và chiều của ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# đảm bảo chiều cao và rộng của khuôn mặt đủ lớn
			if fW < 20 or fH < 20:
				continue

			# tạo một blob từ ROI, rồi chuyển nó qua mô hình nhúng khuôn mặt để thu được một vector 128 chiều của mặt
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# thêm tên người vào đúng từng hình nhúng cụ thể
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# lưu toàn bộ các hình nhúng + tên vào file đuôi pickle
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()