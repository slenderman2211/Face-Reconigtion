# import các thư viện cần thiết
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# tạo các parser và truyền biến vào
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="dường dẫn tới các file pickle chứa các hình nhúng mã hóa tạo ra từ extract_embeddings.py")
ap.add_argument("-r", "--recognizer", required=True,
	help="dường dẫn để xuất mô hình SVM đã được train nhận diện khuôn mặt ")
ap.add_argument("-l", "--le", required=True,
	help="dường dẫn để xuất file pickle chứa các nhãn tên đã được mã hóa")
args = vars(ap.parse_args())

# tiến hành load các hình nhúng từ file pickle
print("[INFO] load các hình nhúng khuôn mặt...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# mã hóa các nhãn
print("[INFO] mã hóa các nhãn tên...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# huấn luyện mô hình bằng các hình nhúng 128 chiều của các khuôn mặt
print("[INFO] huấn luyện mô hình...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# lưu bộ nhận diện khuôn mặt đã train xong vào máy thành file pickle
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# lưu bộ mã hóa nhãn tên thành file pickle
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()