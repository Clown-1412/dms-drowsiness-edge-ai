Bạn là kỹ sư AI/Computer Vision đang xây dựng hệ thống phát hiện ngủ gật của người lái xe.

Hãy thiết kế và triển khai lớp thứ 3 trong pipeline tổng của hệ thống Driver Monitoring System:

Landmark Detection Layer

Lớp này nhận ảnh đã xử lý từ Frame Pre-processing Layer và thực hiện phát hiện khuôn mặt cùng các điểm đặc trưng trên mặt.

Pipeline hiện tại:

Input Camera Layer
↓
Frame Pre-processing Layer
↓
Landmark Detection Layer

==============================
LANDMARK DETECTION LAYER
========================

Chức năng chính:

* Nhận processed_frame từ Frame Pre-processing Layer.
* Phát hiện khuôn mặt tài xế.
* Phát hiện các điểm landmark trên khuôn mặt.
* Tách riêng landmark của mắt, miệng, mũi và các điểm phục vụ head pose.
* Trả về trạng thái có phát hiện được mặt hay không.
* Nếu không phát hiện được mặt, trả về face_detected = False.

Thư viện đề xuất:

* MediaPipe Face Mesh
* OpenCV

Lý do dùng MediaPipe Face Mesh:

* Dễ triển khai.
* Có sẵn 468 điểm landmark trên khuôn mặt.
* Phù hợp để trích xuất mắt, miệng, mũi và tư thế đầu.
* Có thể dùng tiếp cho các lớp sau như EAR, MAR, PERCLOS, head pose.

==============================
INPUT
=====

Input của Landmark Detection Layer gồm:

{
"processed_frame": processed_frame,
"timestamp": timestamp,
"fps": fps
}

Trong đó:

* processed_frame: ảnh đã resize, giảm nhiễu và chuyển sang RGB.
* timestamp: thời điểm frame được đọc.
* fps: tốc độ khung hình hiện tại.

==============================
OUTPUT BẮT BUỘC
===============

Output của lớp này bắt buộc gồm:

{
"face_detected": true/false,
"face_landmarks": face_landmarks,
"eye_landmarks": eye_landmarks,
"mouth_landmarks": mouth_landmarks,
"nose_landmarks": nose_landmarks,
"head_points": head_points
}

Ý nghĩa từng output:

1. face_detected

* Kiểu dữ liệu: boolean.
* true nếu phát hiện được khuôn mặt.
* false nếu không phát hiện được khuôn mặt.

2. face_landmarks

* Chứa toàn bộ landmark khuôn mặt.
* Nếu dùng MediaPipe Face Mesh, đây là danh sách 468 điểm landmark.

3. eye_landmarks

* Chứa các điểm landmark của mắt trái và mắt phải.
* Dùng cho lớp sau để tính EAR và phát hiện nhắm mắt.

4. mouth_landmarks

* Chứa các điểm landmark của miệng.
* Dùng cho lớp sau để tính MAR và phát hiện ngáp.

5. nose_landmarks

* Chứa các điểm landmark vùng mũi.
* Dùng cho head pose hoặc định hướng khuôn mặt.

6. head_points

* Chứa các điểm quan trọng để ước lượng tư thế đầu.
* Có thể gồm:

  * mũi
  * cằm
  * khóe mắt trái
  * khóe mắt phải
  * khóe miệng trái
  * khóe miệng phải

==============================
TRƯỜNG HỢP KHÔNG THẤY MẶT
=========================

Nếu không phát hiện được khuôn mặt:

{
"face_detected": false,
"face_landmarks": None,
"eye_landmarks": None,
"mouth_landmarks": None,
"nose_landmarks": None,
"head_points": None
}

Không được để chương trình bị lỗi khi không thấy mặt.

Cần xử lý rõ ràng trường hợp:

* Không có khuôn mặt trong frame.
* Khuôn mặt bị che.
* Frame bị lỗi.
* processed_frame = None.

==============================
GỢI Ý LANDMARK INDEX
====================

Nếu dùng MediaPipe Face Mesh, hãy khai báo các nhóm index cơ bản.

Mắt trái:

* LEFT_EYE_INDEXES

Mắt phải:

* RIGHT_EYE_INDEXES

Miệng:

* MOUTH_INDEXES

Mũi:

* NOSE_INDEXES

Điểm phục vụ head pose:

* HEAD_POSE_INDEXES

Có thể dùng các index phổ biến của MediaPipe Face Mesh, ví dụ:

* Mũi: 1
* Cằm: 152
* Mắt trái: 33, 133
* Mắt phải: 362, 263
* Miệng trái/phải: 61, 291

==============================
GỢI Ý THIẾT KẾ CLASS
====================

Tạo class:

LandmarkDetector

Các hàm nên có:

* **init**()
* detect()
* extract_eye_landmarks()
* extract_mouth_landmarks()
* extract_nose_landmarks()
* extract_head_points()
* release()

Hàm chính:

detect(processed_frame, timestamp=None, fps=None)

Output của hàm detect() là dictionary:

{
"face_detected": face_detected,
"face_landmarks": face_landmarks,
"eye_landmarks": eye_landmarks,
"mouth_landmarks": mouth_landmarks,
"nose_landmarks": nose_landmarks,
"head_points": head_points,
"timestamp": timestamp,
"fps": fps
}

==============================
YÊU CẦU FILE
============

Hãy tạo hoặc cập nhật cấu trúc project như sau:

dms_pipeline/
│
├── app/
│   └── demo_landmark_detection.py
│
├── src/
│   ├── camera/
│   │   └── camera_stream.py
│   │
│   ├── preprocessing/
│   │   └── frame_preprocessor.py
│   │
│   ├── detection/
│   │   └── landmark_detector.py
│   │
│   └── utils/
│       └── fps.py
│
└── requirements.txt

==============================
YÊU CẦU DEMO
============

File app/demo_landmark_detection.py cần:

* Mở webcam.
* Đọc frame từ Input Camera Layer.
* Xử lý frame qua Frame Pre-processing Layer.
* Đưa processed_frame vào Landmark Detection Layer.
* Hiển thị frame lên màn hình.
* Nếu phát hiện được mặt:

  * Vẽ landmark khuôn mặt hoặc ít nhất vẽ các điểm mắt, miệng, mũi.
  * In ra console:

    * face_detected = True
    * số lượng face_landmarks
    * số lượng eye_landmarks
    * số lượng mouth_landmarks
* Nếu không phát hiện được mặt:

  * In ra console: face_detected = False
  * Hiển thị chữ "No face detected" trên frame.
* Nhấn phím q để thoát.
* Khi thoát phải release camera và destroyAllWindows.

==============================
YÊU CẦU CODE
============

Code cần:

* Viết bằng Python.
* Dùng OpenCV và MediaPipe.
* Có comment rõ ràng bằng tiếng Việt.
* Có xử lý lỗi khi không có mặt.
* Không được làm chương trình crash khi frame lỗi.
* Dễ mở rộng cho các lớp sau:

  * Feature Extraction Layer
  * EAR calculation
  * MAR calculation
  * Head Pose Estimation
  * Temporal Buffer
  * Hybrid Decision Layer

==============================
OUTPUT MONG MUỐN
================

Hãy tạo đầy đủ hoặc cập nhật code cho các file sau:

1. src/detection/landmark_detector.py
2. app/demo_landmark_detection.py
3. Cập nhật requirements.txt nếu cần

Ngoài code, hãy giải thích ngắn gọn:

* Vai trò của Landmark Detection Layer.
* Input của lớp này.
* Output của lớp này.
* Cách xử lý khi không phát hiện được mặt.
* Vì sao output này phù hợp để đưa sang Feature Extraction Layer.

Chỉ tập trung vào Landmark Detection Layer, chưa triển khai EAR, MAR, PERCLOS hoặc Decision Layer.
