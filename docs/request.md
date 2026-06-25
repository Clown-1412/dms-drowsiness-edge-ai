Bạn là kỹ sư AI/Computer Vision đang phát triển repo Python cho hệ thống DMS phát hiện ngủ gật người lái xe.

Repo hiện tại đã triển khai đến Feature Extraction Layer và có demo:

Camera
→ Pre-processing
→ Landmark Detection
→ Feature Extraction

Hiện tại demo đã chạy được, nhưng cần điều chỉnh 2 vấn đề sau:

1. Bảng thông tin overlay đang che phần lớn khuôn mặt.
2. Logic `eye_closed` đang bị quá nhạy: chỉ cần 1 mắt nhắm thì `eye_closed=True`. Với hệ thống phát hiện ngủ gật, điều này chưa hợp lý. `eye_closed` nên chỉ True khi cả hai mắt đều nhắm.

Nhiệm vụ:
Hãy chỉnh code hiện tại theo yêu cầu dưới đây, không phá vỡ các layer đã chạy ổn.

==================================================
PHẦN 1: SỬA LOGIC EYE FEATURES
==============================

File cần kiểm tra và chỉnh:

src/features/eye_features.py

Hiện tại nếu logic đang là:

eye_closed = left_EAR < threshold or right_EAR < threshold

thì hãy sửa lại.

Yêu cầu mới:

* Tính riêng:

  * left_eye_closed
  * right_eye_closed
  * both_eyes_closed
  * one_eye_closed
  * eye_closed

Logic bắt buộc:

left_eye_closed = left_EAR < nguong_mat_nham
right_eye_closed = right_EAR < nguong_mat_nham

both_eyes_closed = left_eye_closed and right_eye_closed
one_eye_closed = left_eye_closed != right_eye_closed

eye_closed = both_eyes_closed

Giải thích:

* `eye_closed` trong hệ thống DMS nên hiểu là cả hai mắt đều nhắm.
* Nếu chỉ một mắt nhắm thì xem là `one_eye_closed=True`, không kết luận ngủ gật.
* `one_eye_closed` chỉ dùng để debug hoặc đánh dấu khả năng nháy mắt, che khuất một mắt, hoặc landmark lệch.

Output mới của eye_features phải có dạng:

{
"left_EAR": float,
"right_EAR": float,
"avg_EAR": float,

```
"left_eye_closed": bool,
"right_eye_closed": bool,
"both_eyes_closed": bool,
"one_eye_closed": bool,

"eye_closed": bool,

"is_valid": bool,
"error": None hoặc chuỗi lỗi
```

}

Nếu thiếu landmark hoặc EAR không hợp lệ:

* Không được crash.
* Trả is_valid=False.
* Trả error phù hợp.
* Các giá trị bool nên mặc định False.

Ví dụ output khi chỉ mắt trái nhắm:

{
"left_EAR": 0.12,
"right_EAR": 0.29,
"avg_EAR": 0.205,
"left_eye_closed": true,
"right_eye_closed": false,
"both_eyes_closed": false,
"one_eye_closed": true,
"eye_closed": false,
"is_valid": true,
"error": null
}

Ví dụ output khi cả hai mắt nhắm:

{
"left_EAR": 0.13,
"right_EAR": 0.14,
"avg_EAR": 0.135,
"left_eye_closed": true,
"right_eye_closed": true,
"both_eyes_closed": true,
"one_eye_closed": false,
"eye_closed": true,
"is_valid": true,
"error": null
}

==================================================
PHẦN 2: SỬA DEMO OVERLAY KHÔNG CHE MẶT
======================================

File cần kiểm tra và chỉnh:

app/demo_feature_extraction.py

Hiện tại bảng thông tin màu đen đang vẽ ở góc trên trái và che phần lớn khuôn mặt.

Yêu cầu:

* Không vẽ bảng thông tin đè lên vùng mặt.
* Ưu tiên giải pháp đơn giản, ổn định và dễ đọc.
* Có thể chuyển bảng thông tin xuống góc trái dưới hoặc góc phải.
* Không cần làm thuật toán auto-layout quá phức tạp nếu chưa cần.

Cách đề xuất:

1. Viết hoặc sửa hàm vẽ overlay, ví dụ:

ve_thong_tin_dac_trung(frame, ket_qua_dac_trung)

2. Thay vì đặt overlay ở góc trên trái:

x0 = 10
y0 = 10

hãy đặt overlay ở góc trái dưới:

chieu_cao, chieu_rong = frame.shape[:2]
x0 = 10
y0 = chieu_cao - overlay_height - 10

3. Nếu overlay_height lớn hơn chiều cao frame, giảm số dòng hiển thị hoặc giảm font_scale.

4. Overlay nên hiển thị các thông tin chính:

* FPS
* face_detected
* left_EAR
* right_EAR
* avg_EAR
* left_eye_closed
* right_eye_closed
* both_eyes_closed
* one_eye_closed
* eye_closed
* MAR
* mouth_open
* head_pitch
* head_yaw
* head_roll
* head_down
* head_turned

5. Nên giảm kích thước bảng để không chiếm quá nhiều màn hình:

* font_scale khoảng 0.5 đến 0.6
* line_height khoảng 22 đến 26
* padding khoảng 8 đến 12

6. Nếu không thấy mặt:

* Hiển thị "No face detected"
* Không crash khi eye_features hoặc mouth_features là None.

==================================================
GỢI Ý HÀM OVERLAY
=================

Có thể dùng ý tưởng sau:

def ve_thong_tin_dac_trung(frame, ket_qua_dac_trung):
h, w = frame.shape[:2]

```
eye = ket_qua_dac_trung.get("eye_features") or {}
mouth = ket_qua_dac_trung.get("mouth_features") or {}
head = ket_qua_dac_trung.get("head_pose_features") or {}

lines = [
    f"FPS: {ket_qua_dac_trung.get('fps', 0):.1f} | face_detected: {ket_qua_dac_trung.get('face_detected')}",
    f"L_EAR: {eye.get('left_EAR')} | R_EAR: {eye.get('right_EAR')}",
    f"avg_EAR: {eye.get('avg_EAR')} | eye_closed: {eye.get('eye_closed')}",
    f"L_closed: {eye.get('left_eye_closed')} | R_closed: {eye.get('right_eye_closed')}",
    f"both_closed: {eye.get('both_eyes_closed')} | one_closed: {eye.get('one_eye_closed')}",
    f"MAR: {mouth.get('MAR')} | mouth_open: {mouth.get('mouth_open')}",
    f"pitch: {head.get('head_pitch')} | yaw: {head.get('head_yaw')} | roll: {head.get('head_roll')}",
    f"head_down: {head.get('head_down')} | head_turned: {head.get('head_turned')}",
]

# Có thể format float bằng helper để tránh None gây lỗi.
```

Yêu cầu:

* Các giá trị float nên được format 3 chữ số thập phân nếu có giá trị.
* Nếu giá trị None thì hiển thị "None" hoặc "N/A".
* Không để demo crash vì format None.

==================================================
PHẦN 3: KIỂM TRA SAU KHI SỬA
============================

Sau khi chỉnh code, hướng dẫn chạy:

python app/demo_feature_extraction.py --source 0

Test thủ công:

1. Mở webcam và nhìn thẳng:

* face_detected=True
* eye_closed=False
* both_eyes_closed=False
* one_eye_closed=False

2. Nhắm một mắt:

* left_eye_closed hoặc right_eye_closed=True
* one_eye_closed=True
* both_eyes_closed=False
* eye_closed=False

3. Nhắm cả hai mắt:

* left_eye_closed=True
* right_eye_closed=True
* both_eyes_closed=True
* eye_closed=True

4. Mở miệng:

* MAR tăng
* mouth_open có thể chuyển True nếu vượt ngưỡng

5. Kiểm tra overlay:

* Bảng thông tin không che phần trên khuôn mặt.
* Face mesh vẫn nhìn rõ.
* Nếu không thấy mặt, chương trình không crash.

==================================================
YÊU CẦU CODE STYLE
==================

* Giữ phong cách đặt tên tiếng Việt trong repo.
* Không đổi tên class cũ nếu không cần.
* Không thay đổi pipeline lớn.
* Không thêm ML, PERCLOS, Temporal Buffer ở bước này.
* Chỉ sửa logic eye feature và cách hiển thị overlay trong demo.
* Code phải dễ đọc, có comment ngắn gọn.
* Không in log quá nhiều mỗi frame.

==================================================
OUTPUT CUỐI CÙNG CẦN TRẢ VỀ
===========================

Sau khi hoàn thành, hãy tóm tắt:

1. Đã sửa logic `eye_closed` như thế nào.
2. Đã thêm các feature mới nào:

   * left_eye_closed
   * right_eye_closed
   * both_eyes_closed
   * one_eye_closed
3. Đã chỉnh overlay để không che mặt như thế nào.
4. Cách chạy demo.
5. Lưu ý: ngưỡng `nguong_mat_nham` có thể cần tinh chỉnh theo camera, ánh sáng và khoảng cách mặt.
