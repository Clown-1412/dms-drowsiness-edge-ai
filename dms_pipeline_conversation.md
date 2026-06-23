# Cuộc trò chuyện: Pipeline hệ thống phát hiện ngủ gật người lái xe

**Ngày tạo file:** 2026-06-23  
**Nội dung:** Lưu lại nội dung trao đổi trong cuộc trò chuyện hiện tại về pipeline DMS, hệ thống thực tế trên xe, cải tiến Decision Layer bằng ML/DL, lựa chọn kết hợp Rule-based/ML, và pipeline cuối cùng cho đồ án.

> Lưu ý: File này ghi lại nội dung chính của cuộc trò chuyện hiện tại. Các cuộc trò chuyện cũ khác không được chép nguyên văn, chỉ những nội dung đã trao đổi trong phiên hiện tại được lưu lại.

---

## 1. User

Tạo pipeline cho hệ thống phát hiện ngủ gật của người lái xe

---

## 2. Assistant

Dưới đây là **pipeline thực tế** cho hệ thống phát hiện ngủ gật của người lái xe, phù hợp để triển khai bằng **Python + OpenCV + MediaPipe/YOLO**, sau đó có thể chạy trên **PC, Raspberry Pi, Orange Pi, hoặc board có NPU**.

### Pipeline tổng thể

```text
Camera
  ↓
Tiền xử lý ảnh
  ↓
Phát hiện khuôn mặt / mắt / miệng
  ↓
Trích xuất đặc trưng hành vi
  ↓
Tính điểm buồn ngủ
  ↓
Ra quyết định
  ↓
Cảnh báo + lưu log + gửi app mobile
```

### Bước 1: Nhận dữ liệu từ camera

**Input:**

```text
Frame từ webcam / camera hồng ngoại / camera cabin
```

**Xử lý:**

```text
- Đọc video theo thời gian thực
- Resize frame
- Chuyển BGR → RGB nếu dùng MediaPipe
- Tính FPS
```

**Output:**

```text
Ảnh frame đã sẵn sàng để xử lý
```

Ví dụ:

```text
camera_frame
fps
timestamp
```

### Bước 2: Phát hiện khuôn mặt

Có thể dùng:

```text
MediaPipe Face Mesh
YOLO face detection
OpenCV Haar Cascade
RetinaFace
```

Với giai đoạn đầu, nên dùng:

```text
MediaPipe Face Mesh
```

Vì nó cho luôn các điểm landmark mắt, miệng, mũi, cằm.

**Input:**

```text
Frame từ camera
```

**Output:**

```text
face_landmarks
face_box
face_detected = True / False
```

Nếu không phát hiện được mặt:

```text
- Tăng bộ đếm mất mặt
- Nếu mất mặt quá lâu → cảnh báo "không nhìn thấy tài xế"
```

### Bước 3: Trích xuất đặc trưng hành vi

Đây là phần quan trọng nhất.

#### 3.1. Phát hiện nhắm mắt bằng EAR

EAR là viết tắt của:

```text
Eye Aspect Ratio
```

Nó dùng để đo độ mở của mắt.

```text
EAR cao  → mắt mở
EAR thấp → mắt nhắm
```

Công thức ý tưởng:

```text
EAR = chiều cao mắt / chiều rộng mắt
```

**Tín hiệu cần lấy:**

```text
left_eye_EAR
right_eye_EAR
avg_EAR
```

**Logic ban đầu:**

```text
Nếu EAR < 0.20 trong nhiều frame liên tiếp
→ tài xế có dấu hiệu nhắm mắt
```

Ví dụ:

```text
EAR < 0.20 trong 1.5 - 2 giây
→ cảnh báo ngủ gật
```

#### 3.2. Phát hiện ngáp bằng MAR

MAR là:

```text
Mouth Aspect Ratio
```

Nó đo độ mở của miệng.

```text
MAR thấp → miệng đóng
MAR cao  → miệng mở / ngáp
```

**Tín hiệu cần lấy:**

```text
mouth_MAR
yawn_duration
yawn_count
```

**Logic ban đầu:**

```text
Nếu MAR > 0.60 trong khoảng 1 - 2 giây
→ có thể là ngáp
```

Nếu trong 1 phút có nhiều lần ngáp:

```text
→ tăng điểm buồn ngủ
```

#### 3.3. Phát hiện gật đầu bằng head pose

Dùng các điểm landmark như:

```text
mũi
cằm
hai mắt
hai mép miệng
```

Sau đó ước lượng góc đầu:

```text
pitch → cúi/ngẩng đầu
yaw   → quay trái/phải
roll  → nghiêng đầu
```

**Tín hiệu cần lấy:**

```text
head_pitch
head_yaw
head_roll
```

**Logic ban đầu:**

```text
Nếu pitch thay đổi mạnh theo hướng cúi xuống
và giữ trong một khoảng thời gian
→ có thể là gật đầu do buồn ngủ
```

#### 3.4. Phát hiện hướng nhìn

Có thể dùng:

```text
vị trí mắt
vị trí mống mắt
hướng đầu
```

**Tín hiệu:**

```text
looking_forward
looking_left
looking_right
looking_down
eyes_off_road_time
```

Logic:

```text
Nếu tài xế không nhìn đường quá 2 - 3 giây
→ cảnh báo mất tập trung
```

#### 3.5. Tính PERCLOS

PERCLOS là một chỉ số rất quan trọng trong phát hiện buồn ngủ. Nó đo:

```text
Tỷ lệ thời gian mắt nhắm trong một cửa sổ thời gian
```

Ví dụ:

```text
Trong 60 giây gần nhất,
nếu mắt nhắm tổng cộng 18 giây
→ PERCLOS = 30%
```

Logic:

```text
PERCLOS càng cao → càng buồn ngủ
```

Có thể dùng cửa sổ:

```text
30 giây
60 giây
```

### Tầng đánh giá trạng thái tài xế

Sau khi có các tín hiệu:

```text
EAR
MAR
PERCLOS
blink_duration
yawn_count
head_pitch
eyes_off_road_time
```

Ta đưa vào bộ đánh giá.

#### Cách 1: Rule-based — nên dùng cho bản đầu tiên

Dễ làm, dễ debug.

```text
Nếu mắt nhắm > 2 giây
→ DROWSY

Nếu PERCLOS > 0.25
→ DROWSY

Nếu ngáp nhiều + chớp mắt chậm
→ WARNING

Nếu cúi đầu lâu
→ DROWSY

Nếu không nhìn đường > 3 giây
→ DISTRACTED
```

Trạng thái đầu ra:

```text
NORMAL
WARNING
DROWSY
DISTRACTED
NO_FACE
```

#### Cách 2: Machine Learning

Sau khi có dữ liệu log, có thể train model.

**Input cho model:**

```text
EAR_mean
EAR_min
MAR_mean
blink_rate
blink_duration
PERCLOS
yawn_count
head_pitch_mean
head_pitch_variance
eyes_off_road_time
```

**Model có thể dùng:**

```text
Random Forest
SVM
XGBoost
LSTM
1D-CNN
```

Với hệ thống thời gian thực, nên bắt đầu bằng:

```text
Random Forest hoặc XGBoost
```

Vì dễ train, nhẹ, chạy nhanh.

### Pipeline xử lý theo thời gian thực

```text
Mỗi frame:
    1. Đọc ảnh từ camera
    2. Phát hiện mặt
    3. Lấy landmark mắt, miệng, đầu
    4. Tính EAR, MAR, head pose
    5. Cập nhật buffer thời gian
    6. Tính PERCLOS, thời gian nhắm mắt, số lần ngáp
    7. Đánh giá trạng thái
    8. Hiển thị cảnh báo
    9. Ghi log nếu có sự kiện bất thường
```

### Kiến trúc module nên thiết kế

```text
dms-drowsiness-system/
│
├── app/
│   ├── demo_webcam.py
│   ├── main.py
│
├── src/
│   ├── camera/
│   │   └── camera_stream.py
│   │
│   ├── detection/
│   │   ├── face_detector.py
│   │   ├── landmark_detector.py
│   │   └── head_pose.py
│   │
│   ├── features/
│   │   ├── eye_features.py
│   │   ├── mouth_features.py
│   │   ├── perclos.py
│   │   └── gaze_features.py
│   │
│   ├── decision/
│   │   ├── rule_engine.py
│   │   └── drowsiness_classifier.py
│   │
│   ├── alert/
│   │   ├── buzzer.py
│   │   ├── sound_alert.py
│   │   └── mobile_notify.py
│   │
│   ├── logging/
│   │   └── event_logger.py
│   │
│   └── utils/
│       ├── fps.py
│       └── config.py
│
├── data/
│   ├── logs/
│   └── samples/
│
├── models/
│   └── drowsiness_model.pkl
│
└── requirements.txt
```

### Bộ tín hiệu nên ưu tiên

| Mức ưu tiên | Tín hiệu | Ý nghĩa |
|---|---|---|
| Rất cao | EAR | Phát hiện nhắm mắt |
| Rất cao | PERCLOS | Đánh giá buồn ngủ theo thời gian |
| Cao | Blink duration | Thời gian chớp/nhắm mắt |
| Cao | Head pose | Gật đầu, cúi đầu |
| Trung bình | MAR | Phát hiện ngáp |
| Trung bình | Gaze | Mất tập trung, không nhìn đường |
| Thấp hơn | Emotion/fatigue face | Khó ổn định hơn |

### Trạng thái đầu ra

```text
NORMAL
WARNING
DROWSY
DISTRACTED
NO_FACE
```

### Logic cảnh báo đề xuất

```text
NORMAL
    ↓
WARNING nếu:
    - Ngáp nhiều
    - Chớp mắt chậm
    - EAR thấp ngắn hạn
    - Nhìn lệch đường nhiều lần

WARNING
    ↓
DROWSY nếu:
    - Mắt nhắm liên tục > 2 giây
    - PERCLOS cao
    - Gật đầu mạnh
    - Nhiều tín hiệu bất thường cùng lúc

DROWSY
    ↓
Cảnh báo:
    - Âm thanh lớn
    - Rung
    - Đèn LED
    - Gửi thông báo app mobile
    - Lưu video/log sự kiện
```

### Output mẫu

```json
{
  "timestamp": "2026-06-21 10:30:20",
  "face_detected": true,
  "ear": 0.18,
  "mar": 0.42,
  "perclos": 0.31,
  "head_pitch": 22.5,
  "eyes_closed_time": 2.1,
  "yawn_count": 3,
  "state": "DROWSY",
  "alert": true
}
```

### Pipeline MVP nên làm trước

**Giai đoạn 1:**

```text
OpenCV đọc webcam
↓
MediaPipe Face Mesh
↓
Tính EAR
↓
Nếu mắt nhắm > 2 giây → cảnh báo
```

**Giai đoạn 2:**

```text
EAR + MAR + PERCLOS
↓
Phân loại NORMAL / WARNING / DROWSY
```

**Giai đoạn 3:**

```text
Phát hiện cúi đầu / gật đầu
↓
Cảnh báo ngủ gật chính xác hơn
```

**Giai đoạn 4:**

```text
Lưu timestamp
Lưu EAR/MAR/PERCLOS
Lưu trạng thái
Lưu ảnh khi có cảnh báo
```

**Giai đoạn 5:**

```text
Trạng thái tài xế
Thời gian cảnh báo
Ảnh/video sự kiện
Vị trí GPS nếu có
Lịch sử cảnh báo
```

### Pipeline đề xuất cuối cùng lúc đầu

```text
Camera cabin
  ↓
OpenCV capture
  ↓
MediaPipe Face Mesh
  ↓
Landmark extraction
  ↓
Feature extraction:
    - EAR
    - MAR
    - PERCLOS
    - Blink duration
    - Head pose
    - Gaze direction
  ↓
Temporal buffer 30-60 giây
  ↓
Rule-based decision engine
  ↓
State:
    NORMAL / WARNING / DROWSY / DISTRACTED / NO_FACE
  ↓
Alert system:
    - Buzzer
    - Speaker
    - LED
    - Mobile notification
  ↓
Logger:
    - CSV/JSON log
    - Event image
    - Optional video clip
  ↓
Mobile app / Dashboard
```

Kết luận: với dự án hiện tại, nên bắt đầu bằng **pipeline rule-based dùng MediaPipe Face Mesh + EAR + PERCLOS**, sau đó mở rộng sang **MAR, head pose, gaze và machine learning**.

---

## 3. User

pipeline của các hệ thống thực tế dùng trên xe có những gì?

---

## 4. Assistant

Pipeline của **hệ thống phát hiện ngủ gật thực tế trên xe** phức tạp hơn bản demo webcam rất nhiều. Nó không chỉ tính EAR/MAR, mà là một hệ thống **DMS — Driver Monitoring System** kết hợp camera cabin, tín hiệu xe, AI, cảnh báo HMI, tự chẩn đoán lỗi và tích hợp với ADAS.

Một pipeline thực tế thường như sau:

```text
Camera cabin / IR camera / cảm biến xe
        ↓
Image Signal Processing + đồng bộ dữ liệu CAN
        ↓
Phát hiện tài xế / mặt / mắt / đầu / tay
        ↓
Theo dõi gaze, mắt, đầu, mí mắt, tư thế
        ↓
Trích xuất đặc trưng theo thời gian
        ↓
Driver State Estimation
        ↓
Drowsiness / Distraction / Unresponsive Detection
        ↓
Risk Scoring + Context Filtering
        ↓
Cảnh báo âm thanh / hình ảnh / rung
        ↓
Tích hợp ADAS / phanh an toàn / dừng xe nếu cần
        ↓
Lưu sự kiện / chẩn đoán lỗi / cloud fleet nếu là xe thương mại
```

### 1. Lớp cảm biến

Hệ thống thực tế thường có **camera hướng vào người lái**, hay dùng **IR/NIR camera** để hoạt động ban đêm, khi ánh sáng thay đổi, hoặc khi tài xế đeo kính. Một số hệ thống thương mại còn dùng thêm camera phía trước đường, cảm biến vô-lăng, tốc độ xe, tín hiệu làn đường, trạng thái ADAS, dây an toàn, ghế ngồi và cảm biến tay trên vô-lăng.

```text
Nguồn dữ liệu chính:
- Camera cabin IR/NIR
- Đèn hồng ngoại
- Tốc độ xe
- Góc lái
- Lane keeping / lane departure
- Trạng thái ga, phanh
- Trạng thái ADAS
- Dây an toàn
- Cảm biến tay trên vô-lăng
- Camera trước xe, nếu có
```

### 2. Tiền xử lý ảnh

Trên xe thật, camera không đưa thẳng ảnh vào model. Trước đó thường có bước xử lý ảnh:

```text
- Auto exposure
- Chống lóa
- Giảm nhiễu
- Cân bằng sáng
- Đồng bộ timestamp
- Kiểm tra camera bị che
- Kiểm tra thiếu sáng / quá sáng
- Cắt vùng người lái
```

Điểm khác lớn so với bản demo là hệ thống thực tế phải biết khi nào camera **bị che, lỗi, quá tối, quá sáng, không nhìn rõ mặt**.

### 3. Phát hiện và theo dõi tài xế

Pipeline thực tế không chỉ detect mặt một lần. Nó phải **tracking liên tục**.

```text
- Driver detection
- Face detection
- Face tracking
- Eye tracking
- Eyelid tracking
- Pupil / iris tracking
- Head pose tracking
- Mouth tracking
- Hand / body pose tracking, nếu có
```

### 4. Trích xuất đặc trưng hành vi

Ở bản demo, ta hay dùng:

```text
EAR
MAR
PERCLOS
Head pose
```

Nhưng xe thật thường dùng nhiều đặc trưng hơn:

| Nhóm tín hiệu | Ví dụ |
|---|---|
| Mắt | eye closure, eyelid opening, blink rate, blink duration, PERCLOS |
| Hướng nhìn | gaze direction, eyes-on-road, eyes-off-road time |
| Đầu | yaw, pitch, roll, head movement stability |
| Miệng | yawn detection, mouth opening duration |
| Tư thế | cúi người, quay người, nhìn xuống |
| Tay | tay có trên vô-lăng hay không |
| Xe | tốc độ, làn đường, góc lái, ADAS đang bật hay tắt |
| Ngữ cảnh | ngày/đêm, đường cao tốc, đô thị, xe đang chạy hay dừng |

Điểm quan trọng: hệ thống thương mại thường không kết luận ngủ gật chỉ vì **một frame mắt nhắm**. Nó dùng **chuỗi thời gian** vài giây đến vài chục giây.

### 5. Ước lượng trạng thái tài xế

Sau khi có đặc trưng, hệ thống đưa vào tầng **Driver State Estimation**.

Các trạng thái thường có:

```text
ATTENTIVE
VISUALLY_DISTRACTED
DROWSY
MICROSLEEP
UNRESPONSIVE
NO_DRIVER
FACE_NOT_VISIBLE
SYSTEM_DEGRADED
```

### 6. Phát hiện ngủ gật

Phần drowsiness thường không chỉ có “mắt nhắm”. Nó thường kết hợp:

```text
- Eye closure duration
- PERCLOS
- Blink rate giảm bất thường
- Blink duration tăng
- Head nodding
- Gaze mất ổn định
- Phản ứng chậm với cảnh báo
- Lái xe lệch làn / dao động vô-lăng
```

Ví dụ:

```text
Nếu mắt nhắm lâu
+ đầu cúi xuống
+ không phản ứng cảnh báo
+ xe đang chạy tốc độ cao
→ nguy cơ microsleep rất cao
```

### 7. Phát hiện mất tập trung

Hệ thống thực tế còn phát hiện **distraction**, không chỉ ngủ gật.

```text
- Nhìn xuống điện thoại
- Nhìn sang màn hình trung tâm quá lâu
- Quay đầu sang bên
- Nhìn ghế phụ
- Nhìn xuống sàn
- Không nhìn đường trong thời gian dài
```

### 8. Bộ lọc ngữ cảnh

Đây là phần bản demo thường thiếu.

Xe thật phải tránh cảnh báo sai trong các tình huống hợp lý:

```text
- Tài xế nhìn gương chiếu hậu
- Tài xế nhìn gương bên
- Tài xế quan sát khi chuyển làn
- Tài xế nhìn màn hình trong thời gian ngắn
- Xe đang dừng đèn đỏ
- Xe đang lùi
- Xe đang đỗ
```

Vì vậy hệ thống cần biết:

```text
vehicle_speed
turn_signal
steering_angle
gear_position
```

Ví dụ:

```text
Nhìn sang trái 1 giây + bật xi nhan trái
→ có thể là kiểm tra gương, không cảnh báo

Nhìn xuống 4 giây + xe chạy 80 km/h
→ cảnh báo mất tập trung
```

### 9. Tầng cảnh báo HMI

Hệ thống thực tế thường cảnh báo theo nhiều cấp:

```text
Cấp 1: Nhắc nhẹ bằng biểu tượng / âm thanh nhỏ
Cấp 2: Âm thanh rõ hơn
Cấp 3: Rung vô-lăng / rung ghế
Cấp 4: Cảnh báo mạnh
Cấp 5: Nếu ADAS hỗ trợ, xe có thể giảm tốc hoặc dừng an toàn
```

### 10. Tích hợp với ADAS

Trên xe đời mới, DMS không hoạt động riêng lẻ. Nó liên kết với:

```text
- Adaptive Cruise Control
- Lane Keeping Assist
- Lane Centering
- Emergency Braking
- Hands-free driving
- Minimal Risk Maneuver
```

Ví dụ:

```text
Nếu xe đang bật hỗ trợ giữ làn
và tài xế không nhìn đường
→ hệ thống tăng cảnh báo

Nếu tài xế không phản ứng
→ xe có thể giảm tốc / bật hazard / dừng an toàn
```

### 11. Tự chẩn đoán và fail-safe

Hệ thống thật phải có trạng thái lỗi:

```text
- Camera lỗi
- Camera bị che
- LED IR lỗi
- Không thấy mặt
- Mắt bị che bởi kính đen
- Model confidence thấp
- ECU lỗi
- Nhiệt độ quá cao
- Kết nối CAN lỗi
```

Khi lỗi, xe không được im lặng như không có chuyện gì. Nó phải báo:

```text
DMS unavailable
Camera blocked
Driver monitoring limited
```

### Pipeline thực tế đầy đủ

```text
[Sensor Layer]
    IR camera
    IR LED
    Vehicle CAN signals
    Steering / speed / lane / ADAS state
    Seat / belt / hands-on-wheel

        ↓

[Pre-processing Layer]
    ISP
    Exposure control
    Noise reduction
    Face ROI crop
    Sensor health check
    Timestamp sync

        ↓

[Perception Layer]
    Driver detection
    Face tracking
    Eye tracking
    Eyelid tracking
    Gaze estimation
    Head pose estimation
    Mouth / yawn detection
    Hand / body pose detection

        ↓

[Feature Layer]
    PERCLOS
    Blink duration
    Eye closure time
    Eyes-off-road time
    Gaze zone
    Head nodding
    Yawn frequency
    Face visibility score
    Driver attention score

        ↓

[State Estimation Layer]
    Attentive
    Drowsy
    Microsleep
    Distracted
    Unresponsive
    No face
    System degraded

        ↓

[Decision Layer]
    Rule-based logic
    ML / Deep Learning model
    Temporal filtering
    Context filtering
    Risk scoring

        ↓

[HMI / Intervention Layer]
    Dashboard warning
    Sound alert
    Seat vibration
    Steering wheel vibration
    ADAS sensitivity adjustment
    Safe stop strategy

        ↓

[Data / Diagnostic Layer]
    Event log
    Fault code
    Calibration data
    Privacy filtering
    Optional fleet cloud report
```

### So sánh với pipeline đồ án

| Thành phần | Bản đồ án nên làm | Xe thực tế |
|---|---|---|
| Camera | Webcam / camera IR đơn giản | Camera cabin IR/NIR automotive-grade |
| Detect mặt | MediaPipe / YOLO | Model tối ưu riêng, chịu được kính, tối, rung |
| Tín hiệu | EAR, MAR, PERCLOS, head pose | Eye tracking, gaze, eyelid, head, body, CAN, ADAS |
| Quyết định | Rule-based | Rule + ML + temporal model + context |
| Cảnh báo | Âm thanh / giao diện | Âm thanh, hình ảnh, rung, can thiệp ADAS |
| Lỗi hệ thống | Thường chưa có | Bắt buộc có self-diagnosis/fail-safe |
| Dữ liệu | Lưu CSV/video | Lưu sự kiện giới hạn, chú trọng quyền riêng tư |
| Chuẩn xe | Không cần | DDAW, ADDW, Euro NCAP, an toàn chức năng |

### Kết luận ngắn gọn

Pipeline thực tế trên xe gồm **5 khối lớn**:

```text
1. Sensor + CAN data
2. Computer Vision perception
3. Temporal driver-state estimation
4. Warning / intervention logic
5. Diagnostics + privacy + ADAS integration
```

Còn với đồ án, bản hợp lý nhất là mô phỏng pipeline thực tế nhưng rút gọn:

```text
Camera
→ Face/Mesh detection
→ EAR + PERCLOS + MAR + Head pose
→ Rule-based risk score
→ NORMAL / WARNING / DROWSY / DISTRACTED
→ Cảnh báo + lưu log + gửi app mobile
```

---

## 5. User

Tôi muốn cải thiện Decision Layer bằng các tự train mô hình ML/DL thì có khả thi ?

---

## 6. Assistant

Có, **rất khả thi**. Nhưng hướng đúng là:

```text
Không nên train DL trực tiếp từ ảnh ngay từ đầu.
Nên train ML từ các đặc trưng đã trích xuất: EAR, MAR, PERCLOS, head pose, blink duration...
```

Cách này phù hợp với đồ án vì **dễ làm, dễ debug, chạy nhẹ trên board nhúng**.

### Decision Layer hiện tại

Bản rule-based hiện tại có dạng:

```python
if eyes_closed_time > 2.0:
    state = "DROWSY"

elif perclos > 0.25:
    state = "DROWSY"

elif yawn_count > 3:
    state = "WARNING"

else:
    state = "NORMAL"
```

Cách này đơn giản nhưng có nhược điểm:

```text
- Dễ cảnh báo sai
- Ngưỡng EAR/MAR không phù hợp với mọi người
- Không học được hành vi theo thời gian
- Khó kết hợp nhiều tín hiệu cùng lúc
```

### Decision Layer cải tiến bằng ML

Pipeline mới nên là:

```text
Camera
  ↓
Face / Eye / Mouth / Head tracking
  ↓
Feature extraction
  ↓
Temporal window 5s / 10s / 30s
  ↓
ML model
  ↓
Probability output
  ↓
Risk score
  ↓
NORMAL / WARNING / DROWSY / DISTRACTED
```

Thay vì model nhận ảnh, model nhận các đặc trưng dạng số.

Ví dụ input cho ML:

```text
EAR_mean
EAR_min
EAR_std
MAR_mean
MAR_max
PERCLOS
blink_rate
blink_duration_mean
eyes_closed_time
yawn_count
head_pitch_mean
head_pitch_std
head_yaw_mean
eyes_off_road_time
```

Output:

```text
NORMAL
WARNING
DROWSY
DISTRACTED
```

### Mô hình nên dùng

#### Giai đoạn đầu — ML truyền thống

Phù hợp nhất:

```text
Random Forest
XGBoost
LightGBM
SVM
KNN
Logistic Regression
```

Với đồ án, đề xuất:

```text
Random Forest hoặc XGBoost
```

Vì:

```text
- Không cần dataset quá lớn
- Train nhanh
- Dễ hiểu
- Chạy nhẹ
- Dễ xuất ra file model.pkl
- Dễ tích hợp vào app Python realtime
```

### Không nên train DL từ ảnh ngay

DL trực tiếp từ ảnh/video có dạng:

```text
Frame ảnh mắt / mặt
  ↓
CNN / LSTM / Transformer
  ↓
Drowsy / Normal
```

Cách này mạnh hơn nhưng khó hơn nhiều:

```text
- Cần dataset lớn
- Cần gán nhãn chính xác
- Cần GPU
- Dễ overfit
- Khó giải thích vì sao model cảnh báo
- Khó chạy realtime trên board yếu
```

Vì vậy:

```text
Feature-based ML trước
DL sau
```

### Dataset cần tự thu thập

Nên lưu dữ liệu thành file CSV. Mỗi dòng là một cửa sổ thời gian 5 giây:

```csv
ear_mean,ear_min,ear_std,mar_mean,mar_max,perclos,blink_rate,eyes_closed_time,yawn_count,head_pitch_mean,head_pitch_std,label
0.28,0.21,0.03,0.35,0.48,0.08,18,0.4,0,5.2,2.1,NORMAL
0.19,0.12,0.04,0.40,0.55,0.32,7,2.3,1,18.5,6.4,DROWSY
0.25,0.18,0.05,0.72,0.91,0.12,12,0.8,3,7.1,3.2,WARNING
```

Các nhãn nên có:

```text
NORMAL
WARNING
DROWSY
DISTRACTED
NO_FACE
```

Lưu ý an toàn: không cần và không nên tạo dữ liệu bằng cách cố ngủ gật khi đang lái xe thật. Có thể thu trong môi trường mô phỏng, ngồi trước camera, hoặc dùng video có kiểm soát.

### Cách tạo dữ liệu training

```text
Webcam
  ↓
Tính EAR, MAR, PERCLOS, head pose
  ↓
Gom dữ liệu theo cửa sổ 5 giây
  ↓
Tính mean, min, max, std
  ↓
Gán label
  ↓
Lưu CSV
```

Không nên train từng frame riêng lẻ, vì ngủ gật là hành vi theo thời gian. Nên train theo **window**.

### Kiến trúc Decision Layer mới

```text
Feature Buffer
  ↓
Window Aggregator
  ↓
ML Classifier
  ↓
Probability Smoothing
  ↓
Risk Engine
  ↓
Alert Controller
```

Ví dụ:

```python
proba = model.predict_proba(features)

normal_prob = proba["NORMAL"]
warning_prob = proba["WARNING"]
drowsy_prob = proba["DROWSY"]

if drowsy_prob > 0.75:
    state = "DROWSY"
elif warning_prob > 0.60:
    state = "WARNING"
else:
    state = "NORMAL"
```

Sau đó thêm bộ lọc thời gian:

```text
Nếu model báo DROWSY 1 lần → chưa cảnh báo mạnh
Nếu báo DROWSY liên tục 3 cửa sổ → cảnh báo mạnh
```

### Pipeline training model

```text
Bước 1: Thu dữ liệu
Bước 2: Tính feature
Bước 3: Gán nhãn
Bước 4: Train model
Bước 5: Đánh giá model
Bước 6: Lưu model.pkl
Bước 7: Tích hợp vào realtime app
```

Code train cơ bản:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/drowsiness_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, "models/drowsiness_model.pkl")
```

### Tích hợp model vào realtime

```python
import joblib

model = joblib.load("models/drowsiness_model.pkl")

features = extract_window_features(buffer)

state = model.predict([features])[0]
proba = model.predict_proba([features])[0]
```

Sau đó:

```python
if state == "DROWSY":
    trigger_alert()

elif state == "WARNING":
    show_warning()

else:
    normal_display()
```

### Nếu muốn dùng DL

Nếu dùng DL, không nên bắt đầu bằng ảnh thô. Nên dùng DL trên chuỗi feature.

Ví dụ input:

```text
30 frame hoặc 5 giây dữ liệu
```

Mỗi frame có:

```text
EAR
MAR
head_pitch
head_yaw
head_roll
eye_closed
mouth_open
```

Input dạng:

```text
sequence_length x feature_count
```

Ví dụ:

```text
150 x 7
```

Model phù hợp:

```text
LSTM
GRU
1D-CNN
Temporal CNN
```

Pipeline:

```text
Feature sequence
  ↓
LSTM / GRU
  ↓
DROWSY / NORMAL / DISTRACTED
```

### Mức độ khả thi

| Hướng | Khả thi | Độ khó | Nên làm không? |
|---|---:|---:|---|
| Rule-based | Rất cao | Thấp | Có |
| Random Forest từ feature | Rất cao | Trung bình | Rất nên |
| XGBoost từ feature | Cao | Trung bình | Nên |
| LSTM từ chuỗi feature | Trung bình | Khá cao | Làm sau |
| CNN từ ảnh mắt | Trung bình | Cao | Chưa nên |
| Video Transformer | Thấp | Rất cao | Không nên cho giai đoạn đầu |

### Hướng khuyên làm

**Version 1**

```text
Rule-based:
EAR + PERCLOS + MAR + head pose
```

**Version 2**

```text
ML-based:
Random Forest / XGBoost
Input: feature window 5 giây
Output: NORMAL / WARNING / DROWSY / DISTRACTED
```

**Version 3**

```text
Time-series DL:
LSTM / GRU
Input: chuỗi EAR, MAR, head pose theo thời gian
Output: trạng thái tài xế
```

Kết luận: **Có khả thi**, và rất nên cải thiện Decision Layer bằng mô hình tự train. Nhưng hướng tốt nhất là:

```text
Không train ảnh thô ngay.
Hãy train ML từ các đặc trưng đã tính được.
```

---

## 7. User

Tôi nên sử dụng một trong hai rule-base/ML trong Decision Layer hay là nên kết hợp cả hai?

---

## 8. Assistant

Bạn **nên kết hợp cả hai**.

Không nên chọn hoàn toàn **rule-based** hoặc hoàn toàn **ML**. Với hệ thống phát hiện ngủ gật, cách tốt nhất là:

```text
Rule-based + ML = Hybrid Decision Layer
```

### Vì sao nên kết hợp?

#### Rule-based mạnh ở phần an toàn rõ ràng

Ví dụ:

```text
Mắt nhắm liên tục > 2 giây
→ cảnh báo ngay
```

Hoặc:

```text
Không phát hiện mặt > 3 giây
→ NO_FACE
```

Những trường hợp này không cần ML, vì logic đã quá rõ.

#### ML mạnh ở phần đánh giá phức tạp

Ví dụ:

```text
EAR hơi thấp
+ blink duration tăng
+ PERCLOS tăng
+ head pitch thay đổi
+ MAR cao
```

Một tín hiệu riêng lẻ chưa chắc ngủ gật, nhưng nhiều tín hiệu kết hợp lại có thể là dấu hiệu nguy hiểm. Phần này ML làm tốt hơn rule-based.

### Kiến trúc nên dùng

Pipeline khuyến nghị:

```text
Camera
  ↓
Feature Extraction
  ↓
Rule-based Safety Check
  ↓
ML Classifier
  ↓
Risk Fusion
  ↓
Temporal Smoothing
  ↓
Alert Decision
```

Chi tiết:

```text
EAR, MAR, PERCLOS, head pose, blink rate
        ↓
Rule-based kiểm tra lỗi và tình huống nguy hiểm rõ ràng
        ↓
ML dự đoán NORMAL / WARNING / DROWSY / DISTRACTED
        ↓
Kết hợp kết quả rule + ML
        ↓
Cảnh báo
```

### Rule-based nên dùng để làm gì?

Rule-based nên xử lý các trường hợp:

```text
- Không thấy mặt
- Camera lỗi
- Mắt nhắm quá lâu
- Người lái nhìn lệch quá lâu
- Head pose bất thường rõ ràng
- Model ML confidence thấp
- Điều kiện cảnh báo khẩn cấp
```

Ví dụ:

```python
if no_face_time > 3.0:
    state = "NO_FACE"

elif eyes_closed_time > 2.0:
    state = "DROWSY"

elif eyes_off_road_time > 3.0:
    state = "DISTRACTED"
```

### ML nên dùng để làm gì?

ML nên xử lý các trường hợp không rõ ràng:

```text
- Buồn ngủ nhẹ
- Mệt mỏi tăng dần
- Nhiều tín hiệu yếu kết hợp lại
- Phân biệt NORMAL và WARNING
- Giảm cảnh báo sai
```

Input cho ML:

```text
EAR_mean
EAR_min
MAR_mean
PERCLOS
blink_rate
blink_duration
head_pitch_mean
head_yaw_mean
eyes_closed_time
yawn_count
```

Output:

```text
NORMAL
WARNING
DROWSY
DISTRACTED
```

### Cách kết hợp hợp lý nhất

Nên dùng kiểu:

```text
Rule-based override + ML classification
```

Tức là:

```text
Nếu có tình huống nguy hiểm rõ ràng
→ Rule quyết định ngay

Nếu không có tình huống nguy hiểm rõ ràng
→ Cho ML phân loại

Nếu ML không chắc chắn
→ Giữ trạng thái WARNING hoặc dùng rule dự phòng
```

Ví dụ logic:

```python
def decision_layer(features, ml_model):
    # 1. Rule-based safety override
    if features["no_face_time"] > 3.0:
        return "NO_FACE"

    if features["eyes_closed_time"] > 2.0:
        return "DROWSY"

    if features["eyes_off_road_time"] > 3.0:
        return "DISTRACTED"

    # 2. ML classification
    ml_state = ml_model.predict([features["ml_input"]])[0]
    ml_proba = max(ml_model.predict_proba([features["ml_input"]])[0])

    # 3. Confidence check
    if ml_proba < 0.55:
        return "WARNING"

    return ml_state
```

### Thêm Risk Score để mượt hơn

Thay vì chỉ dùng nhãn, có thể dùng điểm nguy cơ:

```text
risk_score = rule_score * 0.4 + ml_score * 0.6
```

Ví dụ:

```text
0.0 - 0.3 → NORMAL
0.3 - 0.6 → WARNING
0.6 - 0.8 → DROWSY
0.8 - 1.0 → DROWSY_CRITICAL
```

### Bảng so sánh

| Cách làm | Ưu điểm | Nhược điểm | Có nên dùng? |
|---|---|---|---|
| Chỉ rule-based | Dễ làm, dễ debug | Cảnh báo sai, kém linh hoạt | Dùng cho bản đầu |
| Chỉ ML | Học được nhiều tín hiệu | Cần dữ liệu, khó kiểm soát | Không nên dùng một mình |
| Kết hợp rule + ML | Ổn định, an toàn, thực tế hơn | Phức tạp hơn một chút | Nên dùng nhất |

### Khuyến nghị cho đồ án

Nên thiết kế Decision Layer như sau:

```text
Feature Extraction
  ↓
Rule Engine
  ↓
ML Classifier
  ↓
Risk Fusion
  ↓
Alert Controller
```

Trong đó:

```text
Rule Engine:
- Xử lý lỗi
- Xử lý tình huống nguy hiểm rõ ràng
- Làm fallback khi ML không chắc chắn

ML Classifier:
- Phân loại trạng thái tài xế
- Nhận biết mệt mỏi nhẹ
- Kết hợp nhiều đặc trưng cùng lúc

Risk Fusion:
- Kết hợp rule_score và ml_score
- Làm mượt kết quả theo thời gian
```

Kết luận:

```text
Rule-based dùng làm lớp an toàn và fallback.
ML dùng để phân loại trạng thái phức tạp.
```

Pipeline nên chọn:

```text
EAR/MAR/PERCLOS/Head pose
→ Rule-based override
→ ML classifier
→ Risk score
→ Alert
```

---

## 9. User

Hãy trình bày lại Pipeline đề xuất cuối cùng cho đồ án sau khi chọn phương án kết hợp Rule-based/ML thay cho pipeline đồ án cũ

---

## 10. Assistant

Dưới đây là **Pipeline đề xuất cuối cùng cho đồ án hệ thống phát hiện ngủ gật người lái xe**, sau khi thay **Decision Layer cũ** bằng phương án **kết hợp Rule-based + ML**.

### Pipeline tổng thể cuối cùng

```text
Camera
  ↓
Frame Pre-processing
  ↓
Face / Landmark Detection
  ↓
Feature Extraction
  ↓
Temporal Buffer
  ↓
Hybrid Decision Layer
      ├── Rule-based Engine
      ├── ML Classifier
      └── Risk Fusion
  ↓
Driver State Output
  ↓
Alert System
  ↓
Event Logging
  ↓
Mobile App / Dashboard
```

### 1. Input Layer — Camera

**Chức năng:** Thu hình ảnh khuôn mặt tài xế theo thời gian thực.

```text
Webcam / Camera IR / Camera cabin
```

**Output:**

```text
frame
timestamp
fps
```

### 2. Frame Pre-processing Layer

**Chức năng:** Chuẩn hóa ảnh trước khi đưa vào xử lý.

```text
- Resize frame
- Chuyển BGR → RGB
- Giảm nhiễu cơ bản
- Cân bằng sáng nếu cần
- Kiểm tra FPS
```

**Output:**

```text
processed_frame
```

### 3. Face / Landmark Detection Layer

**Chức năng:** Phát hiện khuôn mặt và các điểm đặc trưng trên mặt.

Nên dùng:

```text
MediaPipe Face Mesh
```

hoặc:

```text
OpenCV + model face landmark khác
```

**Output:**

```text
face_detected
face_landmarks
eye_landmarks
mouth_landmarks
nose_landmarks
head_points
```

Nếu không thấy mặt:

```text
face_detected = False
```

### 4. Feature Extraction Layer

Đây là tầng trích xuất đặc trưng hành vi của tài xế.

**Các feature chính:**

```text
EAR
MAR
PERCLOS
blink_rate
blink_duration
eyes_closed_time
yawn_count
head_pitch
head_yaw
head_roll
eyes_off_road_time
```

#### 4.1. Eye Features

```text
- EAR
- Mắt đang mở hay nhắm
- Thời gian nhắm mắt
- Tần suất chớp mắt
- Thời gian mỗi lần chớp mắt
```

Ví dụ output:

```json
{
  "ear": 0.18,
  "eye_closed": true,
  "eyes_closed_time": 1.7,
  "blink_rate": 8
}
```

#### 4.2. Mouth Features

```text
- MAR
- Miệng mở hay đóng
- Phát hiện ngáp
- Số lần ngáp trong một khoảng thời gian
```

Ví dụ output:

```json
{
  "mar": 0.72,
  "mouth_open": true,
  "yawn_detected": true,
  "yawn_count": 2
}
```

#### 4.3. Head Pose Features

```text
- Cúi đầu
- Ngẩng đầu
- Quay trái/phải
- Nghiêng đầu
```

Ví dụ output:

```json
{
  "head_pitch": 18.5,
  "head_yaw": -12.0,
  "head_roll": 3.2
}
```

#### 4.4. Gaze / Attention Features

```text
- Nhìn thẳng
- Nhìn xuống
- Nhìn trái/phải
- Không nhìn đường quá lâu
```

Ví dụ output:

```json
{
  "looking_forward": false,
  "eyes_off_road_time": 2.4
}
```

### 5. Temporal Buffer Layer

Hệ thống không nên quyết định chỉ dựa trên 1 frame. Cần lưu dữ liệu trong một khoảng thời gian.

```text
Buffer 5 giây
Buffer 10 giây
Buffer 30 giây
```

Ví dụ:

```text
150 frame gần nhất nếu camera chạy 30 FPS
```

**Chức năng:**

```text
- Lưu lịch sử EAR
- Lưu lịch sử MAR
- Lưu lịch sử head pose
- Lưu lịch sử trạng thái mắt
- Tính PERCLOS
- Tính số lần ngáp
- Tính thời gian mắt nhắm liên tục
```

**Output cho Decision Layer:**

```json
{
  "ear_mean": 0.24,
  "ear_min": 0.12,
  "ear_std": 0.05,
  "mar_mean": 0.42,
  "mar_max": 0.81,
  "perclos": 0.31,
  "blink_rate": 7,
  "eyes_closed_time": 2.1,
  "yawn_count": 3,
  "head_pitch_mean": 15.2,
  "head_yaw_mean": -8.5,
  "eyes_off_road_time": 2.8
}
```

### 6. Hybrid Decision Layer

Đây là phần thay đổi chính so với pipeline cũ.

Pipeline cũ:

```text
Feature Extraction
  ↓
Rule-based Decision
  ↓
Alert
```

Pipeline mới:

```text
Feature Extraction
  ↓
Temporal Buffer
  ↓
Rule-based Engine
  ↓
ML Classifier
  ↓
Risk Fusion
  ↓
Final State
```

#### 6.1. Rule-based Engine

**Vai trò:** Rule-based dùng để xử lý các tình huống rõ ràng, nguy hiểm hoặc lỗi hệ thống.

```text
- Không thấy mặt
- Mắt nhắm quá lâu
- Không nhìn đường quá lâu
- Head pose bất thường rõ ràng
- Camera lỗi
- ML không chắc chắn
```

Ví dụ rule:

```python
if no_face_time > 3.0:
    return "NO_FACE"

if eyes_closed_time > 2.0:
    return "DROWSY"

if eyes_off_road_time > 3.0:
    return "DISTRACTED"

if perclos > 0.35:
    return "DROWSY"
```

**Output:**

```text
rule_state
rule_score
rule_reason
```

Ví dụ:

```json
{
  "rule_state": "DROWSY",
  "rule_score": 0.85,
  "rule_reason": "eyes_closed_time > 2.0s"
}
```

#### 6.2. ML Classifier

**Vai trò:** ML dùng để phân loại trạng thái phức tạp hơn dựa trên nhiều feature kết hợp.

Input của ML là vector đặc trưng:

```text
ear_mean
ear_min
ear_std
mar_mean
mar_max
perclos
blink_rate
blink_duration_mean
eyes_closed_time
yawn_count
head_pitch_mean
head_yaw_mean
eyes_off_road_time
```

Model nên dùng trong đồ án:

```text
Random Forest hoặc XGBoost
```

**Output:**

```text
ml_state
ml_confidence
ml_probability
```

Ví dụ:

```json
{
  "ml_state": "WARNING",
  "ml_confidence": 0.72,
  "probability": {
    "NORMAL": 0.12,
    "WARNING": 0.72,
    "DROWSY": 0.14,
    "DISTRACTED": 0.02
  }
}
```

#### 6.3. Risk Fusion

Đây là tầng kết hợp kết quả từ rule-based và ML.

```text
Rule-based xử lý tình huống nguy hiểm rõ ràng.
ML xử lý trạng thái mệt mỏi phức tạp.
Risk Fusion kết hợp cả hai để đưa ra quyết định cuối.
```

Công thức đơn giản:

```text
final_risk = 0.5 * rule_score + 0.5 * ml_score
```

Hoặc ưu tiên Rule-based hơn trong trường hợp nguy hiểm:

```text
Nếu rule_state là DROWSY hoặc DISTRACTED
→ ưu tiên rule_state

Nếu rule_state bình thường
→ dùng ML state

Nếu ML confidence thấp
→ chuyển sang WARNING hoặc dùng rule fallback
```

Logic đề xuất:

```python
def hybrid_decision(rule_result, ml_result):
    # Rule override cho tình huống nguy hiểm rõ ràng
    if rule_result["state"] in ["NO_FACE", "DROWSY", "DISTRACTED"]:
        if rule_result["score"] >= 0.75:
            return rule_result["state"]

    # Nếu ML không chắc chắn
    if ml_result["confidence"] < 0.55:
        return "WARNING"

    # Nếu rule chỉ cảnh báo nhẹ, kết hợp với ML
    if rule_result["state"] == "WARNING" and ml_result["state"] == "WARNING":
        return "WARNING"

    if rule_result["state"] == "WARNING" and ml_result["state"] == "DROWSY":
        return "DROWSY"

    return ml_result["state"]
```

### 7. Temporal Smoothing Layer

Sau khi có trạng thái cuối từ Hybrid Decision Layer, không nên cảnh báo ngay sau 1 lần dự đoán.

```text
Nếu DROWSY xuất hiện 1 lần → chưa cảnh báo mạnh
Nếu DROWSY xuất hiện liên tục 3 lần → cảnh báo mạnh
Nếu WARNING kéo dài → cảnh báo nhẹ
Nếu NORMAL ổn định → tắt cảnh báo
```

Ví dụ:

```text
3 window gần nhất:
WARNING, DROWSY, DROWSY
→ trạng thái cuối: DROWSY
```

### 8. Driver State Output Layer

Hệ thống nên có các trạng thái:

```text
NORMAL
WARNING
DROWSY
DISTRACTED
NO_FACE
SYSTEM_ERROR
```

| Trạng thái | Ý nghĩa |
|---|---|
| NORMAL | Tài xế bình thường |
| WARNING | Có dấu hiệu mệt hoặc mất tập trung nhẹ |
| DROWSY | Có dấu hiệu ngủ gật |
| DISTRACTED | Không nhìn đường / mất tập trung |
| NO_FACE | Không phát hiện được mặt |
| SYSTEM_ERROR | Camera hoặc module xử lý lỗi |

### 9. Alert System Layer

Logic cảnh báo:

```text
NORMAL
→ Không cảnh báo

WARNING
→ Cảnh báo nhẹ

DROWSY
→ Cảnh báo mạnh

DISTRACTED
→ Cảnh báo vừa hoặc mạnh

NO_FACE
→ Nhắc tài xế quay lại vị trí

SYSTEM_ERROR
→ Báo lỗi hệ thống
```

Các kiểu cảnh báo:

```text
- Âm thanh beep
- Giọng nói cảnh báo
- Hiển thị trên màn hình
- Đèn LED
- Rung
- Gửi cảnh báo lên app mobile
```

Ví dụ:

```text
WARNING:
"Bạn có dấu hiệu mệt mỏi, hãy tập trung lái xe."

DROWSY:
"Cảnh báo! Phát hiện ngủ gật."

DISTRACTED:
"Cảnh báo! Bạn không nhìn đường."
```

### 10. Event Logging Layer

Khi có sự kiện bất thường, hệ thống nên lưu log.

Dữ liệu cần lưu:

```text
timestamp
state
ear
mar
perclos
eyes_closed_time
yawn_count
head_pitch
head_yaw
ml_state
ml_confidence
rule_reason
alert_level
```

Ví dụ log:

```json
{
  "timestamp": "2026-06-22 10:25:12",
  "state": "DROWSY",
  "ear": 0.14,
  "mar": 0.38,
  "perclos": 0.36,
  "eyes_closed_time": 2.3,
  "yawn_count": 1,
  "rule_reason": "eyes_closed_time > 2.0s",
  "ml_state": "DROWSY",
  "ml_confidence": 0.81,
  "alert_level": "HIGH"
}
```

### 11. Mobile App / Dashboard Layer

Nếu đồ án có app mobile, app nên hiển thị:

```text
- Trạng thái tài xế hiện tại
- Mức độ cảnh báo
- Thời gian xảy ra cảnh báo
- Lịch sử cảnh báo
- Số lần ngủ gật
- Số lần mất tập trung
- Biểu đồ theo ngày/tuần
```

### Pipeline runtime cuối cùng

```text
Start system
  ↓
Open camera
  ↓
Read frame
  ↓
Pre-process frame
  ↓
Detect face and landmarks
  ↓
Extract EAR, MAR, head pose, gaze
  ↓
Update temporal buffer
  ↓
Calculate window features
  ↓
Run rule-based engine
  ↓
Run ML classifier
  ↓
Fuse rule result and ML result
  ↓
Smooth state over time
  ↓
Generate final driver state
  ↓
Trigger alert if needed
  ↓
Save event log
  ↓
Send data to dashboard/app
  ↓
Repeat
```

### Pipeline training ML riêng

```text
Collect video/data
  ↓
Extract features
  ↓
Create time windows
  ↓
Label data
  ↓
Train ML model
  ↓
Evaluate model
  ↓
Save model.pkl
  ↓
Load model in realtime system
```

Dataset dùng để train:

```csv
ear_mean,ear_min,ear_std,mar_mean,mar_max,perclos,blink_rate,eyes_closed_time,yawn_count,head_pitch_mean,head_yaw_mean,eyes_off_road_time,label
0.28,0.21,0.03,0.35,0.48,0.08,18,0.4,0,5.2,2.1,0.3,NORMAL
0.21,0.15,0.04,0.50,0.74,0.22,10,1.1,2,8.1,4.5,1.4,WARNING
0.16,0.10,0.05,0.42,0.58,0.38,6,2.5,1,18.4,6.2,2.1,DROWSY
0.26,0.20,0.03,0.36,0.44,0.10,15,0.5,0,6.1,24.5,4.2,DISTRACTED
```

### Cấu trúc thư mục đề xuất

```text
dms-drowsiness-hybrid/
│
├── app/
│   ├── main.py
│   └── demo_webcam.py
│
├── src/
│   ├── camera/
│   │   └── camera_stream.py
│   │
│   ├── detection/
│   │   ├── face_landmark_detector.py
│   │   └── head_pose_estimator.py
│   │
│   ├── features/
│   │   ├── eye_features.py
│   │   ├── mouth_features.py
│   │   ├── gaze_features.py
│   │   ├── perclos.py
│   │   └── feature_buffer.py
│   │
│   ├── decision/
│   │   ├── rule_engine.py
│   │   ├── ml_classifier.py
│   │   ├── risk_fusion.py
│   │   └── state_smoother.py
│   │
│   ├── alert/
│   │   ├── sound_alert.py
│   │   ├── visual_alert.py
│   │   └── mobile_notify.py
│   │
│   ├── logging/
│   │   └── event_logger.py
│   │
│   └── utils/
│       ├── fps.py
│       └── config.py
│
├── training/
│   ├── collect_features.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── data/
│   ├── raw/
│   ├── features/
│   └── logs/
│
├── models/
│   └── drowsiness_model.pkl
│
└── requirements.txt
```

### Đoạn có thể đưa vào báo cáo

```text
Hệ thống đề xuất sử dụng kiến trúc Hybrid Decision Layer kết hợp Rule-based và Machine Learning. 
Dữ liệu hình ảnh từ camera được xử lý để phát hiện khuôn mặt và các điểm landmark. 
Từ đó hệ thống trích xuất các đặc trưng như EAR, MAR, PERCLOS, thời gian nhắm mắt, số lần ngáp và tư thế đầu. 
Các đặc trưng này được lưu trong bộ đệm thời gian để tính toán đặc trưng theo cửa sổ. 
Decision Layer gồm ba thành phần: Rule-based Engine, ML Classifier và Risk Fusion. 
Rule-based Engine xử lý các tình huống nguy hiểm rõ ràng như mắt nhắm quá lâu, không phát hiện khuôn mặt hoặc tài xế không nhìn đường. 
ML Classifier phân loại trạng thái tài xế dựa trên vector đặc trưng đã trích xuất. 
Risk Fusion kết hợp kết quả của hai thành phần để đưa ra trạng thái cuối cùng gồm NORMAL, WARNING, DROWSY, DISTRACTED, NO_FACE hoặc SYSTEM_ERROR. 
Khi phát hiện trạng thái nguy hiểm, hệ thống kích hoạt cảnh báo và lưu lại sự kiện để phục vụ theo dõi trên dashboard hoặc ứng dụng mobile.
```

### Sơ đồ pipeline cuối cùng

```text
┌──────────────┐
│   Camera     │
└──────┬───────┘
       ↓
┌──────────────┐
│ Pre-process  │
└──────┬───────┘
       ↓
┌────────────────────┐
│ Face / Landmark    │
│ Detection          │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Feature Extraction │
│ EAR, MAR, Pose...  │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Temporal Buffer    │
│ 5s / 10s / 30s     │
└──────┬─────────────┘
       ↓
┌──────────────────────────────┐
│ Hybrid Decision Layer         │
│ ┌──────────────┐              │
│ │ Rule Engine  │              │
│ └──────┬───────┘              │
│        ↓                      │
│ ┌──────────────┐              │
│ │ ML Classifier│              │
│ └──────┬───────┘              │
│        ↓                      │
│ ┌──────────────┐              │
│ │ Risk Fusion  │              │
│ └──────────────┘              │
└──────┬───────────────────────┘
       ↓
┌────────────────────┐
│ State Smoothing    │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Driver State       │
│ NORMAL / WARNING   │
│ DROWSY / DISTRACTED│
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Alert System       │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Event Logging      │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Mobile Dashboard   │
└────────────────────┘
```

### Kết luận

Pipeline cuối cùng cho đồ án nên là:

```text
Camera
→ Pre-processing
→ Face/Landmark Detection
→ Feature Extraction
→ Temporal Buffer
→ Rule-based Engine
→ ML Classifier
→ Risk Fusion
→ State Smoothing
→ Alert
→ Logging
→ Mobile App/Dashboard
```

Điểm nâng cấp quan trọng so với pipeline cũ là:

```text
Decision Layer không còn chỉ dùng rule-based.
Thay vào đó, hệ thống dùng Hybrid Decision Layer:
Rule-based để xử lý tình huống nguy hiểm rõ ràng.
ML để phân loại trạng thái phức tạp.
Risk Fusion để kết hợp hai kết quả và giảm cảnh báo sai.
```

Đây là pipeline phù hợp nhất cho đồ án DMS: **thực tế hơn, có tính AI rõ ràng hơn, nhưng vẫn đủ khả thi để triển khai.**

---

## 11. User

lưu toàn bộ dữ liệu cuộc trò chuyện vào file markdown

---

## 12. Assistant

Đã tạo file markdown chứa nội dung cuộc trò chuyện hiện tại.

