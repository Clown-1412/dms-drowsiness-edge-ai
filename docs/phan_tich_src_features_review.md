# Technical Review `src/features/` - DMS Feature Layer

Ngày phân tích: 2026-07-11

## 1. Tổng quan Feature Layer

Feature Layer hiện tại nằm sau `Landmark Detection` và trước các tầng dự kiến như
`Temporal Buffer`, `Hybrid Decision Layer`, `Driver State Output`.

Vai trò chính của layer này là chuyển đổi landmark thô từ MediaPipe Face Mesh
thành các đặc trưng có ý nghĩa cho bài toán Driver Monitoring System:

- Đặc trưng mắt: EAR, trạng thái từng mắt, trạng thái cả hai mắt nhắm.
- Đặc trưng miệng: MAR, trạng thái miệng mở.
- Đặc trưng tư thế đầu: pitch, yaw, roll, head_down, head_turned.
- Metadata chất lượng: feature hợp lệ hay không, lỗi feature nếu có.

Feature Layer hiện đang xử lý ở mức frame-level. Các feature theo thời gian như
`blink_duration`, `blink_rate`, `eyes_closed_time`, `perclos`, `yawn_count`,
`eyes_off_road_time` chưa thuộc trách nhiệm của layer này và nên nằm ở Temporal
Layer hoặc Hybrid Decision Layer.

## 2. Pipeline Feature Layer

```text
Camera
  -> Pre-processing
  -> Landmark Detection
       outputs:
       face_detected
       eye_landmarks
       mouth_landmarks
       head_points
       image_size
       timestamp
       fps
  -> Feature Extraction
       + geometry.py
       + eye.py
       + mouth.py
       + head_pose.py
       + feature_extractor.py
  -> quality_features
  -> Temporal Buffer
  -> Hybrid Decision Layer
       + Rule-based Engine
       + ML Classifier
       + Risk Fusion
```

Luồng dữ liệu thực tế:

```text
src/detection/landmark_detector.py
  BoPhatHienDiemMat.phat_hien()
    -> eye_landmarks: {"left_eye": [...], "right_eye": [...]}
    -> mouth_landmarks: [...]
    -> head_points: {"nose_tip": ..., "chin": ..., ...}
    -> image_size: (width, height)

src/features/feature_extractor.py
  BoTrichXuatDacTrung.trich_xuat()
    -> BoTrichXuatDacTrungMat.trich_xuat()
    -> BoTrichXuatDacTrungMieng.trich_xuat()
    -> BoTrichXuatDacTrungTuTheDau.trich_xuat()
    -> quality_features
```

## 3. Vấn đề ưu tiên cao

Workspace hiện tại có dấu hiệu refactor đổi tên file chưa hoàn tất:

- `src/features/eye_features.py` đã bị xóa.
- `src/features/mouth_features.py` đã bị xóa.
- `src/features/head_pose_features.py` đã bị xóa.
- File mới tồn tại là `eye.py`, `mouth.py`, `head_pose.py`.
- Tuy nhiên `src/features/feature_extractor.py` vẫn import từ các module cũ:
  `src.features.eye_features`, `src.features.mouth_features`,
  `src.features.head_pose_features`.
- `src/features/__init__.py` cũng vẫn import các module cũ.

Hệ quả: Feature Layer có khả năng không import được, kéo theo pipeline không thể
khởi tạo `BoTrichXuatDacTrung`.

Đây là lỗi kiến trúc/đóng gói cần xử lý trước khi thêm Hybrid Decision Layer.

## 4. Phân tích từng file

### 4.1 `src/features/geometry.py`

Vai trò:

- Cung cấp helper dùng chung cho validation landmark và tính khoảng cách 2D.

Hàm chính:

- `diem_hop_le(diem)`
  - Input: object bất kỳ.
  - Output: `True/False`.
  - Kiểm tra `diem` là dict và có thể ép `x`, `y` sang float.

- `danh_sach_diem_hop_le(danh_sach_diem, so_diem_toi_thieu)`
  - Input: iterable landmark và số điểm tối thiểu.
  - Output: `True/False`.
  - Kiểm tra danh sách đủ dài và các điểm đầu tiên hợp lệ.

- `khoang_cach_euclid(diem_1, diem_2)`
  - Input: hai landmark dict.
  - Output: khoảng cách Euclid dạng float.
  - Hiện dùng `np.array` và `np.linalg.norm`.

Đánh giá:

- Cohesion tốt, file nhỏ và đúng trách nhiệm.
- Có thể tối ưu cho Edge AI bằng `math.hypot(dx, dy)` để tránh tạo NumPy array
  nhiều lần trong mỗi frame.
- `danh_sach_diem_hop_le()` convert iterable sang list; ổn với danh sách nhỏ,
  nhưng nếu input lớn thì có chi phí phụ.

### 4.2 `src/features/eye.py`

Class: `BoTrichXuatDacTrungMat`

Vai trò:

- Tính Eye Aspect Ratio cho mắt trái/phải.
- Suy luận trạng thái mắt nhắm/mở theo ngưỡng.
- Trả output frame-level cho DMS.

Input:

```python
{
    "left_eye": [point1, point2, point3, point4, point5, point6],
    "right_eye": [point1, point2, point3, point4, point5, point6],
}
```

Output:

```python
{
    "left_EAR": float,
    "right_EAR": float,
    "avg_EAR": float,
    "left_eye_closed": bool,
    "right_eye_closed": bool,
    "both_eyes_closed": bool,
    "one_eye_closed": bool,
    "eye_closed": bool,
    "is_valid": bool,
    "error": None | str,
}
```

Luồng xử lý:

```text
eye_landmarks
  -> validate left/right eye
  -> tinh_ear(left_eye)
  -> tinh_ear(right_eye)
  -> avg_EAR
  -> compare threshold
  -> derive eye state flags
```

Hàm `tinh_ear()`:

- Công thức:

```text
EAR = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))
```

- Nếu landmark không hợp lệ hoặc khoảng ngang bằng 0 thì trả `0.0`.
- Thuật toán O(1), chi phí thấp.

Hàm `trich_xuat()`:

- Validate input là dict.
- Kiểm tra mỗi mắt có đủ 6 điểm.
- Tính `left_EAR`, `right_EAR`, `avg_EAR`.
- Nếu một EAR không hợp lệ thì trả `is_valid=False`, giữ bool mặc định `False`.
- Nếu hợp lệ thì tính:

```python
left_eye_closed = left_ear < self.nguong_mat_nham
right_eye_closed = right_ear < self.nguong_mat_nham
both_eyes_closed = left_eye_closed and right_eye_closed
one_eye_closed = left_eye_closed != right_eye_closed
eye_closed = both_eyes_closed
```

Đánh giá SRP:

- Tốt. Class chỉ xử lý đặc trưng mắt frame-level.
- Chưa nên nhét blink duration hoặc perclos vào class này.

Điểm mạnh:

- Logic `eye_closed = both_eyes_closed` phù hợp hơn cho DMS so với chỉ cần một
  mắt nhắm.
- Có error handling tốt, không crash khi landmark thiếu.

Điểm yếu:

- Ngưỡng `0.20` đang hard-code trong constructor mặc định, chưa đọc từ
  `config.yaml`.
- Chưa có confidence score hoặc quality score cho landmark mắt.
- Chưa có smoothing/hysteresis nên tín hiệu vẫn có thể nhiễu ở frame-level.

### 4.3 `src/features/mouth.py`

Class: `BoTrichXuatDacTrungMieng`

Vai trò:

- Tính Mouth Aspect Ratio.
- Suy luận `mouth_open` theo ngưỡng.

Input:

```python
[
    p1, p2, p3, p4, p5, p6, p7, p8
]
```

Output:

```python
{
    "MAR": float,
    "mouth_open": bool,
    "is_valid": bool,
    "error": None | str,
}
```

Hàm `tinh_mar()`:

- Công thức:

```text
MAR = (vertical_1 + vertical_2 + vertical_3) / (2 * horizontal)
```

- Dùng 8 điểm miệng.
- Nếu landmark thiếu hoặc chiều ngang bằng 0 thì trả `0.0`.

Hàm `trich_xuat()`:

- Validate đủ 8 điểm.
- Kiểm tra mouth width.
- Tính MAR.
- Kiểm tra `math.isfinite(mar)` và `mar >= 0`.
- Trả `mouth_open = mar > nguong_mieng_mo`.

Đánh giá SRP:

- Tốt. Class chỉ xử lý feature miệng frame-level.

Điểm mạnh:

- Output đơn giản, dễ dùng.
- Có validation tránh crash.

Điểm yếu:

- `horizontal` được tính trong `trich_xuat()` rồi tính lại trong `tinh_mar()`.
- Ngưỡng `0.60` chưa đọc từ config.
- `mouth_open` chưa đủ để kết luận ngáp; cần temporal feature như
  `yawn_duration`, `yawn_count`.

### 4.4 `src/features/head_pose.py`

Class: `BoTrichXuatDacTrungTuTheDau`

Vai trò:

- Ước lượng tư thế đầu bằng `cv2.solvePnP`.
- Trả Euler angles và các flag `head_down`, `head_turned`.

Input:

```python
head_points = {
    "nose_tip": point,
    "chin": point,
    "left_eye_corner": point,
    "right_eye_corner": point,
    "left_mouth_corner": point,
    "right_mouth_corner": point,
}

image_size = (width, height)
```

Output:

```python
{
    "head_pitch": float | None,
    "head_yaw": float | None,
    "head_roll": float | None,
    "head_down": bool,
    "head_turned": bool,
    "is_valid": bool,
    "error": None | str,
}
```

Luồng xử lý:

```text
head_points + image_size
  -> validate image size
  -> convert 6 image points to np.ndarray
  -> build camera matrix
  -> cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
  -> cv2.Rodrigues(rotation_vector)
  -> Euler angles
  -> normalize pitch/roll
  -> threshold head_down/head_turned
```

Hàm quan trọng:

- `_lay_image_points()`
  - Validate đủ 6 điểm chính.
  - Convert dict landmark sang `np.ndarray`.

- `_lay_kich_thuoc_anh()`
  - Validate `image_size`.
  - Trả `(width, height)` dạng float.

- `_solve_pnp()`
  - Tạo camera matrix mỗi frame.
  - Gọi `cv2.solvePnP`.
  - Đây là phần nặng nhất trong `src/features/`.

- `_tinh_goc_euler()`
  - Convert rotation vector sang matrix.
  - Tính pitch/yaw/roll.
  - Có xử lý trường hợp suy biến.

- `_chuan_hoa_goc_quanh_0()`
  - Chuẩn hóa góc quanh 0 độ để tránh nghiệm tương đương quanh +/-180 độ.

Đánh giá SRP:

- Trung bình. Class đang làm cả ước lượng pose, chuyển đổi góc, validate input,
  tạo camera matrix và diễn giải threshold.
- Với demo thì chấp nhận được; với production nên tách nhỏ.

Điểm mạnh:

- Có try/except cho lỗi OpenCV và NumPy.
- Model points được cache trong instance.
- Output đủ tốt cho rule frame-level.

Điểm yếu:

- Camera matrix và dist coeffs tạo lại mỗi frame.
- `solvePnP` tốn CPU trên thiết bị edge.
- Threshold `head_down`, `head_turned` hard-code.
- Chưa có confidence hoặc stability score.

### 4.5 `src/features/feature_extractor.py`

Class: `BoTrichXuatDacTrung`

Vai trò:

- Orchestrator tổng hợp feature từ output Landmark Detection.
- Gọi lần lượt eye, mouth, head pose extractors.
- Trả schema thống nhất cho pipeline.

Input:

```python
{
    "timestamp": float,
    "fps": float,
    "image_size": (width, height),
    "face_detected": bool,
    "eye_landmarks": dict,
    "mouth_landmarks": list,
    "head_points": dict,
}
```

Output khi không thấy mặt:

```python
{
    "timestamp": ...,
    "fps": ...,
    "face_detected": False,
    "eye_features": None,
    "mouth_features": None,
    "head_pose_features": None,
    "quality_features": {
        "is_feature_valid": False,
        "feature_error": "NO_FACE",
        "image_size": image_size,
        "fps": fps,
    },
}
```

Output khi thấy mặt:

```python
{
    "timestamp": ...,
    "fps": ...,
    "face_detected": True,
    "eye_features": {...},
    "mouth_features": {...},
    "head_pose_features": {...},
    "quality_features": {
        "is_feature_valid": bool,
        "feature_error": None | str,
        "image_size": image_size,
        "fps": fps,
    },
}
```

Hàm `trich_xuat()`:

- Chuẩn hóa input không phải dict thành `{}`.
- Nếu không có mặt thì trả feature rỗng và lỗi `NO_FACE`.
- Nếu có mặt thì gọi ba extractor.
- `is_feature_valid` hiện chỉ xét eye hoặc mouth hợp lệ:

```python
is_feature_valid = bool(
    eye_features.get("is_valid") or mouth_features.get("is_valid")
)
```

Đánh giá:

- Vai trò aggregator rõ ràng.
- Có dependency injection cho các extractor, tốt cho test.
- Nhưng đang import sai tên module do refactor dang dở.
- `is_feature_valid` chưa xét head pose, cần định nghĩa lại policy.

### 4.6 `src/features/__init__.py`

Vai trò:

- Expose public API của package `src.features`.

Vấn đề:

- Đang import từ module cũ đã bị xóa:
  `eye_features`, `mouth_features`, `head_pose_features`.
- Nếu người dùng import trực tiếp `src.features`, khả năng cao sẽ lỗi.

Đề xuất:

- Đồng bộ import với tên file hiện tại.
- Hoặc khôi phục tên file cũ nếu muốn giữ backward compatibility.

## 5. Đánh giá OOP

### SRP

- `eye.py`: tốt.
- `mouth.py`: tốt.
- `geometry.py`: tốt.
- `head_pose.py`: trung bình vì đang gộp nhiều bước kỹ thuật.
- `feature_extractor.py`: tốt ở vai trò orchestration, nhưng đang phụ thuộc nhiều
  vào key string.

### OCP

Mức trung bình thấp.

Lý do:

- Thêm feature mới như gaze hoặc perclos sẽ cần sửa `feature_extractor.py` và
  schema dict.
- Chưa có interface/base class chung cho các extractor.
- Chưa có registry để bật/tắt extractor theo config.

### Coupling

Coupling còn cao qua dictionary key:

- `eye_landmarks`
- `mouth_landmarks`
- `head_points`
- `left_EAR`
- `MAR`
- `head_pitch`
- `is_valid`
- `error`

Nếu typo key hoặc đổi schema, lỗi có thể chỉ xuất hiện runtime.

### Cohesion

Tốt trong từng file. Mỗi file tập trung vào một nhóm feature.

### Maintainability

Ổn cho demo, chưa đủ chắc cho production edge AI.

Các điểm cần cải thiện:

- Đồng bộ tên file/import.
- Chuẩn hóa schema.
- Thêm test.
- Đưa threshold ra config.
- Tách temporal feature khỏi frame-level feature.

### Reusability

- `geometry.py` tái sử dụng tốt.
- Eye/mouth/head pose extractors có thể tái sử dụng nếu input schema ổn định.
- `feature_extractor.py` có dependency injection nên dễ mock trong test.

## 6. Đánh giá Edge AI

### Thành phần tốn CPU

Trong phạm vi `src/features/`, thành phần tốn CPU nhất là:

- `cv2.solvePnP` trong `head_pose.py`.
- `cv2.Rodrigues` và tính Euler angles.
- Tạo camera matrix và `np.ndarray` mỗi frame.

EAR/MAR rất nhẹ so với head pose.

### Tính toán dư thừa

- `mouth.py` tính mouth width hai lần.
- `geometry.py` tạo NumPy array cho từng khoảng cách rất nhỏ.
- `head_pose.py` tạo camera matrix và `dist_coeffs` mỗi frame dù `image_size`
  thường không đổi.

### Khả năng cache

Nên cache:

- `camera_matrix` theo `(width, height)`.
- `dist_coeffs`.
- Có thể giữ `image_points` array buffer nếu cần tối ưu sâu hơn.

### Tối ưu cho Orange Pi Zero 3W

Đề xuất:

- Dùng `math.hypot` cho EAR/MAR.
- Cache camera matrix.
- Cho phép skip head pose mỗi N frame nếu FPS thấp.
- Cho phép tắt head pose bằng config trong chế độ tiết kiệm CPU.
- Chỉ tính feature temporal bằng sliding window O(1).
- Tránh log mỗi frame.

## 7. Mức độ sẵn sàng cho Hybrid Decision Layer

### Feature hiện có

Đã có:

- `left_EAR`
- `right_EAR`
- `avg_EAR`
- `left_eye_closed`
- `right_eye_closed`
- `both_eyes_closed`
- `one_eye_closed`
- `eye_closed`
- `MAR`
- `mouth_open`
- `head_pitch`
- `head_yaw`
- `head_roll`
- `head_down`
- `head_turned`
- `timestamp`
- `fps`
- `face_detected`
- `quality_features`

Các feature này đủ cho rule đơn giản ở mức frame-level.

### Feature còn thiếu

Để Hybrid Decision Layer đáng tin cậy hơn, cần thêm temporal/behavioral features:

- `blink_duration`
- `blink_rate`
- `eyes_closed_time`
- `perclos`
- `yawn_count`
- `yawn_duration`
- `gaze_direction`
- `eyes_off_road_time`
- `face_missing_duration`
- `head_down_duration`
- `head_turned_duration`
- `feature_confidence`
- `landmark_quality`

### Đánh giá readiness

Mức sẵn sàng: trung bình.

Feature Layer đã có nền tảng tốt, nhưng chưa đủ để làm decision ổn định. Nên
hoàn thiện Temporal Layer trước khi viết Hybrid Decision Layer.

## 8. Điểm mạnh

- Phân tách tương đối rõ giữa eye, mouth, head pose.
- Output có `is_valid` và `error`, giảm nguy cơ crash pipeline.
- Logic mắt đã hợp lý hơn cho DMS: `eye_closed` chỉ true khi cả hai mắt nhắm.
- `feature_extractor.py` có dependency injection.
- `display_utils.py` đã tiêu thụ output feature theo cách không crash khi feature
  là `None`.
- Các feature frame-level cơ bản đã đủ để demo.

## 9. Điểm yếu

- Import đang lệch với tên file hiện tại, có thể làm pipeline không chạy.
- Chưa có test tự động.
- Threshold hard-code, chưa lấy từ `config.yaml`.
- Chưa có schema typed/dataclass cho feature output.
- Error hiện là string đơn lẻ hoặc chuỗi ghép bằng `;`, chưa phù hợp cho xử lý
  downstream phức tạp.
- Head pose hơi nặng cho edge device nếu chạy mỗi frame.
- Chưa có temporal feature nên chưa thể kết luận buồn ngủ ổn định.
- Chưa có gaze estimation hoặc off-road attention.

## 10. Đề xuất refactor

### Ưu tiên 1: Sửa module naming/import

Chọn một trong hai hướng:

Hướng A:

- Giữ tên file mới `eye.py`, `mouth.py`, `head_pose.py`.
- Sửa import trong `feature_extractor.py` và `__init__.py`.

Hướng B:

- Khôi phục tên file cũ `eye_features.py`, `mouth_features.py`,
  `head_pose_features.py`.
- Xóa hoặc không dùng file mới để tránh trùng trách nhiệm.

Khuyến nghị: dùng hướng A nếu muốn tên file ngắn gọn hơn, nhưng cần cập nhật toàn
bộ import.

### Ưu tiên 2: Chuẩn hóa schema

Tạo dataclass hoặc TypedDict:

- `EyeFeatures`
- `MouthFeatures`
- `HeadPoseFeatures`
- `QualityFeatures`
- `FrameFeatures`

Sau đó serialize sang dict ở boundary nếu display/pipeline vẫn cần dict.

### Ưu tiên 3: Đưa threshold ra config

Các ngưỡng nên lấy từ `config.yaml`:

- `ear_closed`
- `mar_yawn`
- `head_down`
- `head_turned`
- `perclos_window_sec`
- `perclos_threshold`

### Ưu tiên 4: Tối ưu head pose

- Cache camera matrix theo `image_size`.
- Cho phép cấu hình tần suất tính head pose.
- Tách `HeadPoseEstimator` và `HeadPoseRuleInterpreter`.

### Ưu tiên 5: Tạo Temporal Layer

Không nên đưa temporal feature vào `src/features/eye.py`.

Nên tạo module riêng, ví dụ:

```text
src/temporal/
  buffer.py
  eye_temporal.py
  mouth_temporal.py
  head_temporal.py
```

Temporal Layer nhận frame-level feature và trả:

- blink metrics
- yawn metrics
- perclos
- duration metrics

### Ưu tiên 6: Chuẩn bị Hybrid Decision Layer

Đề xuất cấu trúc:

```text
src/decision/
  rules.py
  ml_classifier.py
  fusion.py
  driver_state.py
```

Input của Hybrid Decision Layer nên là schema đã chuẩn hóa từ Feature Layer và
Temporal Layer.

## 11. Roadmap cải tiến

### Bước 1: Khôi phục khả năng chạy

- Sửa import sai trong `feature_extractor.py`.
- Sửa import sai trong `__init__.py`.
- Chạy smoke test import.

### Bước 2: Thêm unit test frame-level

Test cần có:

- EAR hợp lệ.
- EAR thiếu điểm.
- Một mắt nhắm.
- Hai mắt nhắm.
- MAR hợp lệ.
- Mouth landmark thiếu.
- Head pose thiếu điểm.
- `feature_extractor` khi `face_detected=False`.

### Bước 3: Chuẩn hóa schema và error

- Chuyển output sang TypedDict/dataclass.
- Error nên là list code thay vì string ghép.

### Bước 4: Kết nối config

- Đưa threshold từ `config.yaml` vào extractor constructors.
- Cho phép tune theo camera/ánh sáng/khoảng cách mặt.

### Bước 5: Tối ưu Edge AI

- Cache camera matrix.
- Tối ưu distance bằng `math.hypot`.
- Skip hoặc giảm tần suất head pose khi FPS thấp.

### Bước 6: Xây Temporal Buffer

- Sliding window theo timestamp.
- Tính `perclos`, `eyes_closed_time`, `blink_rate`, `blink_duration`,
  `yawn_count`, `yawn_duration`.

### Bước 7: Xây Rule-based Decision Engine

Rule ban đầu:

- eyes closed quá N giây.
- PERCLOS vượt ngưỡng.
- yawning kéo dài hoặc lặp lại.
- head down kéo dài.
- head turned/off-road kéo dài.

### Bước 8: Thêm ML Classifier

- Input là feature vector đã chuẩn hóa.
- Model nhẹ phù hợp edge: Logistic Regression, SVM, Random Forest nhỏ, hoặc TinyML
  nếu cần.

### Bước 9: Risk Fusion

- Kết hợp rule score và ML probability.
- Có hysteresis để tránh cảnh báo nhấp nháy.
- Có state machine: normal, attention_low, drowsy, alert.

## 12. Ghi chú kiểm chứng

Các lệnh đã dùng để phân tích:

```powershell
rg --files src/features
rg -n "features|Feature|EAR|MAR|PERCLOS|blink|yawn|gaze|drows" src docs -S
git status --short
git diff -- src/features
```

Không chạy được test/import Python vì:

- `python` không có trên PATH.
- `py` báo không tìm thấy Python.
- `.venv\Scripts\python.exe` trỏ tới Python 3.10 không còn tồn tại.

Vì vậy các kết luận runtime cần được xác nhận lại sau khi sửa môi trường Python.

## 13. Kết luận

Feature Layer hiện đã có nền tốt cho DMS demo ở mức frame-level, đặc biệt là các
feature EAR, MAR và head pose. Tuy nhiên trước khi tích hợp Hybrid Decision Layer,
cần ưu tiên sửa lệch import/module naming, chuẩn hóa output schema, đưa threshold
ra config, thêm test và xây Temporal Layer.

Không nên thêm ML ngay ở bước hiện tại. Việc quan trọng hơn là làm cho Feature
Layer ổn định, đo được, test được và nhẹ đủ để chạy trên edge device.
