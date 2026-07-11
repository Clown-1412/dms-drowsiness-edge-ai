# Prompt phân tích `src/features/` của dự án DMS

## Vai trò

Bạn là một **Senior AI Software Architect** và **Computer Vision
Engineer**. Hãy thực hiện **technical review** toàn bộ thư mục
`src/features/` của repository Driver Monitoring System (DMS).

Mục tiêu **không phải sửa code ngay**, mà là **phân tích kiến trúc hiện
tại**, đánh giá chất lượng thiết kế và đề xuất hướng cải thiện để chuẩn
bị tích hợp **Hybrid Decision Layer (Rule-based + ML)**.

## Yêu cầu phân tích

1.  Phân tích tổng quan Feature Layer và vai trò trong pipeline.
2.  Phân tích từng file trong `src/features/`:
    -   Vai trò
    -   Input/Output
    -   Quan hệ với các file khác
    -   Luồng dữ liệu
3.  Phân tích từng class:
    -   Trách nhiệm
    -   Thuộc tính
    -   Phương thức
    -   Đánh giá SRP
4.  Phân tích các hàm quan trọng:
    -   Mục đích
    -   Input/Output
    -   Thuật toán
    -   Điểm tối ưu
5.  Vẽ pipeline Feature Layer bằng sơ đồ ASCII.
6.  Đánh giá OOP:
    -   SRP
    -   OCP
    -   Coupling
    -   Cohesion
    -   Maintainability
    -   Reusability
7.  Đánh giá cho Edge AI:
    -   Thành phần tốn CPU
    -   Tính toán dư thừa
    -   Khả năng cache
    -   Khả năng tối ưu cho Orange Pi Zero 3W
8.  Đánh giá mức độ sẵn sàng cho Hybrid Decision Layer:
    -   Output hiện tại đã đủ chưa?
    -   Thiếu feature nào (blink_duration, blink_rate, eyes_closed_time,
        yawn_count, perclos, gaze_direction, eyes_off_road_time...)
9.  Đề xuất refactor:
    -   File nên tách
    -   File nên gộp
    -   Class nên chia nhỏ
    -   Hàm nên chuyển module
    -   Kiến trúc mới nếu cần
10. Đề xuất roadmap cải tiến theo từng bước.

## Yêu cầu đầu ra

-   Tổng quan Feature Layer.
-   Phân tích từng file.
-   Phân tích từng class.
-   Phân tích từng hàm.
-   Sơ đồ pipeline.
-   Đánh giá OOP.
-   Điểm mạnh.
-   Điểm yếu.
-   Đề xuất refactor.
-   Chuẩn bị cho Hybrid Decision Layer.
-   Roadmap cải tiến.

Lưu ý: - Không chỉnh sửa code. - Chỉ phân tích, đánh giá và đưa ra đề
xuất có giải thích rõ ràng.
