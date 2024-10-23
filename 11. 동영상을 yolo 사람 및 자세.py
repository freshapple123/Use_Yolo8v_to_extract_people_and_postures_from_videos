import cv2
from ultralytics import YOLO

# YOLOv8 Pose 모델 불러오기
model = YOLO("yolov8n-pose.pt")

# 동영상 파일 또는 웹캠에서 영상 스트림 가져오기
video_path = (
    "C:/Users/PC/Desktop/배구/short.mp4"  # 동영상 파일 경로 (웹캠 사용 시 0으로 변경)
)
cap = cv2.VideoCapture(video_path)

# 영상 스트림이 열렸는지 확인
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
else:
    while cap.isOpened():
        ret, frame = cap.read()

        # 프레임이 제대로 불러와졌는지 확인
        if not ret:
            print("프레임을 읽을 수 없습니다. 비디오가 끝났을 수 있습니다.")
            break

        # YOLOv8 모델로 프레임에서 포즈 탐지
        results = model(frame)

        # 탐지된 결과를 프레임 위에 시각화
        annotated_frame = results[0].plot()

        # 실시간 프레임 출력
        cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

        # 'q' 키를 누르면 창 닫기
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 캡처 객체 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()
