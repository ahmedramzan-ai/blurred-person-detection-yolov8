from ultralytics import YOLO
import cv2
import os

def run_video_inference():
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(ROOT_DIR, "models", "blur_finetuned.pt")
    video_path = os.path.join(ROOT_DIR, "Eval-test media", "blurred_video2.mp4")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated_frame = results[0].plot()

        cv2.imshow("Blur-trained Model Output", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_inference()
