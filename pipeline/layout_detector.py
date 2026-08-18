from doclayout_yolo import YOLOv10


class LayoutDetector:
    def __init__(self, model_path="models/doclayout_yolo.pt"):
        print("Loading DocLayout-YOLO...")
        self.model = YOLOv10(model_path)
        print("✓ Model loaded!")

    def detect(self, image_path):
        results = self.model.predict(
            source=image_path,
            imgsz=1024,
            conf=0.20,
            device="cpu",
            save=False,
            verbose=False
        )

        return results