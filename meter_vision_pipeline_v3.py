# ============================================================
# Meter Vision Pipeline v3
# RCNN localization + SmallDigitCNN with segment confidence
# ============================================================

import json
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps


# ============================================================
# Segment preprocessing (must match training)
# ============================================================

class SegmentNormalize:
    def __call__(self, img):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        return img


# ============================================================
# Digit classifier with segment head (must match training)
# ============================================================

class SmallDigitCNNWithSegments(nn.Module):
    def __init__(self, num_digits, num_segments=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.digit_head = nn.Linear(64, num_digits)
        self.segment_head = nn.Linear(64, num_segments)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.digit_head(x), self.segment_head(x)


# ============================================================
# Segment truth table (fixed)
# ============================================================

DIGIT_TO_SEGMENTS = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1],
}


# ============================================================
# Pipeline
# ============================================================

class MeterVisionPipelineV3:

    MIN_METER_SCORE = 0.5
    MIN_DIGIT_SCORE = 0.4
    X_OVERLAP_THRESHOLD = 0.5

    OUTPUT_IMAGE = "meter_reading_debug.png"

    # --------------------------------------------------------
    def __init__(
        self,
        meter_model_pth,
        meter_label_map_json,
        digit_localizer_pth,
        digit_localizer_label_map_json,
        digit_classifier_pth,
        digit_classifier_meta_json,
        debug=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        print("Using device:", self.device)

        # ---- Meter localizer ----
        self.meter_label_map = self._load_label_map(meter_label_map_json)
        self.meter_model = self._load_rcnn(
            meter_model_pth,
            num_classes=max(self.meter_label_map.keys()) + 1
        )

        # ---- Digit localizer ----
        self.digit_loc_label_map = self._load_label_map(digit_localizer_label_map_json)
        self.digit_localizer = self._load_rcnn(
            digit_localizer_pth,
            num_classes=max(self.digit_loc_label_map.keys()) + 1
        )

        # ---- Digit classifier (segment-aware) ----
        with open(digit_classifier_meta_json, "r") as f:
            meta = json.load(f)

        self.digit_classes = meta["classes"]
        img_size = tuple(meta["image_size"])
        mean = meta["normalize"]["mean"]
        std = meta["normalize"]["std"]

        self.classifier_tfms = transforms.Compose([
            SegmentNormalize(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.digit_classifier = SmallDigitCNNWithSegments(
            num_digits=len(self.digit_classes)
        )
        self.digit_classifier.load_state_dict(
            torch.load(digit_classifier_pth, map_location=self.device)
        )
        self.digit_classifier.to(self.device)
        self.digit_classifier.eval()

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------
    @staticmethod
    def _load_label_map(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {int(k): v for k, v in data["label_map"].items()}

    def _load_rcnn(self, pth, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(pth, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def x_overlap_ratio(a, b):
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        return overlap / min(a[2] - a[0], b[2] - b[0])

    def suppress_overlaps(self, detections):
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept, rejected = [], []
        for d in detections:
            if any(self.x_overlap_ratio(d["bbox"], k["bbox"]) > self.X_OVERLAP_THRESHOLD for k in kept):
                rejected.append(d)
            else:
                kept.append(d)
        return kept, rejected

    # --------------------------------------------------------
    # Debug drawing
    # --------------------------------------------------------
    def _draw_meter_detections(self, image, detections):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for d in detections:
            x1,y1,x2,y2 = map(int, d["bbox"])
            draw.rectangle((x1,y1,x2,y2), outline="blue", width=3)
            draw.text((x1, max(0,y1-14)), d["label"], fill="blue")
        return img

    def _draw_digit_detections(self, img, kept, rejected):
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for d in rejected:
            x1,y1,x2,y2 = map(int, d["bbox"])
            draw.rectangle((x1,y1,x2,y2), outline="red", width=2)
        for d in kept:
            x1,y1,x2,y2 = map(int, d["bbox"])
            draw.rectangle((x1,y1,x2,y2), outline="green", width=3)
        return out

    # --------------------------------------------------------
    # Segment-aware digit scoring
    # --------------------------------------------------------
    def _score_digit(self, digit_probs, segment_probs, digit_idx):
        pattern = DIGIT_TO_SEGMENTS[int(self.digit_classes[digit_idx])]
        score = digit_probs[digit_idx]
        for p, req in zip(segment_probs, pattern):
            score *= p if req else (1 - p)
        return score

    # --------------------------------------------------------
    # Main pipeline
    # --------------------------------------------------------
    def run(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = F.to_tensor(image).to(self.device)

        # ---- Meter localization ----
        with torch.no_grad():
            out = self.meter_model([tensor])[0]

        screens = []
        detections = []

        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if score < self.MIN_METER_SCORE:
                continue
            name = self.meter_label_map.get(int(label))
            bbox = box.cpu().tolist()
            detections.append({"bbox": bbox, "label": name})
            if name == "digital_meter_screen":
                screens.append((score.item(), bbox))

        if self.debug:
            self._draw_meter_detections(image, detections).show("Meter localization")

        if not screens:
            raise RuntimeError("No digital_meter_screen detected")

        screens.sort(reverse=True)
        _, (sx1,sy1,sx2,sy2) = screens[0]
        screen = image.crop((int(sx1),int(sy1),int(sx2),int(sy2)))

        # ---- Digit localization ----
        with torch.no_grad():
            d_out = self.digit_localizer([F.to_tensor(screen).to(self.device)])[0]

        raw_digits = [
            {"bbox": box.cpu().tolist(), "score": score.item()}
            for box, score in zip(d_out["boxes"], d_out["scores"])
            if score >= self.MIN_DIGIT_SCORE
        ]

        kept, rejected = self.suppress_overlaps(raw_digits)

        if self.debug:
            self._draw_digit_detections(screen, kept, rejected).show(
                "Digit localization (green=kept, red=rejected)"
            )

        # ---- Digit classification (segment-aware) ----
        digits = []
        for d in kept:
            x1,y1,x2,y2 = map(int, d["bbox"])
            crop = screen.crop((x1,y1,x2,y2))
            x = self.classifier_tfms(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                digit_logits, seg_logits = self.digit_classifier(x)
                digit_probs = torch.softmax(digit_logits, dim=1)[0]
                seg_probs = torch.sigmoid(seg_logits)[0]

            scores = [
                self._score_digit(digit_probs, seg_probs, i)
                for i in range(len(self.digit_classes))
            ]

            idx = int(torch.argmax(torch.tensor(scores)))
            digits.append({
                "digit": self.digit_classes[idx],
                "center_x": (x1 + x2) / 2,
                "bbox": [int(sx1)+x1, int(sy1)+y1, int(sx1)+x2, int(sy1)+y2],
            })

        digits.sort(key=lambda d: d["center_x"])
        reading = "".join(d["digit"] for d in digits)

        # ---- Final visualization ----
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        for d in digits:
            x1,y1,x2,y2 = d["bbox"]
            draw.rectangle((x1,y1,x2,y2), outline="green", width=3)
            draw.text((x1, max(0,y1-14)), d["digit"], fill="green")

        vis.save(self.OUTPUT_IMAGE)
        if self.debug:
            vis.show("Final reading")

        print("Final meter reading:", reading)
        return reading


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    pipeline = MeterVisionPipelineV3(
        meter_model_pth="meter_localizer_fasterrcnn_state_dict.pth",
        meter_label_map_json="meter_localizer_label_map.json",
        digit_localizer_pth="digit_localizer_fasterrcnn_state_dict.pth",
        digit_localizer_label_map_json="digit_localizer_label_map.json",
        digit_classifier_pth="digit_classifier_smallcnn_segments.pth",
        digit_classifier_meta_json="digit_classifier_smallcnn_segments_meta.json",
        debug=True,
    )


    image_path = "example_image1.png"
    image_path = "example_image2.png"
    pipeline.run(image_path)
