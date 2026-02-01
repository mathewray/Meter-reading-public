## Quick run order (scripts)

### Update labelled images from label studio

1. `split_meter_dataset.py`

### Dataset preparation

1. `check_categories.py`
2. `generate_meter_localization_dataset.py`
3. `generate_digit_localization_dataset.py`
4. `generate_digit_classification_dataset.py`

### Model fine-tuning (training)

5. `finetune_meter_localizer.py`
6. `finetune_digit_localizer.py`
7. `finetune_digit_classifier_smallcnn.py`

### Inference

8. `infer_meter_localizer.py`
9. `infer_digit_localizer.py`
10. `infer_digit_classifier.py`
11. `meter_vision_pipeline_v3.py`

---

## Step descriptions (short)

### 1. `check_categories.py`

Verifies category IDs and names in the original COCO annotation file.

---

### 2. `generate_meter_localization_dataset.py`

Creates a reduced COCO dataset containing only:

* `digital_meter`
* `digital_meter_screen`
* `gas_meter`

Output:

```
meter_localization_dataset/{train,test}
```

---

### 3. `generate_digit_localization_dataset.py`

Crops meter screens and keeps only digit boxes.
All digits are merged into a single class: `digit`.

Output:

```
digit_localization_dataset/{train,test}
```

---

### 4. `generate_digit_classification_dataset.py`

Crops individual digit boxes and creates a folder-based dataset for classification.

Output:

```
digit_classification_dataset/
  train/0..9
  test/0..9
```

---

### 5. `finetune_meter_localizer.py`

Fine-tunes a Faster R-CNN model to detect meters and meter screens.

Output:

```
meter_localizer_fasterrcnn_state_dict.pth
meter_localizer_label_map.json
```

---

### 6. `finetune_digit_localizer.py`

Fine-tunes a Faster R-CNN model to detect digit bounding boxes inside a screen.

Output:

```
digit_localizer_fasterrcnn_state_dict.pth
digit_localizer_label_map.json
```

---

### 7. `finetune_digit_classifier.py`

Fine-tunes a CNN (ResNet18) to classify digit crops as `0–9`.

Output:

```
digit_classifier_resnet18.pth
digit_classifier_meta.json
```

---

### 8. `meter_vision_pipeline_v2.py`

Runs the full pipeline on a single image:

1. Meter localization
2. Screen crop
3. Digit localization
4. Overlap suppression
5. Digit classification
6. Left-to-right ordering
7. Final meter reading

Input:

* Model files from steps 5–7
* A meter image

Output:

* Meter reading (string)
* Debug image with detected digits
