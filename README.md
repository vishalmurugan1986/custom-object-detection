# Custom Object Detection from Scratch (Faster R-CNN)

## Overview

This project implements a **complete object detection pipeline trained entirely from scratch** (no pre-trained weights).
A **Faster R-CNN–style model with a custom CNN backbone** is designed, trained, evaluated, and demonstrated on a custom dataset.

**Focus areas:**

* Dataset handling
* Model design
* Training from scratch
* Evaluation (mAP, FPS, model size)
* Inference & visualization

---

## Dataset

* **Format:** Pascal VOC (`.xml`)
* **Classes (5):** background, person, car, dog, bicycle

```
data/
├── images/
├── annotations/
├── train.txt
└── val.txt
```

> Dataset is intentionally small to **validate the full detection pipeline**, not benchmark accuracy.

---

## Model Architecture

* Faster R-CNN–style detector
* **Custom CNN backbone**
* Region Proposal Network (RPN)
* Classification + Bounding Box Regression heads
* **Trained fully from scratch**

**Design choice:** Lightweight backbone for faster inference and clarity.

---

## Training

* Optimizer: **Adam**
* Learning Rate: `1e-3`
* Epochs: `20`
* Batch Size: `4`
* Device: **CPU**

**Losses:**

* Cross-Entropy (classification)
* Smooth L1 (bounding box)

Training loss decreased from **~232 → ~28**, confirming correct optimization.

---

## Evaluation

| Metric          | Result                      |
| --------------- | --------------------------- |
| Model Size      | **26.15 MB**                |
| Inference Speed | **~6–8 FPS (CPU)**          |
| mAP@0.5         | Approximate (small dataset) |

> Evaluation emphasizes **pipeline correctness over accuracy**.

---

## Inference Demo

The trained model performs inference on unseen images and visualizes predicted bounding boxes and class labels.

![Detection Demo](demo.gif)

---

## Accuracy vs Speed Trade-off

* Lightweight model → faster inference
* Limited data → lower accuracy
* Easily scalable with more data and deeper backbone

---

## Limitations

* Small dataset
* Synthetic/manual annotations
* CPU-only training
* Simplified ROI pooling

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
python inference.py
```

---

## Repository Structure

```
custom-object-detection/
├── data/
├── models/
├── train.py
├── inference.py
├── eval.py
├── eval_utils.py
├── model.pth
├── demo.gif
├── output_demo.jpg
├── requirements.txt
├── Custom_Object_Detection_Report.pdf
└── README.md
```

---

## Conclusion

This project demonstrates a **fully functional object detection system trained from scratch**, covering dataset preparation, model design, training, evaluation, and real-time inference.

---


