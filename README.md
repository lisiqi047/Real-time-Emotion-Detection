# Multi-Face Emotion Detector / 多人人脸情绪检测

## 简介 (Introduction)
本项目基于YOLOv8和OpenCV，实现了多人人脸与情绪的实时检测。支持多种情绪识别，检测速度快，结构简洁，适用于需要实时多脸情绪分析的场景。

This project uses YOLOv8 and OpenCV for real-time multi-face and emotion detection. It supports various emotion recognition, features fast detection and simple structure, and is suitable for scenarios requiring real-time multi-face emotion analysis.

## 依赖 (Dependencies)
- Python >= 3.7
- opencv-python
- numpy
- ultralytics

## 安装依赖 (Install dependencies)
```bash
pip install -r requirements.txt
```

## 用法示例 (Usage Example)
```bash
python simple_multi_face_detector.py
```

请根据实际模型路径修改脚本中的 `model_path` 变量。

## 注意事项 (Notes)
- 请勿上传大型数据集和模型权重到GitHub。
- 如需使用自定义模型，请将权重文件路径指向本地文件。
- 仅供学习和研究使用。 