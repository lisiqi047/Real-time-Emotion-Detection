#!/usr/bin/env python3
"""
简化的多个人脸检测器 - 避免复杂平滑处理
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

class SimpleMultiFaceDetector:
    def __init__(self, model_path):
        """
        简化的多个人脸检测器
        
        Args:
            model_path: YOLOv8情绪识别模型路径
        """
        self.emotion_model = YOLO(model_path)
        
        # 强制使用DNN人脸检测器
        print("使用DNN人脸检测器")
        self.use_dnn_face = True
        
        # 尝试加载DNN人脸检测器
        try:
            self.face_net = cv2.dnn.readNet(
                cv2.data.haarcascades + 'opencv_face_detector_uint8.pb',
                cv2.data.haarcascades + 'opencv_face_detector.pbtxt'
            )
            print("✓ DNN人脸检测器加载成功")
        except Exception as e:
            print(f"✗ DNN人脸检测器加载失败: {e}")
            print("尝试下载DNN模型文件...")
            
            # 尝试下载模型文件
            import urllib.request
            import os
            
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            try:
                # 下载prototxt文件
                if not os.path.exists('deploy.prototxt'):
                    print("下载prototxt文件...")
                    urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
                
                # 下载caffemodel文件
                if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
                    print("下载caffemodel文件...")
                    urllib.request.urlretrieve(caffemodel_url, 'res10_300x300_ssd_iter_140000.caffemodel')
                
                # 重新加载DNN检测器
                self.face_net = cv2.dnn.readNet('res10_300x300_ssd_iter_140000.caffemodel', 'deploy.prototxt')
                print("✓ DNN人脸检测器重新加载成功")
                
            except Exception as download_error:
                print(f"✗ 模型文件下载失败: {download_error}")
                print("回退到Haar级联分类器")
                self.use_dnn_face = False
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 保留Haar作为备用
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 情绪类别名称
        self.emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # 置信度阈值
        self.conf_threshold = 0.3
        
        # 检测频率控制
        self.last_detection_time = 0
        self.detection_interval = 0.2  # 每0.2秒检测一次
        
    def detect_faces(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            
        Returns:
            faces: 人脸边界框列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.use_dnn_face:
            # 使用DNN人脸检测器
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            h, w = image.shape[:2]
            
            # 置信度阈值
            confidence_threshold = 0.5
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype(int)
                    
                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # 确保检测框有效且足够大
                    if x2 > x and y2 > y and (x2-x) > 30 and (y2-y) > 30:
                        faces.append((x, y, x2-x, y2-y))
            
            return faces
        else:
            # 使用Haar级联检测器
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            return faces
    
    def detect_emotions(self, image, show_faces=True):
        """
        检测图像中的情绪
        
        Args:
            image: 输入图像
            show_faces: 是否显示人脸检测框
            
        Returns:
            results: 检测结果列表
        """
        # 检测人脸
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return []
        
        results = []
        
        for (x, y, w, h) in faces:
            # 扩展人脸区域，确保包含完整的面部表情
            margin = int(min(w, h) * 0.2)  # 20%的边距
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # 提取人脸区域
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # 使用YOLOv8进行情绪识别
            emotion_results = self.emotion_model(face_roi)
            
            # 只保留置信度最高的情绪
            best_result = None
            best_conf = -1
            for r in emotion_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf > self.conf_threshold and conf > best_conf:
                            cls = int(box.cls[0])
                            emotion = self.emotion_names[cls]
                            box_x1, box_y1, box_x2, box_y2 = map(int, box.xyxy[0])
                            abs_x1 = x1 + box_x1
                            abs_y1 = y1 + box_y1
                            abs_x2 = x1 + box_x2
                            abs_y2 = y1 + box_y2
                            
                            best_result = {
                                'emotion': emotion,
                                'confidence': conf,
                                'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                                'face_bbox': (x, y, w, h)
                            }
                            best_conf = conf
            
            if best_result is not None:
                results.append(best_result)
            else:
                # 强制添加默认情绪
                results.append({
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'bbox': (x, y, x + w, y + h),
                    'face_bbox': (x, y, w, h)
                })
            
            # 显示人脸检测框
            if show_faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return results
    
    def visualize_results(self, image, results):
        """
        可视化检测结果（支持多个人脸）
        
        Args:
            image: 输入图像
            results: 检测结果列表
        """
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            emotion = result['emotion']
            conf = result['confidence']
            
            # 根据情绪选择颜色
            color_map = {
                'happy': (0, 255, 0),    # 绿色
                'sad': (0, 165, 255),    # 橙色
                'angry': (0, 0, 255),    # 红色
                'surprise': (255, 255, 0), # 青色
                'fear': (255, 0, 255),   # 洋红色
                'disgust': (0, 255, 255), # 黄色
                'neutral': (128, 128, 128) # 灰色
            }
            
            color = color_map.get(emotion, (0, 255, 0))
            
            # 绘制情绪边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签（包含人脸编号）
            text = f'Face{i+1}: {emotion} {conf:.2f}'
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 在人脸框右上角添加编号
            cv2.putText(image, f'#{i+1}', (x2-30, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

def detect_camera_simple(model_path):
    """
    简化的实时摄像头情绪检测
    """
    detector = SimpleMultiFaceDetector(model_path)
    
    # 尝试多个摄像头ID
    camera_id = 0
    cap = None
    
    for i in range(3):  # 尝试前3个摄像头
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            camera_id = i
            break
    
    if not cap or not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print(f"使用摄像头 ID: {camera_id}")
    print("开始简化检测，按 'q' 退出...")
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    last_results = []
    last_faces = []
    
    # 性能监控
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # 计算FPS
        fps_counter += 1
        if current_time - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = current_time
        
        # 控制检测频率
        if current_time - detector.last_detection_time >= detector.detection_interval:
            try:
                # 检测情绪
                current_results = detector.detect_emotions(frame, show_faces=False)
                current_faces = detector.detect_faces(frame)
                
                # 直接使用当前结果（简化处理）
                last_results = current_results
                last_faces = current_faces
                detector.last_detection_time = current_time
                
                # 显示检测状态
                if len(current_faces) > 0:
                    emotions = [r['emotion'] for r in current_results]
                    print(f"检测到 {len(current_faces)} 个人脸，情绪: {emotions}")
                
            except Exception as e:
                print(f"检测过程中出现错误: {e}")
                # 保持使用上一次的结果
        
        # 绘制人脸框
        for i, (x, y, w, h) in enumerate(last_faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Face #{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 绘制情绪结果
        frame_with_results = detector.visualize_results(frame, last_results)
        
        # 添加性能信息
        cv2.putText(frame_with_results, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_results, f'Faces: {len(last_faces)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_results, f'Frame: {frame_count}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示多脸检测提示
        if len(last_faces) > 1:
            cv2.putText(frame_with_results, 'MULTI-FACE DETECTED!', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('简化多脸检测', frame_with_results)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("简化检测结束")

if __name__ == "__main__":
    # 配置路径
    model_path = r'D:\ultralytics-main\runs\detect\train\weights(1)\best.pt'
    
    print("简化多脸情绪检测器")
    print("=" * 30)
    print("特点:")
    print("- 支持多个人脸检测")
    print("- 简化处理逻辑")
    print("- 避免复杂平滑")
    print("- 稳定可靠")
    
    detect_camera_simple(model_path) 