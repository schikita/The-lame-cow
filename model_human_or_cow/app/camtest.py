import cv2
import time
import argparse
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import Predictor
except ImportError:
    print("⚠️ Модуль predict.py не найден. AI режимы будут недоступны.")
    Predictor = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics не установлен. Human detection будет недоступен.")
    YOLO_AVAILABLE = False


class HumanDetector:
    """Детектор людей на основе YOLO"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = None
        self.conf_threshold = conf_threshold
        
        if YOLO_AVAILABLE:
            try:
                print(f"🔄 Загрузка YOLO модели: {model_path}")
                self.model = YOLO(model_path)
                print("✅ YOLO модель загружена успешно")
            except Exception as e:
                print(f"❌ Ошибка загрузки YOLO: {e}")
                self.model = None
        else:
            print("❌ YOLO недоступен")
    
    def detect(self, frame):
        """Детекция объектов на кадре"""
        if not self.model:
            return []
        
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Координаты bbox
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Уверенность
                        conf = float(box.conf[0])
                        
                        # Класс
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Проверяем, является ли объект человеком
                        is_human = class_name.lower() == 'person'
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_human': is_human
                        }
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"❌ Ошибка детекции: {e}")
            return []


class CameraHandler:
    """Обработчик камеры с различными режимами детекции"""
    
    def __init__(self, mode="human", camera_id=0, weights_path=None, target_fps=30):
        self.mode = mode
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_count = 0
        self.start_time = time.time()
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Не удалось открыть камеру {camera_id}")
        
        # Настройка камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Получаем реальные размеры
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Камера инициализирована: {self.width}x{self.height}")
        
        # Инициализация детекторов
        self.human_detector = None
        self.predictor = None
        self.enable_human_detection = True
        
        # Инициализация в зависимости от режима
        if mode in ["human", "both", "both_with_human"]:
            self.human_detector = HumanDetector()
        
        if mode in ["track", "both", "both_with_human"]:
            if Predictor:
                try:
                    weights = weights_path or "artifacts/train-seg/weights/best.pt"
                    print(f"🔄 Загрузка AI модели: {weights}")
                    self.predictor = Predictor(weights)
                    print("✅ AI модель загружена успешно")
                except Exception as e:
                    print(f"❌ Ошибка загрузки AI модели: {e}")
                    self.predictor = None
            else:
                print("❌ AI модуль недоступен")
    
    def draw_info_panel(self, frame, humans=0, non_humans=0, ai_detections=0):
        """Отрисовка информационной панели"""
        try:
            current_fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
            
            # Размеры панели
            panel_height = 140
            panel_width = 400
            
            # Черный фон панели
            cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
            
            # Белая рамка панели
            cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
            
            # Информация
            info_lines = [
                f"FPS: {current_fps:.1f}",
                f"Frame: {self.frame_count}",
                f"Mode: {self.mode.upper()}",
                f"Humans: {humans}",
                f"Others: {non_humans}",
                f"AI Objects: {ai_detections}"
            ]
            
            # DEBUG вывод каждые 30 кадров
            if self.frame_count % 30 == 0:
                print(f"DEBUG: H={humans}, O={non_humans}, AI={ai_detections}")
            
            for i, line in enumerate(info_lines):
                y = 30 + i * 18
                
                # Цвета для разных типов информации
                if "Humans:" in line and humans > 0:
                    color = (0, 255, 0)  # Зеленый
                elif "Others:" in line and non_humans > 0:
                    color = (0, 0, 255)  # Красный
                elif "AI Objects:" in line and ai_detections > 0:
                    color = (0, 255, 255)  # Желтый
                else:
                    color = (255, 255, 255)  # Белый
                
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Статус индикаторы в правом верхнем углу
            status_x = self.width - 80
            
            # AI статус (зеленый круг)
            if self.predictor:
                cv2.circle(frame, (status_x, 30), 12, (0, 255, 0), -1)
                cv2.putText(frame, "AI", (status_x - 12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Human detection статус (желтый круг)
            if self.human_detector and self.human_detector.model:
                cv2.circle(frame, (status_x, 55), 12, (0, 255, 255), -1)
                cv2.putText(frame, "HD", (status_x - 12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
        except Exception as e:
            print(f"❌ Ошибка отрисовки панели: {e}")
        
        return frame
    
    def draw_human_detections(self, frame, detections):
        """Отрисовка детекций людей на кадре"""
        if not detections:
            return frame
        
        try:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                if det['is_human']:
                    color = (0, 255, 0)  # Зеленый для людей
                    label = f"HUMAN {det['conf']:.2f}"
                else:
                    color = (0, 0, 255)  # Красный для остальных
                    label = f"{det['class_name'].upper()} {det['conf']:.2f}"
                
                # Рамка
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Фон для текста
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Текст
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"❌ Ошибка отрисовки детекций: {e}")
        
        return frame
    
    def run(self):
        """Основной цикл обработки видео"""
        print(f"🎬 Запуск камеры в режиме: {self.mode}")
        print("📋 Управление:")
        print("   Q - выход")
        if self.mode == "both":
            print("   SPACE - переключение human detection")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Ошибка чтения кадра")
                break
            
            self.frame_count += 1
            
            # Инициализируем счетчики
            humans = 0
            non_humans = 0
            ai_detections = 0
            
            try:
                # Обработка в зависимости от режима
                if self.mode == "human":
                    if self.human_detector and self.human_detector.model:
                        detections = self.human_detector.detect(frame)
                        frame = self.draw_human_detections(frame, detections)
                        
                        # Подсчет детекций
                        for det in detections:
                            if det['is_human']:
                                humans += 1
                            else:
                                non_humans += 1
                
                elif self.mode == "track":
                    if self.predictor:
                        try:
                            results = self.predictor.predict(frame)
                            frame = self.predictor.draw_detections(frame, results)
                            
                            # Подсчет AI детекций
                            if results:
                                if hasattr(results, '__len__'):
                                    ai_detections = len(results)
                                else:
                                    ai_detections = 1
                        except Exception as e:
                            print(f"❌ Ошибка AI предсказания: {e}")
                
                elif self.mode == "both":
                    # AI детекция
                    if self.predictor:
                        try:
                            results = self.predictor.predict(frame)
                            frame = self.predictor.draw_detections(frame, results)
                            
                            if results:
                                if hasattr(results, '__len__'):
                                    ai_detections = len(results)
                                else:
                                    ai_detections = 1
                        except Exception as e:
                            print(f"❌ Ошибка AI предсказания: {e}")
                    
                    # Human детекция по клавише
                    if self.enable_human_detection and self.human_detector and self.human_detector.model:
                        detections = self.human_detector.detect(frame)
                        frame = self.draw_human_detections(frame, detections)
                        
                        for det in detections:
                            if det['is_human']:
                                humans += 1
                            else:
                                non_humans += 1
                
                elif self.mode == "both_with_human":
                    # AI детекция
                    if self.predictor:
                        try:
                            results = self.predictor.predict(frame)
                            frame = self.predictor.draw_detections(frame, results)
                            
                            if results:
                                if hasattr(results, '__len__'):
                                    ai_detections = len(results)
                                else:
                                    ai_detections = 1
                        except Exception as e:
                            print(f"❌ Ошибка AI предсказания: {e}")
                    
                    # Human детекция всегда активна
                    if self.human_detector and self.human_detector.model:
                        detections = self.human_detector.detect(frame)
                        frame = self.draw_human_detections(frame, detections)
                        
                        for det in detections:
                            if det['is_human']:
                                humans += 1
                            else:
                                non_humans += 1
            
            except Exception as e:
                print(f"❌ Ошибка обработки кадра: {e}")
            
            # Отрисовываем информационную панель
            frame = self.draw_info_panel(frame, humans, non_humans, ai_detections)
            
            # Показываем кадр
            cv2.imshow('Human Detection', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and self.mode == "both":
                self.enable_human_detection = not self.enable_human_detection
                status = "ON" if self.enable_human_detection else "OFF"
                print(f"🔄 Human detection: {status}")
            
            # FPS ограничение
            if self.target_fps > 0:
                time.sleep(1.0 / self.target_fps)
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Ресурсы освобождены")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Камера с детекцией людей и AI')
    parser.add_argument('mode', choices=['human', 'track', 'both', 'both_with_human'], 
                       help='Режим работы')
    parser.add_argument('--camera', type=int, default=0, help='ID камеры')
    parser.add_argument('--weights', type=str, help='Путь к весам AI модели')
    parser.add_argument('--fps', type=int, default=30, help='Целевой FPS')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    
    args = parser.parse_args()
    
    print("🚀 Запуск системы детекции...")
    print(f"📋 Режим: {args.mode}")
    print(f"📹 Камера: {args.camera}")
    print(f"🎯 FPS: {args.fps}")
    print(f"📊 Confidence: {args.conf}")
    print()
    
    camera_handler = None
    
    try:
        camera_handler = CameraHandler(
            mode=args.mode,
            camera_id=args.camera,
            weights_path=args.weights,
            target_fps=args.fps
        )
        
        camera_handler.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Остановка по Ctrl+C")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        if camera_handler:
            camera_handler.cleanup()


if __name__ == "__main__":
    main()