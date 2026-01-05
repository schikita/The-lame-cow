import argparse
import os
import cv2
import time
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Optional

# Импортируем код автора
try:
    from predict import Predictor, DetectedObject
    PREDICTOR_AVAILABLE = True
    logger.info("✅ Predictor модуль загружен успешно")
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    logger.warning(f"⚠️ Predictor недоступен: {e}")
    logger.warning("Работает только простой режим камеры")


class IntegratedCamera:
    def __init__(self, args):
        self.args = args
        self.cap = None
        self.predictor = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Инициализация камеры
        self.init_camera()
        
        # Инициализация AI (если доступно и требуется)
        if PREDICTOR_AVAILABLE and args.mode in ['track', 'both']:
            self.init_ai_predictor()
        elif args.mode in ['track', 'both'] and not PREDICTOR_AVAILABLE:
            logger.error("❌ AI режим недоступен - Predictor не найден")
            logger.info("Переключение на простой режим камеры...")
            self.args.mode = 'simple'
    
    def init_camera(self):
        """Инициализация камеры"""
        logger.info(f"🎥 Инициализация камеры с ID: {self.args.camera_id}")
        
        # Попробуем разные камеры
        for camera_id in [self.args.camera_id, 0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    logger.info(f"✅ Камера {camera_id} работает")
                    self.args.camera_id = camera_id
                    break
                else:
                    self.cap.release()
            else:
                if self.cap:
                    self.cap.release()
        else:
            raise RuntimeError("❌ Не удалось найти рабочую камеру")
        
        # Настройка разрешения
        if self.args.width and self.args.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        
        # Получение параметров
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        logger.info(f"✅ Камера: {self.width}x{self.height} @ {self.fps}fps")
        
        # Инициализация записи видео
        if self.args.save_video:
            self.init_video_writer()
    
    def init_video_writer(self):
        """Инициализация записи видео"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        mode_name = {
            'simple': 'simple_camera',
            'track': 'ai_tracking', 
            'both': 'combined'
        }.get(self.args.mode, 'camera')
        
        filename = f"{mode_name}_{timestamp}.mp4"
        self.output_path = Path(self.args.output_dir) / filename
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        logger.info(f"📹 Видео будет сохранено: {self.output_path}")
    
    def init_ai_predictor(self):
        """Инициализация AI предиктора"""
        try:
            logger.info("🤖 Инициализация AI предиктора...")
            
            weights_path = Path(self.args.weights)
            if not weights_path.exists():
                # Попробуем найти стандартные веса
                possible_weights = [
                    "artifacts/train-seg/weights/best.pt",
                    "best.pt",
                    "yolov8n-seg.pt"
                ]
                
                for weight_file in possible_weights:
                    if Path(weight_file).exists():
                        weights_path = Path(weight_file)
                        logger.info(f"🔍 Найдены веса: {weights_path}")
                        break
                else:
                    logger.warning("⚠️ Веса модели не найдены")
                    logger.info("💡 Загружаем предтренированную модель YOLOv8n-seg...")
                    weights_path = "yolov8n-seg.pt"  # Ultralytics загрузит автоматически
            
            # Создание предиктора
            self.predictor = Predictor(
                weights=weights_path,
                device=self.args.device
            )
            
            logger.info("✅ AI предиктор инициализирован успешно!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации AI: {e}")
            import traceback
            traceback.print_exc()
            self.predictor = None
    
    def draw_detections(self, frame, detections):
        """Отрисовка детекций на кадре"""
        if not detections:
            return frame
        
        result_frame = frame.copy()
        
        for det in detections:
            # Рисуем бounding box
            if det.bbox_xyxy:
                x1, y1, x2, y2 = map(int, det.bbox_xyxy)
                
                # Цвет в зависимости от класса
                color = (0, 255, 0) if det.cls_name == 'cow' else (255, 0, 0)
                
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # Подпись
                label = f"{det.cls_name}: {det.conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Фон для текста
                cv2.rectangle(result_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Текст
                cv2.putText(result_frame, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Рисуем сегментацию (если есть)
            if det.seg_xy and len(det.seg_xy) >= 6:  # Минимум 3 точки
                try:
                    # Преобразуем в numpy array точек
                    points = []
                    for i in range(0, len(det.seg_xy), 2):
                        if i + 1 < len(det.seg_xy):
                            points.append([int(det.seg_xy[i]), int(det.seg_xy[i + 1])])
                    
                    if len(points) >= 3:
                        import numpy as np
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # Полупрозрачная заливка
                        overlay = result_frame.copy()
                        color = (0, 255, 0) if det.cls_name == 'cow' else (255, 0, 0)
                        cv2.fillPoly(overlay, [pts], color)
                        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)
                        
                        # Контур
                        cv2.polylines(result_frame, [pts], True, color, 2)
                        
                except Exception as e:
                    logger.debug(f"Ошибка отрисовки сегментации: {e}")
        
        return result_frame
    
    def run_simple_camera(self):
        """Простой режим камеры"""
        logger.info("🎥 Запуск простого режима камеры")
        logger.info("Управление: ESC/Q - выход, S - скриншот, R - запись")
        
        recording = False
        screenshot_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("❌ Не удалось получить кадр с камеры")
                break
            
            self.frame_count += 1
            
            # Добавление информации на кадр
            current_fps = self.frame_count / (time.time() - self.start_time)
            info_text = f"FPS: {current_fps:.1f} | Frame: {self.frame_count}"
            
            if recording:
                info_text += " | ⏺ REC"
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            
            cv2.putText(frame, info_text, (10, self.height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Сохранение кадра
            if recording and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            # Отображение
            cv2.imshow("Camera - Simple Mode", frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('s'), ord('S')]:
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📷 Скриншот сохранен: {screenshot_path}")
            elif key in [ord('r'), ord('R')]:
                if not hasattr(self, 'video_writer'):
                    self.init_video_writer()
                recording = not recording
                logger.info(f"📹 Запись {'включена' if recording else 'выключена'}")
    
    def run_ai_tracking(self):
        """Режим AI детекции"""
        if not self.predictor:
            logger.error("❌ AI предиктор недоступен")
            logger.info("🔄 Переключение на простой режим...")
            self.run_simple_camera()
            return
        
        logger.info("🤖 Запуск AI детекции")
        logger.info("Управление: ESC/Q - выход, S - скриншот")
        
        screenshot_count = 0
        detection_results = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("❌ Не удалось получить кадр с камеры")
                break
            
            self.frame_count += 1
            
            try:
                # AI предсказание
                # Сохраняем временный кадр для предсказания
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                predictions = self.predictor.predict(
                    source=temp_path,
                    conf=self.args.conf,
                    iou=self.args.iou,
                    save=False
                )
                
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Получаем детекции для первого (и единственного) кадра
                detections = predictions[0] if predictions else []
                
                # Отрисовка детекций
                result_frame = self.draw_detections(frame, detections)
                
                # Сохраняем результаты
                if detections:
                    frame_results = {
                        'frame': self.frame_count,
                        'timestamp': time.time(),
                        'detections': [
                            {
                                'class': det.cls_name,
                                'confidence': det.conf,
                                'bbox': det.bbox_xyxy
                            }
                            for det in detections
                        ]
                    }
                    detection_results.append(frame_results)
                
            except Exception as e:
                logger.error(f"❌ Ошибка AI обработки: {e}")
                result_frame = frame
            
            # Добавляем информацию на кадр
            current_fps = self.frame_count / (time.time() - self.start_time)
            total_detections = sum(len(r['detections']) for r in detection_results)
            
            info_text = f"FPS: {current_fps:.1f} | Frame: {self.frame_count} | Det: {total_detections}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Сохранение видео
            if self.args.save_video and hasattr(self, 'video_writer'):
                self.video_writer.write(result_frame)
            
            # Отображение
            cv2.imshow("Camera - AI Detection", result_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('s'), ord('S')]:
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"ai_screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                logger.info(f"📷 AI скриншот сохранен: {screenshot_path}")
            
            # Логирование прогресса
            if self.frame_count % 30 == 0:
                current_detections = len(detections) if 'detections' in locals() else 0
                logger.info(f'Frame {self.frame_count} | FPS: {current_fps:.1f} | Current detections: {current_detections}')
        
        # Сохранение результатов
        if detection_results and self.args.save_results:
            self.save_detection_results(detection_results)
    
    def run_combined_mode(self):
        """Комбинированный режим"""
        logger.info("🔄 Запуск комбинированного режима")
        logger.info("Управление: ESC/Q - выход, T - переключение AI, S - скриншот")
        
        ai_enabled = self.predictor is not None
        ai_active = ai_enabled
        screenshot_count = 0
        
        if not ai_enabled:
            logger.warning("⚠️ AI недоступен, работает простой режим")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("❌ Не удалось получить кадр с камеры")
                break
            
            self.frame_count += 1
            original_frame = frame.copy()
            
            # AI обработка (если включена)
            if ai_active and self.predictor:
                try:
                    temp_path = "temp_frame.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    predictions = self.predictor.predict(
                        source=temp_path,
                        conf=self.args.conf,
                        iou=self.args.iou,
                        save=False
                    )
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    detections = predictions[0] if predictions else []
                    frame = self.draw_detections(frame, detections)
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка AI обработки: {e}")
                    frame = original_frame
            
            # Добавление информации на кадр
            current_fps = self.frame_count / (time.time() - self.start_time)
            status = "AI ON" if ai_active else "AI OFF"
            info_text = f"FPS: {current_fps:.1f} | {status} | Frame: {self.frame_count}"
            
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Индикатор режима
            color = (0, 255, 0) if ai_active else (0, 0, 255)
            cv2.circle(frame, (self.width - 30, 30), 10, color, -1)
            
            # Сохранение видео
            if self.args.save_video and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            # Отображение
            cv2.imshow("Camera - Combined Mode", frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('t'), ord('T')] and ai_enabled:
                ai_active = not ai_active
                logger.info(f"🔄 AI детекция {'включена' if ai_active else 'выключена'}")
            elif key in [ord('s'), ord('S')]:
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"combined_screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📷 Скриншот сохранен: {screenshot_path}")
    
    def save_detection_results(self, results):
        """Сохранение результатов детекции"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(self.args.output_dir) / f"detection_results_{timestamp}.txt"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            f.write("frame,class,confidence,x1,y1,x2,y2\n")
            for frame_result in results:
                frame_num = frame_result['frame']
                for det in frame_result['detections']:
                    if det['bbox']:
                        x1, y1, x2, y2 = det['bbox']
                        f.write(f"{frame_num},{det['class']},{det['confidence']:.3f},"
                               f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}\n")
        
        logger.info(f"💾 Результаты детекции сохранены: {results_path}")
    
    def run(self):
        """Главный метод запуска"""
        try:
            if self.args.mode == 'simple':
                self.run_simple_camera()
            elif self.args.mode == 'track':
                self.run_ai_tracking()
            elif self.args.mode == 'both':
                self.run_combined_mode()
            else:
                logger.error(f"❌ Неизвестный режим: {self.args.mode}")
        
        except KeyboardInterrupt:
            logger.info("👋 Остановка по Ctrl+C")
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()
        logger.info("🧹 Ресурсы освобождены")


def make_parser():
    parser = argparse.ArgumentParser(
        "Integrated Camera Application with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python integrated_camera_fixed.py simple                    # Простая камера
  python integrated_camera_fixed.py track                     # AI детекция
  python integrated_camera_fixed.py both                      # Комбинированный режим
  python integrated_camera_fixed.py simple --save_video       # С сохранением видео
        """
    )
    
    # Основные параметры
    parser.add_argument("mode", choices=["simple", "track", "both"], 
                       help="Режим работы")
    
    # Параметры камеры
    parser.add_argument("--camera_id", type=int, default=0, help="ID камеры")
    parser.add_argument("--width", type=int, default=None, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=None, help="Высота кадра")
    
    # Сохранение
    parser.add_argument("--save_video", action="store_true", help="Сохранять видео")
    parser.add_argument("--save_results", action="store_true", help="Сохранять результаты детекции")
    parser.add_argument("--output_dir", type=str, default="./output", help="Папка для сохранения")
    
    # AI параметры (совместимые с кодом автора)
    parser.add_argument("--weights", type=str, default="artifacts/train-seg/weights/best.pt", 
                       help="Путь к весам модели")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство: cpu или cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--iou", type=float, default=0.7, help="Порог IoU для NMS")
    
    return parser


if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""), 
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    # Проверка аргументов
    if len(sys.argv) == 1:
        parser = make_parser()
        parser.print_help()
        print("\n🚀 Быстрый старт:")
        print("python integrated_camera_fixed.py simple      # Простая камера")
        print("python integrated_camera_fixed.py track       # AI детекция") 
        print("python integrated_camera_fixed.py both        # Комбинированный")
        sys.exit(0)
    
    # Парсинг аргументов
    parser = make_parser()
    args = parser.parse_args()
    
    logger.info("🚀 Запуск интегрированного приложения камеры")
    logger.info(f"📋 Режим: {args.mode}")
    
    # Создание и запуск приложения
    app = IntegratedCamera(args)
    app.run()