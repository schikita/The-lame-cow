import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import sys
from datetime import datetime
from ultralytics import YOLO
from loguru import logger

try:
    from yolox.data.data_augment import preproc
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info, postprocess
    from yolox.utils.visualize import plot_tracking
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracking_utils.timer import Timer
    
    BYTETRACK_AVAILABLE = True
    logger.info("✅ YOLOX модули загружены успешно")
except ImportError as e:
    BYTETRACK_AVAILABLE = False
    logger.warning(f"⚠️ YOLOX недоступен: {e}")
    logger.warning("Работает только простой режим камеры")


class Predictor(object):
    """Класс для предсказаний"""
    def __init__(self, model, exp, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        
        return outputs, img_info


class IntegratedCamera:
    def __init__(self, args):
        self.args = args
        self.cap = None
        self.predictor = None
        self.tracker = None
        self.timer = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Инициализация камеры
        self.init_camera()
        
        # Инициализация AI (если доступно и требуется)
        if BYTETRACK_AVAILABLE and args.mode in ['track', 'both']:
            self.init_ai_tracking()
        elif args.mode in ['track', 'both'] and not BYTETRACK_AVAILABLE:
            logger.error("❌ AI режим недоступен - YOLOX не установлен")
            logger.info("Переключение на простой режим камеры...")
            self.args.mode = 'simple'
    
    def init_camera(self):
        """Инициализация камеры"""
        logger.info(f"🎥 Инициализация камеры с ID: {self.args.camera_id}")
        self.cap = cv2.VideoCapture(self.args.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Не удалось открыть камеру с ID {self.args.camera_id}")
        
        # Настройка разрешения (если указано)
        if self.args.width and self.args.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        
        # Получение параметров камеры
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        logger.info(f"✅ Камера: {self.width}x{self.height} @ {self.fps}fps")
        
        # Инициализация записи видео (если нужно)
        if self.args.save_video:
            self.init_video_writer()
    
    def init_video_writer(self):
        """Инициализация записи видео"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.args.mode == 'simple':
            filename = f"simple_camera_{timestamp}.mp4"
        elif self.args.mode == 'track':
            filename = f"tracking_{timestamp}.mp4"
        else:
            filename = f"integrated_{timestamp}.mp4"
        
        self.output_path = osp.join(self.args.output_dir, filename)
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        logger.info(f"📹 Видео будет сохранено: {self.output_path}")
    
    def init_ai_tracking(self):
        """Инициализация AI трекинга"""
        try:
            logger.info("🤖 Инициализация AI трекинга...")
            
            # Проверка файлов модели
            if not self.args.exp_file:
                # Попробуем найти стандартные файлы
                possible_exp_files = [
                    "yolox/exp/yolox_s.py",
                    "exps/example/yolox_voc/yolox_voc_s.py",
                    "exps/default/yolox_s.py"
                ]
                for exp_file in possible_exp_files:
                    if os.path.exists(exp_file):
                        self.args.exp_file = exp_file
                        break
                else:
                    logger.warning("⚠️ Файл эксперимента не найден, используется стандартная конфигурация")
                    self.args.name = "yolox-s"
            
            # Загрузка эксперимента
            if self.args.exp_file:
                exp = get_exp(self.args.exp_file, self.args.name)
            else:
                # Создаем минимальную конфигурацию
                from yolox.exp.yolox_base import Exp
                exp = Exp()
                exp.num_classes = 80  # COCO classes
                exp.test_conf = self.args.conf
                exp.nmsthre = self.args.nms
                exp.test_size = (self.args.tsize, self.args.tsize)
            
            # Настройка параметров
            if self.args.conf is not None:
                exp.test_conf = self.args.conf
            if self.args.nms is not None:
                exp.nmsthre = self.args.nms
            if self.args.tsize is not None:
                exp.test_size = (self.args.tsize, self.args.tsize)
            
            # Устройство
            self.device = torch.device("cuda" if self.args.device == "gpu" and torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 Используется устройство: {self.device}")
            
            # Модель
            model = exp.get_model().to(self.device)
            model.eval()
            
            # Загрузка чекпоинта (если указан)
            if self.args.ckpt and os.path.exists(self.args.ckpt):
                ckpt = torch.load(self.args.ckpt, map_location="cpu")
                model.load_state_dict(ckpt["model"])
                logger.info(f"✅ Модель загружена из: {self.args.ckpt}")
            else:
                logger.warning("⚠️ Чекпоинт не найден, используется неинициализированная модель")
            
            # Оптимизация модели
            if self.args.fuse:
                model = fuse_model(model)
            if self.args.fp16:
                model = model.half()
            
            # Создание предиктора
            self.predictor = Predictor(
                model, exp, device=self.device, fp16=self.args.fp16
            )
            
            # Инициализация трекера
            self.tracker = BYTETracker(self.args, frame_rate=self.fps)
            self.timer = Timer()
            
            logger.info("✅ AI трекинг инициализирован успешно!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации AI: {e}")
            import traceback
            traceback.print_exc()
            self.predictor = None
            self.tracker = None
    
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
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Красная точка
            
            cv2.putText(frame, info_text, (10, self.height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Сохранение кадра (если запись включена)
            if recording and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            # Отображение
            cv2.imshow("Integrated Camera - Simple Mode", frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:  # ESC или Q
                break
            elif key in [ord('s'), ord('S')]:  # Скриншот
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📷 Скриншот сохранен: {screenshot_path}")
            elif key in [ord('r'), ord('R')]:  # Переключение записи
                if not hasattr(self, 'video_writer'):
                    self.init_video_writer()
                recording = not recording
                logger.info(f"📹 Запись {'включена' if recording else 'выключена'}")
    
    def run_ai_tracking(self):
        """Режим AI трекинга"""
        if not self.predictor or not self.tracker:
            logger.error("❌ AI трекинг недоступен")
            logger.info("🔄 Переключение на простой режим...")
            self.run_simple_camera()
            return
        
        logger.info("🤖 Запуск AI трекинга")
        logger.info("Управление: ESC/Q - выход")
        
        results = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("❌ Не удалось получить кадр с камеры")
                break
            
            self.frame_count += 1
            
            try:
                # AI обработка
                outputs, img_info = self.predictor.inference(frame, self.timer)
                
                if outputs[0] is not None:
                    # Трекинг
                    online_targets = self.tracker.update(
                        outputs[0], 
                        [img_info['height'], img_info['width']], 
                        self.predictor.test_size
                    )
                    
                    # Подготовка данных для отображения
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                        
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            
                            # Сохранение результатов
                            results.append(
                                f"{self.frame_count},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                                f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    
                    self.timer.toc()
                    
                    # Отрисовка результатов
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, 
                        frame_id=self.frame_count, 
                        fps=1. / max(1e-5, self.timer.average_time)
                    )
                else:
                    self.timer.toc()
                    online_im = frame
                    
            except Exception as e:
                logger.error(f"❌ Ошибка AI обработки: {e}")
                online_im = frame
                self.timer.toc() if hasattr(self, 'timer') else None
            
            # Сохранение видео
            if self.args.save_video and hasattr(self, 'video_writer'):
                self.video_writer.write(online_im)
            
            # Отображение
            cv2.imshow("Integrated Camera - AI Tracking", online_im)
            
            # Логирование прогресса
            if self.frame_count % 30 == 0:
                if hasattr(self, 'timer') and self.timer.average_time > 0:
                    fps = 1. / max(1e-5, self.timer.average_time)
                else:
                    fps = self.frame_count / (time.time() - self.start_time)
                logger.info(f'Frame {self.frame_count} | FPS: {fps:.1f}')
            
            # Выход
            if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
                break
        
        # Сохранение результатов трекинга
        if results and self.args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.args.output_dir, exist_ok=True)
            results_path = osp.join(self.args.output_dir, f"tracking_results_{timestamp}.txt")
            with open(results_path, 'w') as f:
                f.writelines(results)
            logger.info(f"💾 Результаты трекинга сохранены: {results_path}")
    
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
            if ai_active and self.predictor and self.tracker:
                try:
                    outputs, img_info = self.predictor.inference(frame, self.timer)
                    
                    if outputs[0] is not None:
                        online_targets = self.tracker.update(
                            outputs[0], 
                            [img_info['height'], img_info['width']], 
                            self.predictor.test_size
                        )
                        
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        
                        for t in online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                            
                            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                        
                        self.timer.toc()
                        frame = plot_tracking(
                            img_info['raw_img'], online_tlwhs, online_ids, 
                            frame_id=self.frame_count, 
                            fps=1. / max(1e-5, self.timer.average_time)
                        )
                    else:
                        self.timer.toc()
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
            cv2.imshow("Integrated Camera - Combined Mode", frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:  # Выход
                break
            elif key in [ord('t'), ord('T')] and ai_enabled:  # Переключение AI
                ai_active = not ai_active
                logger.info(f"🔄 AI трекинг {'включен' if ai_active else 'выключен'}")
            elif key in [ord('s'), ord('S')]:  # Скриншот
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📷 Скриншот сохранен: {screenshot_path}")
    
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
        "Integrated Camera Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python integrated_camera.py simple                    # Простая камера
  python integrated_camera.py track                     # AI трекинг
  python integrated_camera.py both                      # Комбинированный режим
  python integrated_camera.py simple --save_video       # С сохранением видео
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
    parser.add_argument("--save_results", action="store_true", help="Сохранять результаты трекинга")
    parser.add_argument("--output_dir", type=str, default="./output", help="Папка для сохранения")
    
    # AI параметры
    parser.add_argument("-f", "--exp_file", type=str, default=None, help="Файл эксперимента")
    parser.add_argument("-n", "--name", type=str, default=None, help="Имя модели")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="Чекпоинт модели")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство: cpu или gpu")
    parser.add_argument("--conf", type=float, default=0.5, help="Порог уверенности")
    parser.add_argument("--nms", type=float, default=0.45, help="Порог NMS")
    parser.add_argument("--tsize", type=int, default=640, help="Размер входного изображения")
    parser.add_argument("--fp16", action="store_true", help="Использовать FP16")
    parser.add_argument("--fuse", action="store_true", help="Слить conv и bn")
    
    # Параметры трекинга
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Порог трекинга")
    parser.add_argument("--track_buffer", type=int, default=30, help="Буфер трекинга")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Порог сопоставления")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Порог соотношения сторон")
    parser.add_argument("--min_box_area", type=float, default=10, help="Минимальная площадь бокса")
    parser.add_argument("--mot20", action="store_true", help="Режим MOT20")
    
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
        print("python integrated_camera.py simple      # Простая камера")
        print("python integrated_camera.py track       # AI трекинг")
        print("python integrated_camera.py both        # Комбинированный")
        sys.exit(0)
    
    # Парсинг аргументов
    parser = make_parser()
    args = parser.parse_args()
    
    logger.info("🚀 Запуск интегрированного приложения камеры")
    logger.info(f"📋 Режим: {args.mode}")
    
    # Создание и запуск приложения
    app = IntegratedCamera(args)
    app.run()