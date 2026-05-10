import socket
import select
import struct
import cv2
import numpy as np
import time
import os
import multiprocessing as mp
from multiprocessing.connection import Connection
from queue import Empty, Full
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from ultralytics import YOLO
import logging
import random

@dataclass
class DetectionResult:
    id: int = 0
    detected: bool = False
    confidence: float = 0.0
    bbox: tuple = None
    class_name: str = ""
    timestamp: float = 0.0

class Detector(mp.Process):
    CPX_HEADER_SIZE = 4  # 2 bytes length + 1 byte dst + 1 byte src
    IMG_HEADER_MAGIC = 0xBC
    IMG_HEADER_SIZE = 11  # Magic + Width + Height + Depth + Type + Size
    MAGIC_BYTE = b'FER'
    def __init__( 
            self, 
            model_path: str, 
            confidence_threshold: float, 
            cuda: bool, 
            UDP_IP = "0.0.0.0", 
            UDP_PORT = 5001, 
            ESP32_IP = "192.168.4.1", 
            ESP32_PORT = 5000,
            save_images: bool = False,
            result_queue: mp.Queue = None,
            log_queue: mp.Queue = None,
            pipe_conn: Connection = None,
            display_interval: int = 1):
        super().__init__()
        
        self._udp_ip = UDP_IP
        self._udp_port = UDP_PORT
        self._esp_ip = ESP32_IP
        self._esp_port = ESP32_PORT
        self._socket = None 
        self._is_ready = False
        self._model = None
        self._model_path = model_path
        self._display_interval = display_interval
        self._cuda = cuda
        self._run_start_time = 0.0
        self._confidence_threshold = confidence_threshold
        self._save_images = save_images
        self._fps = 0.0
        self._fps_sum = 0.0
        self._dropped_images = 0
        self._image_counter = 0
        self._detection_counter = 0
        self._inference_time = 0.0
        self._rxTime = 0.0
        self._inf_counter = 0

        self._streams: Dict[Any, Dict] = {}

        self._process: mp.Process = None
        self._running = False
        self._result_queue = result_queue
        self._log_queue = log_queue
        if self._log_queue is not None:
            self._setup_logging()
        self._pipe_conn = pipe_conn
            
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove all non-picklable items
        state.pop('_model', None)
        state.pop('_socket', None)
        state.pop('_process', None)
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None
        self._socket = None
        if self._log_queue is not None:
            self._setup_logging()
    

    def run(self):
        self._running = True
        self._run()

    def _setup_logging(self):
        if self._log_queue is None:
            return
        handler = logging.handlers.QueueHandler(self._log_queue)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        # Remove any handlers inherited from the main process
        root.handlers = [h for h in root.handlers 
                         if isinstance(h, logging.handlers.QueueHandler)]

        logging.info("Logger started for detector")

    def stop(self):
        if not self.is_alive():
            print("Detector: Detector isn't running.")
            return
        
        self._running = False
       # self._stop_event.set()
        self.join(timeout=5.0)

        if self.is_alive():
            print("Detector: Failed to stop detector. Forcing termination")
            self.terminate()
            self.join(timeout=2.0)

        cv2.destroyAllWindows()
        print("Detector stopped.")

    def get_average_inference_time(self):
        if self._image_counter == 0:
            return 0
        return self._inference_time/self._image_counter

    def _send_magic_byte(self):
        print(f"Sending magic byte to ESP32 at {self._esp_ip}:{self._esp_port}")
        try:
            self._socket.sendto(self.MAGIC_BYTE, (self._esp_ip, self._esp_port))
        except Exception as e:
            print(f"Failed to send magic byte: {e}")

    def _run(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._udp_ip, self._udp_port))
        self._run_start_time = time.time()
        self._model = YOLO(model=self._model_path)
        if self._cuda is True:
            self._model.to("cuda")

        self._send_magic_byte()


        while self._running:
            try:
                ready = select.select([self._socket], [], [], 0.01)[0]
                if not ready:
                    continue

                data, addr = self._socket.recvfrom(4096)
            except:
                continue

            if addr not in self._streams:
                self._streams[addr] = {
                    'buffer': bytearray(),
                    'expected_size': None,
                    'receiving': False,
                    'packet_count': 0,
                    'width': 0, 'height': 0, 'depth': 0, 'fmt': 0,
                    'last_frame_time': None
                }
                print(f"New stream from {addr}")
            stream = self._streams[addr]
            

            

            if (len(data) >= self.CPX_HEADER_SIZE + self.IMG_HEADER_SIZE and
                data[self.CPX_HEADER_SIZE] == self.IMG_HEADER_MAGIC):

                try:
                    header_data = data[self.CPX_HEADER_SIZE + 1 : self.CPX_HEADER_SIZE + 1 + 10]
                    width, height, depth, fmt, size = struct.unpack('<HHBBI', header_data)

                    img_start = self.CPX_HEADER_SIZE + 1 + 10
                    stream['buffer'] = bytearray(data[img_start:])
                    stream['expected_size'] = size
                    stream['receiving'] = True
                    stream['packet_count'] = 1
                    stream['width'] = width
                    stream['height'] = height
                    stream['depth'] = depth
                    stream['fmt'] = fmt
                    continue
                except Exception:
                    pass

            
            if stream.get('receiving') and len(data) > self.CPX_HEADER_SIZE:
                stream['buffer'].extend(data[self.CPX_HEADER_SIZE:])
                stream['packet_count'] += 1

                if stream['expected_size'] and len(stream['buffer']) >= stream['expected_size']:
                    self._process_frame(addr, stream)
                    stream['receiving'] = False
                    stream['expected_size'] = None
                    stream['buffer'].clear()

    def _process_frame(self, addr, stream: dict):
        if not self._is_ready:
            self._is_ready = True
            logging.info("Setup time: %.4f", time.time()-self._run_start_time)
            self._pipe_conn.send("READY")
        now = time.time()
        if stream['last_frame_time'] is not None:
            delta = now - stream['last_frame_time']
            fps = 1.0 / delta if delta > 0 else 0.0
            self._fps_sum += fps
            logging.info("FPS: %.2f", fps)
            self._image_counter += 1
            logging.info("Received images: %d", self._image_counter)
            self._fps = self._fps_sum/self._image_counter
            print(f"[{addr}] FPS: {fps:.2f} | Avg: {self._fps:.2f}")

        stream['last_frame_time'] = now
        print(f"[{addr}] Image received in {stream['packet_count']} packets")

        width = stream['width']
        height = stream['height']
        depth = stream['depth']
        fmt = stream['fmt']

        try:
            if fmt == 0:  # raw
                if len(stream['buffer']) != width * height * depth:
                    print(f"[{addr}] Raw size mismatch — possible packet loss")
                    self._dropped_images += 1
                    logging.info("Dropped images: %d", self._dropped_images)
                    stream['receiving'] = False
                    return

                raw_img = np.frombuffer(stream['buffer'], dtype=np.uint8).reshape((height, width))
                compatible_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)  # model expects RGB. Doing this to make grayscale compatible

                if compatible_img is not None:
                    inf_start = time.time()
                    results = self._model(compatible_img, 
                                        conf=self._confidence_threshold,
                                        verbose=False,
                                        half=False)[0]
                    inf_time = time.time() - inf_start
                    self._inf_counter += 1
                    

                    
                    self._inference_time = inf_time

                    print(f"Inference took {inf_time}s. Avg {inf_time/self._inf_counter}")
                    logging.info("Inference time %.4f", inf_time)
                    #detections: List[DetectionResult] = []
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes
                        xywh = boxes.xywh.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy().astype(int)
                        highest_conf = 0
                        det = None
                        for i in range(len(boxes)):
                            if confs[i] > highest_conf:
                                x, y, w, h = map(int, xywh[i])
                                det = DetectionResult(
                                    id=random.randrange(0,1000000),
                                    detected=True,
                                    confidence=float(confs[i]),
                                    bbox=(x, y, w, h),
                                    class_name=results.names[clss[i]],
                                    timestamp=now
                                )
                        if det is not None: # Todo: redo this.
                            self._detection_counter += 1
                            try:
                                if self._result_queue is not None:
                                    # Dump the queue
                                    while not self._result_queue.empty():
                                        try:
                                            self._result_queue.get_nowait()
                                        except Empty:
                                            break

                                    self._result_queue.put_nowait(det)

                            except (Full, Empty):
                                pass  # Queue full (shouldn't happen), or race condition. Drop result
                            except Exception as e:
                                print(f"Detector: Queue error: {e}")

                    if self._display_interval != 0:
                        if self._image_counter % self._display_interval == 0:
                            annotated = results.plot()
                            cv2.imshow(f'Stream {addr[0]}', annotated)
                            cv2.waitKey(1)

                    if self._save_images:
                        os.makedirs("stream_out/output", exist_ok=True)
                        cv2.imwrite(f"stream_out/output/img_{self._image_counter:06d}.png", annotated)

            """
            elif fmt == 1:  # JPEG. Leftover code
                nparr = np.frombuffer(stream['buffer'], np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    cv2.imshow(f'Stream {addr[0]}', decoded)
                    cv2.waitKey(1)
            """
            time.sleep(0.0003)
        except Exception as e:
            print(f"[{addr}] Processing error: {e}")
            self._dropped_images += 1
            time.sleep(0.0003)
