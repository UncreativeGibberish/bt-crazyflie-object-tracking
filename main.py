from detector import Detector
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
import time
import logging
from logging import handlers
import struct
import warnings
import time
from threading import Event
import multiprocessing as mp
from queue import Empty, Full
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.commander import Commander
from cflib.utils import uri_helper
from typing import Optional, Tuple, Dict, Any
from cflib.utils.reset_estimator import reset_estimator
import queue
IMG_WIDTH = 324
IMG_HEIGHT = 244


logger = logging.getLogger(__name__)
target_z = 0.3
current_z = 0
land = False
    

class DroneController:
    def __init__(self, link_uri):
        self._cf = Crazyflie(rw_cache='./cache')
        # Callbacks
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        self._start_bat = 0.0
        self._current_bat = 0.0
        self.target_z = 0.1
        print('Attempting connection to %s' % link_uri)
    
        self._cf.open_link(link_uri)
        self.is_connected = True


    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)


        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=300)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.roll', 'float')
        self._lg_stab.add_variable('stabilizer.pitch', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self._lg_stab.add_variable('pm.vbat', 'FP16')
        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')
    
    
    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        global current_z
        current_z = data['stateEstimate.z']
        self._current_bat = data['pm.vbat']
        if self._start_bat == 0:
            self._start_bat = self._current_bat
        
        #print(f'[{timestamp}][{logconf.name}]: ', end='')
        #for name, value in data.items():
        #    print(f'{name}: {value:3.3f} ', end='')
        #print()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

    def cf(self):
        return self._cf
    
    def commander(self):
        return self._cf.commander
    
    def arm(self):
        self._cf.platform.send_arming_request(True)
    
    def non_blocking_hover_setpoint(self, vx, vy, yawrate, zdistance, timeout):
        try:
            pk = CRTPPacket()
            pk.port = CRTPPort.COMMANDER_GENERIC
            pk.channel = 0
            pk.data = struct.pack('<Bffff', 10, vx, vy, yawrate, zdistance)
            self._cf.link.out_queue.put(pk, block=True, timeout=timeout)
        except queue.Full:
            pass
        except Exception as e:
            print(f"Radio send failed: {e}")

def activate_mellinger_controller(cf, use_mellinger):
    controller = 1
    if use_mellinger:
        controller = 2
    cf.param.set_value('stabilizer.controller', controller)

def get_center_correction(
    bbox: tuple[int, int, int, int],
    img_width: int = 324,
    img_height: int = 244,
    central_size: int = 100,        # Central region size
    tolerance: int = 30             # Tolerance (essentially makes the central region 130x130)
) -> Tuple[bool, float, float, float]: 
    # Returns a tuple [is_centered, horiz_error, vert_error, bbox_area]

    if not bbox or len(bbox) != 4:
        return False, 0.0, 0.0, 0.0
    
    bbox_center_x, bbox_center_y, bbox_w, bbox_h = bbox


    img_center_x = img_width // 2
    img_center_y = img_height // 2

    
    half = central_size // 2
    cx1 = img_center_x - half
    cx2 = img_center_x + half
    cy1 = img_center_y - half
    cy2 = img_center_y + half
    # Check if bb is within the central region
    is_centered = (
        cx1 - tolerance <= bbox_center_x <= cx2 + tolerance and
        cy1 - tolerance <= bbox_center_y <= cy2 + tolerance
    )

    horiz_error = (bbox_center_x - img_center_x) / (img_width / 2)
    vert_error  = (bbox_center_y - img_center_y) / (img_height / 2)
    print(f"Debug: horiz_error is {horiz_error}. vert_error is {vert_error}")

    return is_centered, horiz_error, vert_error, bbox_w*bbox_h
    

def console_callback(text: str):
    '''A callback to run when we get console text from the Crazyflie'''
    # The Crazyflie provides newline
    print(text, end='')


if __name__ == "__main__":

    recv_conn, send_conn = mp.Pipe()
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue(maxsize=8)
    log_queue = mp.Queue(-1)

    file_handler = logging.FileHandler(
        'Detector_log.txt',
        mode='a',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(created).3f | %(message)s'))

    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    detector = Detector(
        model_path="detection_models/yolo26n epoch 150, img320, batch 0.70/weights/best.pt", # rt-detr weights won't work here unless you instantiate a RTDETR instance in detector._run. Could add model_type param
        confidence_threshold=0.2,
        cuda=True,
        save_images=False,
        result_queue = result_queue,
        log_queue = log_queue,
        pipe_conn = send_conn,
        display_interval=10
    )
    time.sleep(3)

    print("Going to start detector")
    detector.start()

    # Wait for ready signal from detector
    # Would probably be better to use poll instead of recv. Eliminate the need for sleep
    while recv_conn.recv() != "READY":
        time.sleep(0.05)
    
    cflib.crtp.init_drivers()

    URI = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E701')
    drone = DroneController(URI)
    
    cf = drone.cf()
    #For writing the entire console output to a file
    cf.console.receivedChar.add_callback(console_callback)

    # Reset kalman filter and wait for an accurate positional estimate for the drone
    reset_estimator(cf)

    last_result = None
    last_detection_time = time.time()
    last_setpoint_time = time.time()
    SETPOINT_INTERVAL = 0.2
    MAIN_LOOP_INTERVAL = 0.2
    try:
        drone.arm()
        time.sleep(0.5)
        try:
            result = result_queue.get_nowait()
        except Exception:
            result = None
        last_loop_time = time.time()

        while True:
            # Todo: Measure execution time and sleep for the time remaining AFTER executing the loop body
            if time.time() - last_loop_time < MAIN_LOOP_INTERVAL:
                time.sleep(0)   # Doesn't help the detector process. Mainly hoping the scheduler will give time to cflib threads
                continue
            last_loop_time = time.time()
            try:
                result = result_queue.get(timeout=0.2)
            except queue.Empty:
                result = None
            
            if result is not None and result.detected:
                last_detection_time = time.time()

                if last_result is None or result.id != last_result.id:
                    last_result = result
                    is_centered, horiz_error, vert_error, bbox_area = get_center_correction(bbox = last_result.bbox)
                    
                    print(f"Debug - result id: {last_result.id}, result bb: {last_result.bbox}, timestamp: {last_result.timestamp}")
                    print(f"Debug - bbox area: {bbox_area}")
                    if not is_centered:
                        
                        #target_z = current_z   # Unused in this implementation. Intended for use with vertical error for vertical adjustments
                        print("Making tracking adjustments.")
                        if time.time() - last_setpoint_time >= SETPOINT_INTERVAL:
                            try:
                                vx = 0
                                if bbox_area <= 1000:
                                    vx = 0.1
                                drone.non_blocking_hover_setpoint(vx, 0, -60*horiz_error, 0.1, 0.1)
                            except Exception as e:
                                print(f"Error sending tracking hover setpoint: {e}")
                                last_setpoint_time = time.time()
                                pass
                            last_setpoint_time = time.time()
                            continue                 
            
            if time.time() - last_setpoint_time >= SETPOINT_INTERVAL:
                # Hover in place
                try:
                    drone.non_blocking_hover_setpoint(0, 0, 0, 0.1, 0.1)
                except Exception as e:
                    print(f"Error sending default hover setpoint: {e}")
                    last_setpoint_time = time.time()
                    pass
                last_setpoint_time = time.time()
                time.sleep(0)
                     
    except KeyboardInterrupt:
        print(f"Starting battery was: {drone._start_bat}. Ended with {drone._current_bat}") # pm.vbat is a bit unreliable. Interpret with scepticism.
        

        with MotionCommander(drone.cf()) as mc:
            mc.land(0.05)

        try:
            cf.close_link()
        except Exception as e:
            print(f"Error closing Crazyflie link: {e}")

        try:
            detector.stop()
        except Exception as e:
            print(f"Error stopping detector: {e}")

        try:
            listener.stop()
            result_queue.close()
            result_queue.join_thread()
            log_queue.close()
            log_queue.join_thread()
        except Exception:
            pass

        
        




