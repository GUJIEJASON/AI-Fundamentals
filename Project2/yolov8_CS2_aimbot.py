import sys
import ctypes
import signal
import argparse
import win32con
import win32api
import pydirectinput
import bettercam
import torch
import cv2
import random
import pyautogui
import os
import time
import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
from pynput import mouse
from ultralytics import YOLO
from matplotlib import pyplot as plt
from pynput import mouse, keyboard
from pynput.keyboard import Key

PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

def get_game_window_bounding_box(game_title="Counter-Strike 2"):
    # 查找游戏窗口
    game_window = gw.getWindowsWithTitle(game_title)
    
    if game_window:
        # 获取并构造窗口的边界框字典
        window = game_window[0]
        bounding_box = {
            'left': window.left,
            'top': window.top,
            'width': window.width,
            'height': window.height
        }
        
        print(f"{game_title}窗口的边界框信息：{bounding_box}")
        return bounding_box
    else:
        print(f"未找到名为'{game_title}'的窗口。")
        return None
def capScreen(camera, bound_box):
    # print(f"Bound Box Type: {type(bound_box)}")
    print(f"Bound Box: {bound_box}")

    left = int(bound_box['left'])
    top = int(bound_box['top'])
    right = left + int(bound_box['width'])
    bottom = top + int(bound_box['height'])

    bound_box = (left, top, right, bottom)
    #sct_img = camera.grab(bound_box)

    camera.start(bound_box)
    sct_img = camera.get_latest_frame()
    camera.stop()

    if sct_img is None:
        print("Failed to grab image")
        return None

    # print(f"sct_img Type: {type(sct_img)}")
    # print(f"sct_img Shape: {sct_img.shape}")
    # print(f"sct_img Data Type: {sct_img.dtype}")

    if isinstance(sct_img, np.ndarray) and sct_img.shape[2] == 3:
        return sct_img
    else:
        raise ValueError("sct_img is not a valid image with 3 channels (BGR)")
    #print(f"sct_img Type: {type(sct_img)}")
    #sct_img = np.array(sct_img)         # HWC, and C==4
    #sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)     # 4 channels -> 3 channels
    #return sct_img

def capSave(img_array, img_name, save_path):
    cv2.imwrite(os.path.join(save_path, img_name), img_array)

def pre_process(img0, img_sz):
    """
    img0: from capScreen(), format: HWC, BGR
    """
    # padding resize
    img = cv2.resize(img0, (img_sz, img_sz))  # 修改
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()  # uint8 to fp32
    img /= 255.0  # 0-255 to 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def calculate_position(xyxy):
    """
    计算目标中心坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    center_x = int((c2[0] + c1[0]) / 2)
    center_y = int((c2[1] + c1[1]) / 2)
    return center_x, center_y

def calculate_position_body(xyxy):
    """
    计算身体坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    center_x = int((c2[0] + c1[0]) / 2)
    center_y = int(c1[1] - ((c1[1] - c2[1]) / 5))
    return center_x, center_y

def view_imgs(img0, boxes, confs, classes, model_names, scale_x, scale_y):
    """
    弹窗展示结果，按 'q' 键退出
    """
    display_img = cv2.resize(img0, (640, 400))  # 缩放图像

    # 绘制缩放后的框和标签
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        conf = confs[i]
        cls = classes[i]

        # 缩放坐标
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f'{model_names[cls]} {conf:.2f}'

        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('ws demo', display_img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

def move_mouse(mouse_pynput, aim_persons_center, aim_heads_center):
    """
    移动鼠标
    """
    if aim_persons_center:
        current_x, current_y = mouse_pynput.position
        # current_x, current_y = pyautogui.position()
        best_position = None
        best_area = None

        for aim_person in aim_persons_center:
            dist = ((aim_person[0] - current_x) ** 2 + (aim_person[1] - current_y) ** 2) ** .5
            if not best_position or dist < best_position[1]:
                best_position = (aim_person, dist)
                best_area = (aim_person[2], aim_person[3], aim_person[4], aim_person[5])

        for aim_head in aim_heads_center:
            if aim_head[0] > best_area[0] and aim_head[1] < best_area[1] and aim_head[0] < best_area[2] and aim_head[1] > best_area[3]:
                best_position = (aim_head, 0)
                break

        # pydirectinput.moveTo(int(best_position[0][0]), int(best_position[0][1]), duration=0.5)
        # pyautogui.moveTo(int(best_position[0][0]), int(best_position[0][1]))
        # pydirectinput.moveRel(int(best_position[0][0])-current_x, int(best_position[0][1])-current_y,relative=True)
        # win32api.SetCursorPos((int(best_position[0][0]), int(best_position[0][1])))
        # mouse_control.move(int(best_position[0][0])-current_x, int(best_position[0][1])-current_y)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(best_position[0][0])-current_x, int(best_position[0][1])-current_y, 0, 0)
        # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(best_position[0][0]), int(best_position[0][1]), 0, 0)

class AimYolo:
    def __init__(self, opt):
        self.weights = opt.weights
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.view_img = opt.view_img
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.augment = opt.augment

        # self.bounding_box = {'left': 720, 'top': 0, 'width': 1200, 'height': 1080}
        # self.bounding_box = {'left': 720, 'top': 0, 'width': 1840, 'height': 1440}
        self.bounding_box = get_game_window_bounding_box()

        self.model = YOLO(self.weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
         # 初始化鼠标控制和键盘监听
        self.mouse_control = mouse.Controller()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.mouse_move_enabled = True

        # 初始化阵营选择
        self.team_selected = False
        self.team = None
        self.team_listener = keyboard.Listener(on_press=self.on_team_select)
        self.team_listener.start()


     # 键盘按键监听回调函数（阵营选择）
    def on_team_select(self, key):
        try:
            if key == Key.f5:
                self.team = 'T'
                self.team_selected = True
                print("选择 T 阵营")
            elif key == Key.f6:
                self.team = 'CT'
                self.team_selected = True
                print("选择 CT 阵营")
            elif key == Key.f7:
                self.team = 'ALL'
                self.team_selected = True
                print("没有阵营")
        except AttributeError:
            pass

     # 键盘按键监听回调函数
    def on_press(self, key):
        try:
            if key == Key.f8:
                self.mouse_move_enabled = not self.mouse_move_enabled
                print(f"Mouse movement {'enabled' if self.mouse_move_enabled else 'disabled'}")
        except AttributeError:
            pass

    def run(self):
        img_sz = self.img_size

        camera = bettercam.create(output_color="BGR")
        # mouse_control = mouse.Controller()

        print("请选择你的阵营，F5键代表T阵营，F6键代表CT阵营，F7键代表没有阵营")
        # 等待阵营选择
        while not self.team_selected:
            time.sleep(0.1)

        # 计算缩放比例
        scale_w1 = self.bounding_box['width'] / img_sz
        scale_h1 = self.bounding_box['height'] / img_sz

        scale_w = 640 / img_sz
        scale_h = 400 / img_sz

        while True:
            img0 = capScreen(camera, self.bounding_box)  # HWC and BGR
            # original_img_shape = img0.shape[:2]
            img = pre_process(img0, img_sz)

            results = self.model(img, augment=self.augment, conf=self.conf_thres, iou=self.iou_thres)
            

            # print(results[0])
            # Check if results contain detections
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 提取边界框
            confs = results[0].boxes.conf.cpu().numpy()  # 提取置信度
            classes = results[0].boxes.cls.cpu().numpy().astype(int)  # 提取类别索引并转换为整数类型

            # print(results[0].boxes)
                
            if self.mouse_move_enabled:
                aim_persons_center = []
                aim_heads_center = []
                # aim_persons_xy = []
                for box in enumerate(boxes):
                    xyxy = box[1]
                    # print(classes[box[0]])
                    if (self.team == 'T'):
                        if classes[box[0]] == 2:
                            center_x, center_y = calculate_position(xyxy)
                            aim_heads_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"]])

                        elif classes[box[0]] == 1:
                            # aim_persons_xy.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                            center_x, center_y = calculate_position_body(xyxy)
                            aim_persons_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"],xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

                    elif (self.team == 'CT'):
                        if classes[box[0]] == 5:
                            center_x, center_y = calculate_position(xyxy)
                            aim_heads_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"]])

                        elif classes[box[0]] == 4:
                            # aim_persons_xy.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                            center_x, center_y = calculate_position_body(xyxy)
                            aim_persons_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"], xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

                    else:
                        if classes[box[0]] == 5 or classes[box[0]] == 2:
                            center_x, center_y = calculate_position(xyxy)
                            aim_heads_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"]])

                        elif classes[box[0]] == 4 or classes[box[0]] == 1:
                            # aim_persons_xy.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                            center_x, center_y = calculate_position_body(xyxy)
                            aim_persons_center.append([center_x * scale_w1 + self.bounding_box["left"], center_y * scale_h1 + self.bounding_box["top"], xyxy[0], xyxy[1], xyxy[2], xyxy[3]])


                # 调用鼠标移动函数
                # if len(aim_heads_center) > 0:
                #     move_mouse(self.mouse_control, aim_heads_center)
                # else:
                #     move_mouse(self.mouse_control, aim_persons_center)
                move_mouse(self.mouse_control, aim_persons_center, aim_heads_center)

            

            if self.view_img:
                view_imgs(img0, boxes, confs, classes, self.model.names, scale_w, scale_h)

            

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/v8s_180_epoch.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    return parser.parse_args()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    opt = parseArgs()
    aim_yolo = AimYolo(opt)
    aim_yolo.run()