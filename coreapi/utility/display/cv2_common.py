
"""display.py
"""

import os
from typing import List
import math
import json
import cv2
import numpy as np
from PIL import ImageDraw, Image
from arshot.hwcontrol.camera import Camera

def open_window(window_name, title, width=None, height=None, posx=None, posy=None):
    """Open the display window."""
    cv2.namedWindow(window_name)
    cv2.setWindowTitle(window_name, title)
    if width and height:
        cv2.resizeWindow(window_name, width, height)
    if posx and posy:
        cv2.moveWindow(window_name, posx, posy)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img

def show_prog_time(screen, prog_time, pos, font, lang='en'):
    """Draw processing time at the specified position on the screen."""
    pos_x, pos_y = pos
    #screen[pos_y:, pos_x:] = 255
    if lang == 'jp':
        prog_time_text = f'処理時間: {prog_time:.3f} 秒'
    else:
        prog_time_text = f'Prog Time: {prog_time:.3f} seconds'

    screen = Image.fromarray(screen)
    draw = ImageDraw.Draw(screen)
    draw.text((pos_x, pos_y), prog_time_text, (0, 0, 0), font)
    screen = np.array(screen)
    return screen

def set_display(window_name, full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)

def get_coord(json_path):
    if not os.path.exists(json_path):
        ui_width = None
        ui_height = None
        windows = []
        windows_names = []
        return []

    with open(json_path, 'r', encoding="utf-8") as f:
        ui_info = json.load(f)

    windows = []
    windows_names = []
    for shape in ui_info['objects']:
        x = int(shape['points'][0][0])
        y = int(shape['points'][0][1])
        w = int(shape['points'][1][0])
        h = int(shape['points'][1][1])
        windows.append((x, y, w, h))
        windows_names.append(shape['window'])
    ui_width = ui_info['imageWidth']
    ui_height = ui_info['imageHeight']
    ui_offset_x = ui_info['offsetX']
    ui_offset_y = ui_info['offsetY']
    # windows = windows
    # windows_names = windows_names
    return ui_width, ui_height, ui_offset_x, ui_offset_y, windows_names, windows

class MultipleCameraImage():
    def __init__(self, cam_sources: List[Camera], window_size=(1200, 800)):
        self.cam_sources = cam_sources
        # self.window_size = (1200, 800)
        self.window_size = window_size
        # self.window_name = 'window'
        self.num_cam = len(self.cam_sources)

    def read_cam_and_combine_image(self):
        # window_name = self.window_name
        frames = []
        for cap in self.cam_sources:
            frame = cap.read()
            if frame is None:
                return None
            frames.append(frame)
        window_frame = self.combine_image(frames)
        return window_frame

    def combine_image(self, images, grid_size=None, show_name=True, resize='fill', box_draw=None):
        window_size = self.window_size
        num_images = len(images)
        if grid_size is None:
            grid_row, grid_col = create_grid(num_images)
        else:
            grid_row, grid_col = grid_size
        padding_ratio = 0.05
        padding_width = int(window_size[0] * padding_ratio / (grid_col + 1))
        padding_height = int(window_size[1] * padding_ratio / (grid_row + 1))
        padding_height = padding_width = min(padding_width, padding_height)
        frame_width = (window_size[0] - padding_width * (grid_col + 1)) // grid_col
        frame_height = (window_size[1] - padding_height * (grid_row + 1)) // grid_row
        # padding_width = (window_size[0] - frame_width * grid_col) // 2
        # padding_height = (window_size[1] - frame_height * grid_row) // 2
        window_frame = np.zeros((window_size[1], window_size[0], 3), np.uint8)
        for i, image in enumerate(images):
            index_row = i // grid_col
            index_col = i - grid_col * index_row
            frame_x = padding_width * (index_col + 1) + index_col * frame_width
            frame_y = padding_height * (index_row + 1) + index_row * frame_height
            if resize == 'fill':
                small_image = cv2.resize(image, (frame_width, frame_height))
            else:
                small_image = resize_pad_center(image, frame_width, frame_height, fill=(0, 0, 0))
            if show_name:
                cv2.putText(small_image, f'Device: {i}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if box_draw == i:
                cv2.rectangle(window_frame, (frame_x, frame_y), (frame_x+frame_width, frame_y+frame_height), (0, 0, 255), 10)

            window_frame[frame_y:frame_y+frame_height, frame_x:frame_x+frame_width] = small_image
        if show_name: #### Draw device none text in the empty frame
            for i in range(len(images), grid_row * grid_col):
                index_row = i // grid_col
                index_col = i - grid_col * index_row
                pos_x = padding_width * (index_col + 1) + index_col * frame_width
                pos_y = padding_height * (index_row + 1) + index_row * frame_height + 30
                cv2.putText(window_frame, 'Device: None', (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.rectangle(window_frame, (0, 0), (window_size[0], window_size[1]), (180, 180, 180), 2)
        cv2.line(window_frame, (window_size[0]//2, 0), (window_size[0]//2, window_size[1]), (180, 180, 180), thickness=2)
        cv2.line(window_frame, (0, window_size[1]//2), (window_size[0], window_size[1]//2), (180, 180, 180), thickness=2)

        return window_frame

def resize_pad_center(img, width, height, fill=(0, 0, 0)):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_h, img_w = img.shape[:2]
    ratio_w = width / img_w
    ratio_h = height / img_h
    resize_img = np.zeros((height, width, 3), np.uint8)
    resize_img[:,:] = fill
    ratio = min(ratio_w, ratio_h)
    resize_w = int(ratio * img_w)
    resize_h = int(ratio * img_h)
    small_image = cv2.resize(img, (resize_w, resize_h))
    padding_w = (width - resize_w) // 2
    padding_h = (height - resize_h) // 2
    resize_img[padding_h:padding_h+resize_h, padding_w:padding_w+resize_w] = small_image
    return resize_img

def create_grid(num: int):
    sqrt = math.sqrt(num)
    if sqrt.is_integer():
        col = row = int(sqrt)
    else:
        col = math.ceil(sqrt)
        row = col - 1
        if col * row < num:
            row = row + 1
    # print('Row: {}, Col: {}'.format(row, col))
    return row, col
