import cv2

def highlight(img, ratio=.85):
    h, w = img.shape[:2]
    start_w = int(w * (1 - ratio) / 2)
    start_h = int(h * (1 - ratio) / 2)
    high_w = int(w * ratio)
    high_h = int(h * ratio)
    high_img = img[start_h:start_h + high_h, start_w:start_w+high_w].copy()
    low_img = cv2.convertScaleAbs(img, alpha=1, beta=-40)
    low_img[start_h:start_h + high_h, start_w:start_w+high_w] = high_img
    ### Draw crosshair
    c_length1 = 10
    c_length2 = 100
    color = (255, 255, 255)
    low_img[start_h:start_h+c_length1, start_w:start_w+c_length2] = color
    low_img[start_h:start_h+c_length2, start_w:start_w+c_length1] = color
    low_img[start_h:start_h+c_length1, start_w+high_w-c_length2:start_w+high_w] = color
    low_img[start_h:start_h+c_length2, start_w+high_w-c_length1:start_w+high_w] = color
    low_img[start_h+high_h-c_length1:start_h+high_h, start_w+high_w-c_length2:start_w+high_w] = color
    low_img[start_h+high_h-c_length2:start_h+high_h, start_w+high_w-c_length1:start_w+high_w] = color
    low_img[start_h+high_h-c_length1:start_h+high_h, start_w:start_w+c_length2] = color
    low_img[start_h+high_h-c_length2:start_h+high_h, start_w:start_w+c_length1] = color
    cv2.rectangle(low_img, (start_w, start_h), (start_w+high_w, start_h+high_h), color, 2)
    return low_img
