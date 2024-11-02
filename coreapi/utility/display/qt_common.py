import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
#from PyQt5.QtCore import *

#pylint: disable=invalid-name
def rgbImageToPixmap(frame, size=(480, 480, 3)) -> QPixmap:
    if frame is None:
        image = np.ones(size, np.uint8) * 255
    else:
        image = frame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = image.shape
    bytes_per_line = ch * w
    p = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(p)
