import requests
import numpy as np
import cv2

import os

class REST:
    def __init__(self):
        self.__url = '<SERVER_IP>/api/rpi'
        self.__filename = './WrapperAPI/rest/test.png'
        print("--> REST API has been set")

    def send(self, types, size, image):
        size = str(size)
        if image is None:
            multiple_files = [('type', (None, types)), ('size', (None, size))]
            r = requests.post(self.__url, files=multiple_files)
            return r.status_code
        else:
            cv2.imwrite(self.__filename, image) 
            multiple_files = [('type', (None, types)), ('size', (None, size)), ('file', ('image.png', open(self.__filename, 'rb'), 'image/png'))]
            r = requests.post(self.__url, files=multiple_files)
            os.remove(self.__filename)
            return r.status_code