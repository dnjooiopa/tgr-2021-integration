import requests
import numpy as np
import cv2
# from PIL import Image as im
# import io
import os



# img = cv2.imread('./WrapperAPI/rest/136j_800.png')
# print(type(img))

# test = open('./WrapperAPI/rest/136j_800.png', 'rb')
# print(type(test))

# x = im.fromarray(img)
# print(type(x))

# img_byte_arr = io.BytesIO()
# x.save(img_byte_arr, format='PNG')
# img_byte_arr = img_byte_arr.getvalue()

# fd, path = tempfile.mkstemp()
# try:
#     with os.fdopen(fd, 'w') as tmp:
#         # do stuff with temp file
#         tmp.write('stuff')
# finally:
#     os.remove(path)

# x = np.random.randn(1000)

# with TemporaryFile() as t:
#     x.tofile(t)

# r = requests.get('http://165.22.251.78/api/items')

# multiple_files = [
#         ('type', "help")
# ...     ('images', ('image.png', img_byte_arr, 'image/png'))]

# r = requests.post('http://165.22.251.78/api/items', multiple_files)

# print(r.text)

class REST:
    def __init__(self):
        self.__url = 'http://165.22.251.78/api/items'
        self.__filename = './WrapperAPI/rest/test.png'

    def send(self, types, image):
        # temp = im.fromarray(image)
        cv2.imwrite(self.__filename, image) 
        # img_byte_arr = io.BytesIO()
        # temp.save(img_byte_arr, format='PNG')
        # print(type(img_byte_arr))
        # img_byte_arr = img_byte_arr.getvalue()
        # print(type(img_byte_arr))  ('type', types),

        multiple_files = [('type', types),('images', ('image.png', open(self.__filename, 'rb'), 'image/png'))]

        r = requests.post(self.__url, files=multiple_files)
        os.remove(self.__filename)



def main():
    img = cv2.imread('./WrapperAPI/rest/136j_800.png')
    # test = open('./WrapperAPI/rest/136j_800.png', 'rb')
    # print(type(test))  
    rest = REST()
    rest.send('lime_small', img)

if __name__ == "__main__":
    main()
