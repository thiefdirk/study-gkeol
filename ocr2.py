from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt
#mutilang

ocr = PaddleOCR(use_angle_cls=True, lang="korean")


path =  'D:/sample_resident.jpg'

result = ocr.ocr(path, cls=True)

for line in result:
    print(line)
    
    
image = Image.open(path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/gulim.ttc')
im_show = Image.fromarray(im_show)
plt.imshow(im_show)
im_show.save('result.jpg')

