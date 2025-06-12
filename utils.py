import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_box(img, box, label):
    # img = cv2.imread(img)
    color = (0, 255, 0)
    thickness = 5
    font_scale=0.8
    font_path = "fonts/gulim.ttc"
    font = ImageFont.truetype(font_path, 30)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((box[0]+3, box[1]+3), label, font=font, fill='navy', stroke_width=2, stroke_fill='navy')
    draw.rectangle((box[0],box[1],box[2],box[3]), outline=color, width=thickness)
    img = np.array(img_pil)
    # cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), color, thickness)
    # cv2.putText(img, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), 1)
    
    return img