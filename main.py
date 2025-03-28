import re
import os
import shutil
import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import paddle

# In[47]:


# CUDA-11.8이 설치되어 있는 GPU 사용 가능한 컴퓨터기 때문에 gpu 버전 다운로드
# !pip install paddlepaddle-gpu
# %pip install paddlepaddle-gpu

# PaddleOCR whl package 설치
# !pip install "paddleocr>=2.0.1"
# %pip install paddleocr

# # 한번 사용해보도록 하자
# 
# 실행하기 전, 컴퓨터에 cuDNN이 설치되어있는지 확인하자. 없다면 설치
# 
# CUDA 11.8 사용 기준으로
# 
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cudnn
# sudo apt-get -y install cudnn-cuda-11

print(f"Is compiled with CUDA? {paddle.is_compiled_with_cuda()}")
print(f"CUDA devices:, {paddle.device.cuda.device_count()}")

ocr = PaddleOCR(lang="korean")

image_path = 'test_imgs/best00.png'
result = ocr.ocr(image_path, cls=True)

# draw result
from PIL import Image
result = result[0]
image = Image.open(image_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='MaruBuri-Regular.ttf')
im_show = Image.fromarray(im_show)
im_show.save('first_ocr_result.png')
im_show.show()

# 한글 또는 영어, 숫자 txt만 포함된 box로 재구성
boxes = [boxes[idx] for idx in range(len(boxes)) if txts[idx].isalnum()]

for idx in range(len(boxes)):
    print(f"boxes: {boxes[idx]}")

# 1. cv2 imread로 그레이스케일 이미지 획득
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. 연산 효율성을 올리기 위해 boxes를 numpy array로 변환
boxes_np_arr = np.array([])
if boxes and len(boxes) > 0:
    boxes_np_arr = np.array([np.array(box) for box in boxes])

    for box in boxes:
        print(box)
else:
    print("No boxes")

# 3. 각 box = bounding box
# bounding box 영역 안에 있는 각 영역을 Crop해서 새로운 이미지로 추출한다

current_path = os.getcwd()
shutil.rmtree(current_path + "/cropped", ignore_errors=True)
os.makedirs(current_path + "/cropped", exist_ok=True)
croped_images = []

for idx, box in enumerate(boxes_np_arr):
    # box 점 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)  # 네 좌표의 합
    diff = np.diff(box, axis=1)  # 네 좌표의 차이
    rect[0] = box[np.argmin(s)]  # 좌상단 좌표
    rect[2] = box[np.argmax(s)]  # 우하단 좌표
    rect[1] = box[np.argmin(diff)]  # 우상단 좌표
    rect[3] = box[np.argmax(diff)]  # 좌하단 좌표

    # rect = [ 좌상, 우상, 우하, 좌하 ]

    # 네 꼭짓점 좌표를 numpy array로 변환
    # 이때, 검출 영역이 너무 Fit하게 되어있어서 +10을 해준다
    offset = 10
    image_height, image_width = image.shape[:2]
    image_height -= 1
    image_width -= 1

    # rect[0] = 좌상단
    x1 = max(0, rect[0][0] - offset)
    y1 = max(0, rect[0][1] - offset)
    # rect[1] = 우상단
    x2 = min(rect[1][0] + offset, image_width)
    y2 = max(0, rect[1][1] - offset)
    # rect[2] = 우하단
    x3 = min(rect[2][0] + offset, image_width)
    y3 = min(rect[2][1] + offset, image_height)
    # rect[3] = 좌하단
    x4 = max(0, rect[3][0] - offset)
    y4 = min(rect[3][1] + offset, image_height)
    pts = np.array(
        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ],
        dtype="float32"
    )

    # 네 좌표를 바탕으로 이미지의 너비 및 높이 계산
    widthA = np.sqrt((pts[2][0] - pts[3][0])**2 + (pts[2][1] - pts[3][1])**2)
    widthB = np.sqrt((pts[1][0] - pts[0][0])**2 + (pts[1][1] - pts[0][1])**2)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((pts[1][0] - pts[2][0])**2 + (pts[1][1] - pts[2][1])**2)
    heightB = np.sqrt((pts[0][0] - pts[3][0])**2 + (pts[0][1] - pts[3][1])**2)
    maxHeight = max(int(heightA), int(heightB))

    # 출력될 이미지 좌표 설정
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ],
        dtype="float32"
    )

    # Perspective Transformation
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 음영 제거 -> 모폴로지 연산
    kernel_size = 150
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    background = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)
    gray_minus_bg = cv2.subtract(warped, background)
    normed = cv2.normalize(gray_minus_bg, None, 0, 255, cv2.NORM_MINMAX)

    croped_images.append(normed)
    cv2.imwrite(f"./cropped/croped_{idx}.png", normed)

print(f"# of Cropped Images: {len(croped_images)}")

shutil.rmtree(current_path + "/contours", ignore_errors=True)
os.makedirs(current_path + "/contours", exist_ok=True)
for idx, image in enumerate(croped_images):
    
    # findContours는 글자가 흰색이어야 테두리를 인식할 수 있음
    contours, _ = cv2.findContours(
        cv2.bitwise_not(image),
        cv2.RETR_EXTERNAL, # 글자 외부 외곽선이 필요하고, ㅇ과 같은 경우, 내부 외곽선은 필요 없으므로 RETR_EXTERNAL 사용
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 외곽선 기반으로 각 글자 이미지 추출
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        print(f"Contour: {x}, {y}, {w}, {h}")
        
        # 면적이나 가로세로 비율을 엄격하게 조정
        area = cv2.contourArea(contour)
        if aspect_ratio > 1.2 or aspect_ratio < 0.5 or area < 10:
            continue

        character = image[y:y+h, x:x+w]
        
        # 상하좌우 3px 흰색배경 추가
        padding = 3
        expanded = cv2.copyMakeBorder(
            character, 
            padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )

        cv2.imwrite(f"./contours/contour_{idx}_{x}_{y}_{w}_{h}.png", expanded)

# 테두리 추출한 이미지 크기가 20px 이상인 이미지만 선택
shutil.rmtree(current_path + "/best_size", ignore_errors=True)
os.makedirs(current_path + "/best_size", exist_ok=True)
contours_dir = current_path + "/contours"
for idx, filename in enumerate(os.listdir(contours_dir)):
    character = cv2.imread(f"{contours_dir}/{filename}", cv2.IMREAD_GRAYSCALE)
    height, width = character.shape[:2]

    if height < 20 or width < 20:
        continue
    
    cv2.imwrite(f"./best_size/character_{idx}.png", character)


shutil.rmtree(current_path + "/final", ignore_errors=True)
os.makedirs(current_path + "/final", exist_ok=True)
save_dir = current_path + "/final/"
best_size_dir = current_path + "/best_size/"
result_cnt = 0
result_images = []
for idx, filename in enumerate(sorted(os.listdir(best_size_dir))):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(f"{best_size_dir}" + filename, cv2.IMREAD_GRAYSCALE)
        result = ocr.ocr(img, det=False, rec=True, cls=False)

        if not result or not result[0] or not result[0][0]: # 인식 못하면 continue
            continue

        text = result[0][0][0]
        confidence = result[0][0][1]

        if re.search(r"[^가-힣]", text): # 한글이 아닌 문자가 있는 경우 continue
            continue

        if text and len(text) == 1 and confidence > 0.9:
            result_cnt += 1
            print(f"#### {idx}th // Character: {text}, Confidence: {confidence}")
            result_images.append((img, text))

if result_cnt >= 5:
    for image, text in result_images:
        cv2.imwrite(save_dir + f"{text}.png", image)
else:
    print("Not enough characters ...")
