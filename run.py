################### Installation Requirements ###########################
# CUDA-11.8이 설치되어 있는 GPU 사용 가능한 컴퓨터기 때문에 gpu 버전 다운로드 필수!
# !pip install paddlepaddle-gpu
# %pip install paddlepaddle-gpu
#
# PaddleOCR whl package 설치
# !pip install "paddleocr>=2.0.1"
# %pip install paddleocr
#
# OpenCV 설치
# !pip install opencv-python
# %pip install opencv-python
########################################################################

import paddle
import argparse
import logging
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from typing import Any
import os
import shutil
import re

# export CUDA_VISIBLE_DEVICES='0'

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if PaddleOCR is compiled with CUDA
logger.debug(f"Is compiled with CUDA? {paddle.is_compiled_with_cuda()}")
logger.debug(f"CUDA devices:, {paddle.device.cuda.device_count()}")

# Set arguments
parser: argparse.ArgumentParser = argparse.ArgumentParser(
    prog="PaddleOCR_Korean_Character_Crop",
    description="PaddleOCR Korean character crop program",
)
parser.add_argument("-i") # input image path (Default: ./inputs/image.png)
parser.add_argument("-o") # output image path (Default: ./final/)

# Initialize the output directory
logger.debug("Initializing output directory...")
current_path = os.getcwd()
shutil.rmtree(current_path + "/cropped", ignore_errors=True)
shutil.rmtree(current_path + "/projection", ignore_errors=True)
shutil.rmtree(current_path + "/best_size", ignore_errors=True)
shutil.rmtree(current_path + "/final", ignore_errors=True)
os.makedirs(current_path + "/cropped", exist_ok=True)
os.makedirs(current_path + "/projection", exist_ok=True)
os.makedirs(current_path + "/best_size", exist_ok=True)
os.makedirs(current_path + "/final", exist_ok=True)

# Load the PaddleOCR Korean model
ocr: PaddleOCR = PaddleOCR(lang="korean")

def crop(bounding_boxes: np.ndarray) -> list:
    """
    PaddleOCR로부터 얻은 bounding box를 바탕으로 이미지를 자르는 함수
    bounding box는 4개의 꼭짓점 좌표로 이루어져 있으며,
    이 좌표를 바탕으로 이미지를 자른다.
    """
    cropped_images: list = []
    offset = 5
    for idx, box in enumerate(bounding_boxes):
        # 그레이스케일로 변환
        
        # box 점 정렬
        rect = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1) # 네 좌표의 합
        diff = np.diff(box, axis=1) # 네 좌표의 차이
        rect[0] = box[np.argmin(s)] # 좌상단 좌표
        rect[2] = box[np.argmax(s)] # 우하단 좌표
        rect[1] = box[np.argmin(diff)] # 우상단 좌표
        rect[3] = box[np.argmax(diff)] # 좌하단 좌표

        # rect = [ 좌상, 우상, 우하, 좌하 ]

        # 네 꼭짓점 좌표를 numpy array로 변환
        # 이때, 검출 영역이 너무 Fit하게 되어있어서 +offset 을 해준다
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
        # 최대 너비 및 높이를 선택해야 잘리는 부분을 최소화할 수 있다
        widthA = np.sqrt((pts[2][0] - pts[3][0])**2 + (pts[2][1] - pts[3][1])**2) # 좌하단 - 우하단
        widthB = np.sqrt((pts[1][0] - pts[0][0])**2 + (pts[1][1] - pts[0][1])**2) # 좌상단 - 우상단
        maxWidth = max(int(widthA), int(widthB)) # 두 위아래 변 중에 최대 너비 선택해서 crop 영역으로 선택

        heightA = np.sqrt((pts[1][0] - pts[2][0])**2 + (pts[1][1] - pts[2][1])**2) # 좌상단 - 우상단
        heightB = np.sqrt((pts[0][0] - pts[3][0])**2 + (pts[0][1] - pts[3][1])**2) # 좌하단 - 우하단
        maxHeight = max(int(heightA), int(heightB)) # 두 좌우 변 중에 최대 높이 선택해서 crop 영역으로 선택

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

        cropped_images.append(warped)
        cv2.imwrite(f"./cropped/croped_{idx}.png", warped)

    logger.debug(f"# of Cropped Images: {len(cropped_images)}")
    return cropped_images

def projection(cropped_images: list) -> list:
    """
    이미지의 각 열마다 픽셀 합계를 구하여 글자 사이의 경계를 판단하고,
    각 글자를 분리하여 이미지를 저장하는 핵심 함수
    """
    result_images: list = []
    offset = 5
    minimum_width = 5
    space_threshold = 2500 # 글자 사이의 경계를 판단하는 임계값
    
    for idx, image in sorted(enumerate(cropped_images)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변환
        bin_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)[1] # 흰색 글자로 변환
        
        vertical_sum = np.sum(bin_img, axis=0)  # 각 열의 픽셀 합계
        
        columns = bin_img.shape[1] # 열의 개수
        
        start_col = 0
        for col in range(columns): # 각 열마다 임계값 비교
            if vertical_sum[col] <= space_threshold:
                # [start_col ~ col-1] 구간을 하나의 글자로 추정
                if col - start_col > minimum_width:  # 폭이 너무 좁지 않도록 필터링
                    char_img = bin_img[:, max(0, start_col - offset):min(col + offset, columns)]
                    result_images.append(char_img)
                    cv2.imwrite(f"./projection/projection_{idx}_{start_col}_{col}.png", cv2.bitwise_not(char_img))
                start_col = col + 1
    
    logger.debug(f"# of Projection Images: {len(result_images)}")
    return result_images

def best_size(projection_images: list) -> list:
    """
    글자 이미지 자체가 너무 작은 경우, 좋은 input이 될 수 없기 때문에
    크기가 적당한 이미지만을 추출하는 함수
    이때, 크기가 적당하다는 기준은 height와 width가 모두 20px 이상인 경우로 잡았음
    """
    best_size_images = []
    size_threshold = 20 # 최소 크기
    for idx, image in enumerate(projection_images):
        height, width = image.shape[:2]

        if height < size_threshold or width < size_threshold:
            continue
        
        # gaussian blur
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        cv2.imwrite(f"./best_size/character_{idx}.png", cv2.bitwise_not(image))
        best_size_images.append(cv2.bitwise_not(image))

    logger.debug(f"# of Best Size Images: {len(best_size_images)}")
    return best_size_images

def get_characters(best_size_images: list, output_path: str) -> None:
    """
    최종적으로 OCR을 통해 한글 글자 1문자를 인식하고,
    인식된 글자와 함께 이미지를 저장하는 함수
    이때, 인식된 글자가 한글이 아닌 경우는 제외함
    """
    result_images = []
    for idx, image in enumerate(best_size_images):
        result = ocr.ocr(image, det=False, rec=True, cls=False)

        if not result or not result[0] or not result[0][0]: # 인식 못하면 continue
            continue

        text = result[0][0][0]
        confidence = result[0][0][1]

        if re.search(r"[^가-힣]", text): # 한글이 아닌 문자가 있는 경우 continue
            continue

        if text and len(text) == 1 and confidence > 0.9:
            print(f"#### {idx}th // Character: {text}, Confidence: {confidence}")
            result_images.append((image, text))


    for image, text in result_images:
        cv2.imwrite(output_path + f"{text}.png", image)
    

if __name__ == "__main__":
    args = parser.parse_args()
    input_image_path: str = args.i if args.i else "./inputs/image.png"
    output_image_path: str = args.o if args.o else "./final/"
    
    # Read the image
    logger.debug(f"Input image path: {input_image_path}")
    image: cv2.typing.MatLike = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error(f"Failed to read image: {input_image_path}")
        exit(1)
    
    # Perform OCR
    logger.debug("Performing OCR...")
    ocr_result = ocr.ocr(image, cls=True)
    logger.debug("OCR completed.")
    
    # Draw the OCR results on the image
    logger.debug("Drawing OCR results...")
    boxes: list = [line[0] for line in ocr_result[0]]
    scores: list = [line[1][1] for line in ocr_result[0]]
    texts: list = [line[1][0] for line in ocr_result[0]]
    ocr_result_image: np.ndarray = draw_ocr(image, boxes, texts, scores, font_path="MaruBuri-Regular.ttf")
    cv2.imwrite("paddle_ocr_result.png", ocr_result_image)
    logger.debug("Drawing completed.")
    
    # Get bounding boxes
    logger.debug("Getting bounding boxes...")
    np_boxes: np.ndarray = np.array([])
    if boxes and len(boxes) > 0:
        np_boxes = np.array([np.array(box) for box in boxes])
    logger.debug(f"# of Boxes: {len(boxes)}")
    
    #####################################################################
    #####################################################################
    # 1. Cropping with bounding boxes
    logger.debug("Cropping with bounding boxes...")
    cropped_images: list = crop(np_boxes)
    
    # 2. Projection
    logger.debug("Projection...")
    projection_images: list = projection(cropped_images)
    
    # 3. Get best size images
    logger.debug("Getting best size images...")
    best_size_images: list = best_size(projection_images)
    
    # 4. Get characters
    logger.debug("Getting characters...")
    get_characters(best_size_images, output_image_path)
    
    logger.info("Completed!")
    logger.info("#" * 50)
    #####################################################################
    #####################################################################
    