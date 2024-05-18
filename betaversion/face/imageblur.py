import numpy as np
import cv2
import face_recognition
import os

def input_image(img_path, target_width=640):
    img = face_recognition.load_image_file(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[1], img.shape[0]  # 원래 크기 (width, height)
    img = resize_image(img, target_width)
    return img, original_size

def resize_image(image, target_width):
    height, width = image.shape[:2]
    if width > target_width:
        scaling_factor = target_width / width
        new_width = target_width
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

def process_image(imgTrain, imgToProcess, output_path, original_size):
    # 학습할 이미지에서 얼굴 인코딩 및 위치 추출
    encodeimg = face_recognition.face_encodings(imgTrain)[0]

    # 처리할 이미지에서 얼굴 위치 및 인코딩 추출
    rgb_frame = cv2.cvtColor(imgToProcess, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print("Total faces found in the image to process:", len(face_locations))

    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = face_recognition.compare_faces([encodeimg], face_encoding, tolerance=0.3)
        print("Match found:", match)
        if not match[0]:
            blur_face(imgToProcess, face_location)

    # 원래 크기로 되돌리기
    imgToProcess = cv2.resize(imgToProcess, original_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, imgToProcess)
    print(f"Processed image saved to {output_path}")

def blur_face(image, face):
    (top, right, bottom, left) = face
    face_img = image[top:bottom, left:right]
    blurred_face = cv2.GaussianBlur(face_img, (99, 99), 30)
    image[top:bottom, left:right] = blurred_face

    return image
