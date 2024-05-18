import numpy as np
import cv2
import face_recognition
from moviepy.editor import VideoFileClip, AudioFileClip
from flask import current_app, url_for


def input_image(img_path):
    img = face_recognition.load_image_file(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def blur_face(image, face):
    (startY, endX, endY, startX) = face
    face_img = image[startY:endY, startX:endX]
    blurred_face = cv2.GaussianBlur(face_img, (99, 99), 30)
    image[startY:endY, startX:endX] = blurred_face
    return image


def webcam_face_blur(exclusion_img, save_video_path, audio_path, output_video_path, socketio, app):
    cap = cv2.VideoCapture(save_video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    encode_exclusion_img = face_recognition.face_encodings(exclusion_img)[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for enc, loc in zip(face_encodings, face_locations):
            match = face_recognition.compare_faces([encode_exclusion_img], enc, tolerance=0.4)
            if not match[0]:
                frame = blur_face(frame, loc)

        out.write(frame)
        frame_num += 1
        progress = int((frame_num / total_frames) * 100)
        socketio.emit('progress', {'progress': progress})

    cap.release()
    out.release()
    socketio.emit('progress', {'progress': 100})  # 작업 완료 신호 보내기

    combine_audio_video(output_video_path, audio_path, output_video_path.replace('.mp4', '_with_audio.mp4'), fps,
                        socketio, app)


def combine_audio_video(video_path, audio_path, output_path, fps, socketio, app):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)
        print("Video and audio have been successfully combined.")
    except Exception as e:
        print("Failed to combine video and audio:", e)
    finally:
        video_clip.close()
        audio_clip.close()

    with app.app_context():
        socketio.emit('complete', {'url': url_for('camera_convert')})
