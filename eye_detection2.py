# import cv2
# import csv
# import os

# eye_cascPath = 'harcascade\haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
# face_cascPath = 'harcascade\haarcascade_frontalface_alt.xml'  #face detect model

# def detect_eyes(video_path, output_csv):
#     faceCascade = cv2.CascadeClassifier(face_cascPath)
#     eyeCascade = cv2.CascadeClassifier(eye_cascPath)

#     cap = cv2.VideoCapture(video_path)
#     filename = os.path.basename(video_path)  # Extract only the file name
#     output_video_path = os.path.join('static/videos', os.path.splitext(filename)[0] + "_detections.mp4")  # Use only the file name for output
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))
#     with open(output_csv, 'w', newline='') as csvfile:
#         fieldnames = ['Frame', 'Face Detected', 'Eye Detected']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         frame_count = 0
#         while cap.isOpened():
#             ret, img = cap.read()
#             if ret:
#                 frame_count += 1
#                 frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 faces = faceCascade.detectMultiScale(
#                     frame,
#                     scaleFactor=1.1,
#                     minNeighbors=5,
#                     minSize=(30, 30)
#                 )
#                 face_detected = len(faces) > 0

#                 if face_detected:
#                     for (x, y, w, h) in faces:
#                         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
#                     frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
#                     eyes = eyeCascade.detectMultiScale(
#                         frame,
#                         scaleFactor=1.1,
#                         minNeighbors=5,
#                         minSize=(30, 30)
#                     )
#                     eye_detected = len(eyes) > 0
#                     if eye_detected:
#                         print('eyes!!!')
#                     else:
#                         print('no eyes!!!')
#                 else:
#                     eye_detected = False
#                     print('no faces!!!')

#                 writer.writerow({'Frame': frame_count, 'Face Detected': face_detected, 'Eye Detected': eye_detected})
#                 frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
#                 cv2.imshow('Face Recognition', frame_tmp)
#                 out.write(img)
#                 waitkey = cv2.waitKey(1)
#                 if waitkey == ord('q') or waitkey == ord('Q') or frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#                     cv2.destroyAllWindows()
#                     break

#     cap.release()

import cv2
import csv
import os

eye_cascPath = 'harcascade\haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'harcascade\haarcascade_frontalface_alt.xml'  #face detect model

def detect_eyes(video_path, output_csv):
    faceCascade = cv2.CascadeClassifier(face_cascPath)
    eyeCascade = cv2.CascadeClassifier(eye_cascPath)

    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path)  # Extract only the filename without the path
    filename_no_extension = os.path.splitext(filename)[0]  # Remove the file extension
    output_video_path = os.path.join('static/videos', filename_no_extension+"_detections.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Frame', 'Face Detected', 'Eye Detected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                frame_count += 1
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                face_detected = len(faces) > 0

                if face_detected:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                    frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                    eyes = eyeCascade.detectMultiScale(
                        frame,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    eye_detected = len(eyes) > 0
                    if eye_detected:
                        print('eyes!!!')
                    else:
                        print('no eyes!!!')
                else:
                    eye_detected = False
                    print('no faces!!!')

                writer.writerow({'Frame': frame_count, 'Face Detected': face_detected, 'Eye Detected': eye_detected})
                frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Face Recognition', frame_tmp)
                out.write(img)
                waitkey = cv2.waitKey(1)
                if waitkey == ord('q') or waitkey == ord('Q') or frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    cv2.destroyAllWindows()
                    break

    cap.release()
