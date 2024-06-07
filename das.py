import cv2
from datetime import datetime
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from gpiozero import LED, Buzzer
from time import sleep
from drivers import Lcd
from gtts import gTTS
import pygame
import atexit  # Import the atexit module
import os
import texting
 
pygame.mixer.init()

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh_eye = 0.25
thresh_mouth = 0.5
frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/home/group3/Desktop/das/models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)

led = LED(21)
buzzer = Buzzer(16)
lcd = Lcd()

flag_eye = 0
flag_mouth = 0

# Create a folder to store drowsy images if it doesn't exist
if not os.path.exists('drowsyImages'):
    os.makedirs('drowsyImages')
    
# Function to stop pygame mixer on exit
@atexit.register
def exit_handler():
    pygame.mixer.music.stop()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Mouth detection
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mar > thresh_mouth:
            flag_mouth += 1
            print("Mouth Flag:", flag_mouth)
            if flag_mouth >= frame_check:
                buzzer.on()
                led.on()
                lcd.lcd_display_string("===ALERT!===", 1)
                lcd.lcd_display_string("Yawning detected.", 2)
                lcd.lcd_display_string("Stay alert!", 3)
                texting.text_messasing()


                # Generate image name with current date and time
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"drowsy_{current_time}.jpg"
                
                # Save the image to drowsyImages folder
                cv2.imwrite(os.path.join('drowsyImages', image_name), frame)
                print(f"Drowsy image {image_name} saved")

                try:
                    tts = gTTS("Alert! Yawning detected. Stay alert!")
                    tts.save("alert.mp3")
                    pygame.mixer.music.load("alert.mp3")
                    pygame.mixer.music.play()
                    sleep(3)
                except Exception as e:
                    print(f"Error in TTS: {e}")

                buzzer.off()
                led.off()
                lcd.lcd_clear()
        else:
            flag_mouth = 0
        
        # Eye detection
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh_eye:
            flag_eye += 1
            print("Eye Flag:", flag_eye)
            if flag_eye >= frame_check:
                buzzer.on()
                led.on()
                lcd.lcd_display_string("===ALERT!===", 1)
                lcd.lcd_display_string("You seem drowsy.", 2)
                lcd.lcd_display_string("Wake up!", 3)
                texting.text_messasing()


                # Generate image name with current date and time
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"/home/group3/Desktop/das/drowsyImages/drowsy_{current_time}.jpg"
                
                # Save the image to drowsyImages folder
                cv2.imwrite(os.path.join('drowsyImages', image_name), frame)
                print(f"Drowsy image {image_name} saved")

                try:
                    tts = gTTS("Alert! You seem drowsy. Wake up!")
                    tts.save("alert.mp3")
                    pygame.mixer.music.load("alert.mp3")
                    pygame.mixer.music.play()
                    sleep(3)
                except Exception as e:
                    print(f"Error in TTS: {e}")

                buzzer.off()
                led.off()
                lcd.lcd_clear()
        else:
            flag_eye = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
lcd.lcd_clear()
