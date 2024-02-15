from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from gpiozero import LED, Buzzer
from time import sleep
from drivers import Lcd
from gtts import gTTS
import pygame
import atexit  # Import the atexit module

pygame.mixer.init()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/home/group3/Desktop/das/models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)

# Initialize GPIO components
led = LED(21)
buzzer = Buzzer(16)
lcd = Lcd()

flag = 0

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
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Adjust the threshold for drowsiness detection
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                # Activate buzzer, LED, speaker, and update LCD display
                buzzer.on()
                led.on()
                lcd.lcd_display_string("===ALERT!===", 1)
                lcd.lcd_display_string("You seem drowsy.", 2)
                lcd.lcd_display_string("Wake up!", 3)

                # Text to Speech
                try:
                    tts = gTTS("Alert! You seem drowsy. Wake up!")
                    tts.save("alert.mp3")
                    pygame.mixer.music.load("alert.mp3")
                    pygame.mixer.music.play()

                    sleep(3)  # Adjust the time as needed
                except Exception as e:
                    print(f"Error in TTS: {e}")

                buzzer.off()
                led.off()
                lcd.lcd_clear()
        else:
            flag = 0
            buzzer.off()
            led.off()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
lcd.lcd_clear()
