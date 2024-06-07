**Drowsy Diving Detection System aka Driving Assistance System (DAS)**

The Drowsy Driving Detectiion System is a real-time drowsiness and yawning detection solution designed to enhance driver safety by monitoring and alerting them of signs of drowsiness or yawning. Utilizing computer vision techniques and GPIO components, this system provides immediate alerts through buzzer sounds, LED lights, an LCD display, and text-to-speech notifications.

**Features**

 Real-time Eye and Mouth Monitoring: Uses a webcam to continuously monitor the driver's eyes and mouth for signs of drowsiness and yawning.
 
 Eye Aspect Ratio (EAR) Detection: Calculates the Eye Aspect Ratio to determine if the eyes are closed.
 
Mouth Aspect Ratio (MAR) Detection: Calculates the Mouth Aspect Ratio to detect yawning.

Immediate Alerts: Activates a buzzer, LED, LCD display, and audio alerts to wake the driver if drowsiness or yawning is detected.

Text-to-Speech Notifications: Uses Google Text-to-Speech (gTTS) to play an audible alert.

Image Storage: Takes a picture of the driver when drowsiness or yawning is detected and saves it to a file. The picture includes the date and time it was taken. 

**Components**

Software: OpenCV, Dlib, imutils, scipy, GPIOZero, gTTS, pygame

Hardware: Webcam, LED, Buzzer, LCD Display (I2C)
