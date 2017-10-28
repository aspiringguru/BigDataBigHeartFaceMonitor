# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound #https://pypi.python.org/pypi/playsound/1.2.1
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx  #pip install pyttsx

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
'''
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-a2", "--alarm2", type=str, default="",
	help="path alarm2 .WAV file")
'''
args = vars(ap.parse_args())

print ("args: ", args)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48


# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False
oldeyePos = 150
#counter2 and ALARM_ON2 added for detecting face movement
COUNTER2 = 0
ALARM_ON2 = False
FACE_CONSEC_FRAMES = 48

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) 			= face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) 			= face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) 	= face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(noseStart, noseEnd) 	= face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jawStart, jawEnd) 		= face_utils.FACIAL_LANDMARKS_IDXS["jaw"]


print ("lStart:", lStart, ", lEnd:", lEnd)
print ("lStart:", rStart, ", lEnd:", rEnd)

#initialize text to speech engine.
engine = pyttsx.init()

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		#print ("type(leftEye):", type(leftEye), leftEye.shape)
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		#https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		mouth = shape[mouthStart:mouthEnd]
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of frames
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True
					#
					engine.say('Hey. Eyes closed for too long.')
					engine.runAndWait()

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					'''
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
					'''
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
				print ("DROWSINESS ALERT! playing sound")
				#0, 0, 255 = apears on screen as red, color charts show blue as 0,0,0255
				#255-0-0 = red on color charts, blue on screen.

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# add if-else condition here to test if eyes are moving or stationary.
		#change this to use CONST style variable for easier config
		#print ("leftEye[1]:", type(leftEye[1]), leftEye[1])
		leftEyePos = int(leftEye[1][0])
		if (1.0*leftEyePos/oldeyePos < 1.1) and (1.0*leftEyePos/oldeyePos > 0.9):
			#tod: fix comparator to check x&y co-ords
			COUNTER2 += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER2 >= FACE_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON2:
					ALARM_ON2 = True
					#
					engine.say('Hey. Head not moving.')
					engine.runAndWait()


					# check to see if an alarm2 file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					'''
					if args["alarm2"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm2"],))
						t.deamon = True
						t.start()
					'''
				# draw an alarm on the frame
				cv2.putText(frame, "face not moving!", (200, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
				print ("face not moving! playing sound")
				#0, 0, 255 = apears on screen as red, color charts show blue as 0,0,0255
				#255-0-0 = red on color charts, blue on screen.
				#255, 255, 0 = yellow on color charts. python shows 0, 255, 255 as yellow

		# otherwise, the face has moved enough
		# threshold, so reset the counter and alarm
		else:
			COUNTER2 = 0
			ALARM_ON2 = False
			oldeyePos = leftEye[1][0]


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
