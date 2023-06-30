import PySimpleGUI as sg
import cv2
import os


layout = [
    [sg.Image(key='-IMAGE-')],
    [sg.Text('People in picture', key='-TEXT-', expand_x=True, justification='c')]
]

window = sg.Window('Face Detector', layout)

# get video
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\\Users\\ISHIKA JAGATRAMKA\\Desktop\\face scan\\venv\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml')


while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    ret, frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=7,
        minSize=(50, 50))

    # draw the rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # update the image
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

    # update the text
    window['-TEXT-'].update(f'People in picture: {len(faces)}')

window.close()

