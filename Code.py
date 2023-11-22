import cv2
import time


def detect_abandoned_item(video_file):
    video = cv2.VideoCapture(video_file)
    ret, frame = video.read()

    coord = {}
    color = (0, 0, 255)
    txt = 'ATTENTION! UNKNOWN OBJECT'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output3.MP4', fourcc, 20.0, (1280, 960))  # запись видео в отдельный файл в формате mp4

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        time.sleep(0.03)  # регулировка скорости видео

        ret, frame = video.read()
        gray_frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # преобразование в серый цвет
        frame_diff = cv2.absdiff(gray_frame, gray_frame_next)  # разница между кадрами
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)  # применение порога разницы

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # удаление шумов
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # нахождение контуров

        for contour in contours:
            area = cv2.contourArea(contour)  # S контура
            x, y, w, h = cv2.boundingRect(contour)

            if area > 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # отрисовка контуров
                s = sum([x, y, x + w, y + h])
                if s in coord:
                    coord[s] += 1
                else:
                    coord[s] = 1
                if coord[s] > 100:  # если сумма координат объекта остается неизменной на протяжении более 100 кадров
                    cv2.putText(frame, "abandoned item!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                2)  # аларм на контуре
                    cv2.putText(frame, txt, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, color,
                                2)  # вставка аларма в левый верхний угол
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Abandoned Items Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
