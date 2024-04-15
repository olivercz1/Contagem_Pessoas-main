import numpy as np
import cv2
# https://docs.opencv.org


def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

detects = []

posL = 350  # centralização
offset = 60  # espaço entre as linhas

xy1 = (posL, 20)  # espaço entre o topo do frame e a linha
xy2 = (posL, 450)  # Tamanho da Linha vertical de cima para baixo

total = 0
up = 0
down = 0

while 1:
    ret, frame = cap.read()

    # transforma o frame para preto e branco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # marcara para identificar apenas o preto e branco
    fgmask = fgbg.apply(gray)

    # limpa os nozes sugeiras da imagem
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # para retirar o nozes da imagem
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    # auementa o branco do frame
    dilation = cv2.dilate(opening, kernel, iterations=8)

    closing = cv2.morphologyEx(
        # retira os espaços pretos entre os brancos
        dilation, cv2.MORPH_CLOSE, kernel, iterations=8)

    cv2.imshow("closing", closing)

    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)  # Linha vertical

    cv2.line(frame, (posL-offset, xy1[1]),
             (posL-offset, xy2[1]), (255, 255, 0), 2)
    cv2.line(frame, (posL+offset, xy1[1]),
             (posL+offset, xy2[1]), (255, 255, 0), 2)

    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)

        if int(area) > 3000:
            centro = center(x, y, w, h)

            cv2.putText(frame, str(i), (x+5, y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if len(detects) <= i:
                detects.append([])
            # Verificando a posição vertical
            if centro[0] > posL-offset and centro[0] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1

    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()

    else:

        for detect in detects:
            for (c, l) in enumerate(detect):
                total = down - up

                if down > up:
                 total = 0
                continue

                if detect[c-1][0] < posL and l[0] > posL:  # Verificando a direção vertical
                    detect.clear()
                    down += 1
                    # total += 1
                    cv2.line(frame, xy1, xy2, (0, 255, 0), 5)  # Linha vertical
                    continue

                if detect[c-1][0] > posL and l[0] < posL:  # Verificando a direção vertical
                    detect.clear()
                    up += 1
                    # total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)  # Linha vertical
                    continue

                if c > 0:
                    cv2.line(frame, detect[c-1], l, (0, 0, 255), 1)

    cv2.putText(frame, "TOTAL: "+str(total), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, "ENTRARAM: "+str(up), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "SAIRAM: "+str(down), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
