import cv2


def getCenter(cnt):
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def sortContoursByXY(cnts):
    sorted_cnts = []
    for cnt in cnts:
        cX, cY = getCenter(cnt)
        sorted_cnts.append({'x': cX, 'y': cY, 'cnt': cnt})
    sorted_cnts.sort(key=lambda c: c['y'])
    sorted_cnts.sort(key=lambda c: c['x'])

    return [cnt['cnt'] for cnt in sorted_cnts]
