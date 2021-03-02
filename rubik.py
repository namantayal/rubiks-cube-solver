import cv2 as cv
import numpy as np
import kociemba as cubeSolver

# Global Variables
count = 0
previous_face_colors = []
point_color = (0, 0, 255)
face_registered = 0
cube = []
pos_map = {'R': '', 'G': '', 'B': '', 'W': '', 'R': '', 'O': '', 'Y': ''}
registered_flag = False
cube_solving = False
solve_step = 0
solution = ''
sol = []

cap = cv.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# function to arrange points in order


def arrangePoints(points):
    arrX = [0, 0, 0, 0]
    arrY = [0, 0, 0, 0]
    x = points[0:len(points):2]
    y = points[1:len(points):2]
    x, y = zip(*sorted(zip(x, y)))
    if(y[0] < y[1]):
        index1 = 0
        index2 = 1
    else:
        index1 = 1
        index2 = 0
    arrX[0] = x[index1]
    arrY[0] = y[index1]
    arrX[1] = x[index2]
    arrY[1] = y[index2]

    if(y[2] > y[3]):
        index1 = 2
        index2 = 3
    else:
        index1 = 3
        index2 = 2

    arrX[2] = x[index1]
    arrY[2] = y[index1]
    arrX[3] = x[index2]
    arrY[3] = y[index2]

    return arrX, arrY

# function to devide line into three equal parts


def three(pt1, pt2):
    x1 = (2*(pt1[0]) + pt2[0])//3
    x2 = (2*(pt2[0]) + pt1[0])//3
    y1 = (2*(pt1[1]) + pt2[1])//3
    y2 = (2*(pt2[1]) + pt1[1])//3
    return (x1, y1), (x2, y2)

# function to calculate mid point of line


def mid(pt1, pt2):
    x1 = (pt1[0] + pt2[0])//2
    y1 = (pt1[1] + pt2[1])//2
    return (x1, y1)

# function to calculate error between the set value and recorded value


def error(val1, val2):
    return ((val1-val2)**2).sum()

# function to guess the color of one cubelet


def guessColor(color):
    colors = np.array([[135, 115, 250],
                       [50, 215, 135],
                       [215, 190, 90],
                       [40, 220, 219],
                       [210, 210, 210],
                       [90, 145, 255]])  # r g b y w o
    colorMap = {0: "Red", 1: "Green", 2: "Blue",
                3: "Yellow", 4: "White", 5: "Orange"}
    errors = np.zeros(6)
    for i in range(6):
        errors[i] = error(np.array(color), colors[i])
    index = np.where(errors == np.min(errors))
    return colorMap[index[0][0]]

# function to detect the colors on the face of cube


def getColors(img, boxMids):
    face_colors = []
    for boxMid in boxMids:
        color = img[boxMid[1], boxMid[0]]
        guess = guessColor(color)
        face_colors.append(guess[0])
    return face_colors

# function to scan all the cube faces


def scan_cube_faces(img, boxMids):
    face_colors = getColors(img, boxMids)
    # cv.putText(
    #     img, guess[0], (boxMid[0]-5, boxMid[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    global previous_face_colors
    global count
    global point_color, face_registered, registered_flag, cube, pos_map

    if face_registered == 0:
        cv.putText(img, "Scan Cube Faces", (200, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif 0 < face_registered < 3 or 3 < face_registered:
        cv.putText(img, "Turn the Cube", (200, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if registered_flag:
            cv.arrowedLine(img, boxMids[0], boxMids[2], (0, 0, 255), 2)
            cv.arrowedLine(img, boxMids[3], boxMids[5], (0, 0, 255), 2)
            cv.arrowedLine(img, boxMids[6], boxMids[8], (0, 0, 255), 2)
    elif face_registered == 3:
        cv.putText(img, "Turn the Cube", (200, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if registered_flag:
            cv.arrowedLine(img, boxMids[0], boxMids[6], (0, 0, 255), 2)
            cv.arrowedLine(img, boxMids[1], boxMids[7], (0, 0, 255), 2)
            cv.arrowedLine(img, boxMids[2], boxMids[8], (0, 0, 255), 2)

    count_to_pos = {0: 'F', 1: 'L', 2: 'B', 3: 'U', 4: 'R', 5: 'D'}
    if(face_colors == previous_face_colors):
        count += 1
    else:
        previous_face_colors = face_colors
        point_color = (0, 0, 255)
        count = 0
        registered_flag = False

    if count > 20 and registered_flag == False:
        point_color = (0, 255, 0)
        pos_map[face_colors[4]] = count_to_pos[face_registered]
        face_registered += 1
        cube.extend(face_colors)
        registered_flag = True


# Function to convert cube colors into respective postion


def color_to_pos():
    global cube
    cube_pos = cube
    f = cube_pos[0:9]
    l = cube_pos[9:18]
    b = cube_pos[18:27]
    u = cube_pos[27:36]
    r = cube_pos[36:45]
    d = cube_pos[45:54]
    u = u[::-1]
    r = r[2:9:3] + r[1:9:3] + r[0:9:3]
    cube_pos = u+r+f+d+l+b
    cube = cube_pos
    cube_pos = ''
    for i in range(len(cube)):
        cube_pos += pos_map[cube[i]]
    return cube_pos

# Function to plot arrow corresponding to move


def drawArrows(img, move, boxMids):
    if(move == "F" or move == "B"):
        cv.arrowedLine(img, boxMids[0], boxMids[2], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[2], boxMids[8], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[8], boxMids[6], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[6], boxMids[0], (0, 0, 255), 2)
    elif(move == 'R'):
        cv.arrowedLine(img, boxMids[8], boxMids[2], (0, 0, 255), 2)
    elif(move == 'U'):
        cv.arrowedLine(img, boxMids[2], boxMids[0], (0, 0, 255), 2)
    elif(move == 'L'):
        cv.arrowedLine(img, boxMids[0], boxMids[6], (0, 0, 255), 2)
    # elif(move == 'B'):
    #     cv.arrowedLine(img, boxMids[2], boxMids[0], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[8], boxMids[2], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[6], boxMids[8], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[0], boxMids[6], (0, 0, 255), 2)
    elif(move == 'D'):
        cv.arrowedLine(img, boxMids[6], boxMids[8], (0, 0, 255), 2)
    elif(move == "F'" or move == "B'"):
        cv.arrowedLine(img, boxMids[2], boxMids[0], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[8], boxMids[2], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[6], boxMids[8], (0, 0, 255), 2)
        cv.arrowedLine(img, boxMids[0], boxMids[6], (0, 0, 255), 2)
    elif(move == "R'"):
        cv.arrowedLine(img, boxMids[2], boxMids[8], (0, 0, 255), 2)
    elif(move == "U'"):
        cv.arrowedLine(img, boxMids[0], boxMids[2], (0, 0, 255), 2)
    elif(move == "L'"):
        cv.arrowedLine(img, boxMids[6], boxMids[0], (0, 0, 255), 2)
    # elif(move == "B'"):
    #     cv.arrowedLine(img, boxMids[0], boxMids[2], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[2], boxMids[8], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[8], boxMids[6], (0, 0, 255), 2)
    #     cv.arrowedLine(img, boxMids[6], boxMids[0], (0, 0, 255), 2)
    elif(move == "D'"):
        cv.arrowedLine(img, boxMids[8], boxMids[6], (0, 0, 255), 2)


# function to manipulate the cube list according to the move


def performMove(move):
    global cube
    u = cube[0:9]
    r = cube[9:18]
    f = cube[18:27]
    d = cube[27:36]
    l = cube[36:45]
    b = cube[45:54]
    print("Move Performed - ", move)
    print("Before Move - ", move)
    print("u - ", u)
    print("r - ", r)
    print("f - ", f)
    print("d - ", d)
    print("l - ", l)
    print("b - ", b)
    if move == 'F':
        f = f[6::-3] + f[7::-3] + f[8::-3]
        temp = u[6:9]
        u[6:9] = l[8::-3]
        l[2:9:3] = d[0:3]
        d[2::-1] = r[0:9:3]
        r[0:9:3] = temp
    elif move == 'R':
        r = r[6::-3] + r[7::-3] + r[8::-3]
        temp = u[2:9:3]
        u[2:9:3] = f[2:9:3]
        f[2:9:3] = d[2:9:3]
        d[2:9:3] = b[6::-3]
        b[6::-3] = temp
    elif move == 'U':
        u = u[6::-3] + u[7::-3] + u[8::-3]
        temp = f[0:3]
        f[0:3] = r[0:3]
        r[0:3] = b[0:3]
        b[0:3] = l[0:3]
        l[0:3] = temp
    elif move == 'L':
        l = l[6::-3] + l[7::-3] + l[8::-3]
        temp = u[0:9:3]
        u[0:9:3] = b[8::-3]
        b[8::-3] = d[0:9:3]
        d[0:9:3] = f[0:9:3]
        f[0:9:3] = temp
    elif move == 'D':
        d = d[6::-3] + d[7::-3] + d[8::-3]
        temp = f[6:9]
        f[6:9] = l[6:9]
        l[6:9] = b[6:9]
        b[6:9] = r[6:9]
        r[6:9] = temp
    elif move == 'B':
        b = b[6::-3] + b[7::-3] + b[8::-3]
        temp = u[0:3]
        u[0:3] = r[2:9:3]
        r[2:9:3] = d[8:5:-1]
        d[6:9] = l[0:9:3]
        l[6::-3] = temp
    elif move == "F'":
        f = f[2::3] + f[1::3] + f[0::3]
        temp = u[6:9]
        u[6:9] = r[8::-3]
        r[2:9:3] = d[0:3]
        d[2::-1] = l[0:9:3]
        r[0:9:3] = temp
    elif move == "R'":
        r = r[2::3] + r[1::3] + r[0::3]
        temp = u[2:9:3]
        u[2:9:3] = b[6::-3]
        b[6::-3] = d[2:9:3]
        d[2:9:3] = f[2:9:3]
        f[2:9:3] = temp
    elif move == "U'":
        u = u[2::3] + u[1::3] + u[0::3]
        temp = f[0:3]
        f[0:3] = l[0:3]
        l[0:3] = b[0:3]
        b[0:3] = r[0:3]
        r[0:3] = temp
    elif move == "L'":
        l = l[2::3] + l[1::3] + l[0::3]
        temp = u[0:9:3]
        u[0:9:3] = f[0:9:3]
        f[0:9:3] = d[0:9:3]
        d[0:9:3] = b[8::-3]
        b[8::-3] = temp
    elif move == "D'":
        d = d[2::3] + d[1::3] + d[0::3]
        temp = f[6:9]
        f[6:9] = r[6:9]
        r[6:9] = b[6:9]
        b[6:9] = l[6:9]
        l[6:9] = temp
    elif move == "B'":
        b = b[2::3] + b[1::3] + b[0::3]
        temp = u[0:3]
        u[2::-1] = l[0:9:3]
        l[0:9:3] = d[6:9]
        d[6:9] = r[8::-3]
        r[2:9:3] = temp
    print("After Move - ", move)
    print("u - ", u)
    print("r - ", r)
    print("f - ", f)
    print("d - ", d)
    print("l - ", l)
    print("b - ", b)
    cube = u+r+f+d+l+b


face_match = False

# Function to display moves sequentially on the cube


def solve_cube(img, solution, boxMids):
    # print("cube - ", cube)
    # print("cube Front - ", cube[18:27])
    global solve_step, previous_face_colors, count, face_match, point_color
    if(solve_step < len(solution)):
        move = solution[solve_step]
        face_colors = getColors(img, boxMids)
        # print("Face - ", face_colors)
        if(move[0] == 'B'):
            if(face_colors == cube[45:54]):
                count += 1
            else:
                count = 0
                if face_match == False:
                    cv.putText(img, "Show Back Face", (200, 350),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    drawArrows(img, solution[solve_step], boxMids)
        else:
            if(face_colors == cube[18:27]):
                count += 1
            else:
                count = 0
                if face_match == False:
                    cv.putText(img, "Show Front Face", (200, 350),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    drawArrows(img, solution[solve_step], boxMids)

        if count > 15 and face_match == False:
            face_match = True
            performMove(move)
            count = 0

        if face_match and count > 15:
            face_match = False
            count = 0
            solve_step += 1


# Function to detect the area enclosing the cube

def plotCube(img, imgContour):
    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 8000:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.03*peri, True)
            rect = cv.minAreaRect(approx)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            # cv.drawContours(imgContour, [box], 0, (0, 255, 0), 2)
            coord = box.ravel()
            x, y = arrangePoints(coord)
            if len(box) == 4:
                A = (x[0], y[0])
                D = (x[3], y[3])
                M = (x[1], y[1])
                P = (x[2], y[2])
                B, C = three(A, D)
                N, O = three(M, P)
                E, I = three(A, M)
                H, L = three(D, P)
                F, J = three(B, N)
                G, K = three(C, O)
                global point_color
                cv.circle(imgContour, A, 3, point_color, -1)
                cv.circle(imgContour, B, 3, point_color, -1)
                cv.circle(imgContour, C, 3, point_color, -1)
                cv.circle(imgContour, D, 3, point_color, -1)
                cv.circle(imgContour, E, 3, point_color, -1)
                cv.circle(imgContour, F, 3, point_color, -1)
                cv.circle(imgContour, G, 3, point_color, -1)
                cv.circle(imgContour, H, 3, point_color, -1)
                cv.circle(imgContour, I, 3, point_color, -1)
                cv.circle(imgContour, J, 3, point_color, -1)
                cv.circle(imgContour, K, 3, point_color, -1)
                cv.circle(imgContour, L, 3, point_color, -1)
                cv.circle(imgContour, M, 3, point_color, -1)
                cv.circle(imgContour, N, 3, point_color, -1)
                cv.circle(imgContour, O, 3, point_color, -1)
                cv.circle(imgContour, P, 3, point_color, -1)
                boxMids = [mid(A, F),
                           mid(B, G),
                           mid(C, H),
                           mid(E, J),
                           mid(F, K),
                           mid(G, L),
                           mid(I, N),
                           mid(J, O),
                           mid(K, P)]

                global face_registered, cube, cube_solving, solution, sol
                if face_registered < 6:
                    scan_cube_faces(imgContour, boxMids)
                elif getColors(imgContour, boxMids) != cube[0:9] and cube_solving == False:
                    cv.putText(imgContour, "Show Front Face", (200, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv.arrowedLine(
                        imgContour, boxMids[0], boxMids[6], (0, 0, 255), 2)
                    cv.arrowedLine(
                        imgContour, boxMids[1], boxMids[7], (0, 0, 255), 2)
                    cv.arrowedLine(
                        imgContour, boxMids[2], boxMids[8], (0, 0, 255), 2)
                else:
                    if cube_solving == False:
                        cube_pos = color_to_pos()
                        sol = cubeSolver.solve(cube_pos)
                        sol = sol.replace('R2', 'R R')
                        sol = sol.replace('F2', 'F F')
                        sol = sol.replace('U2', 'U U')
                        sol = sol.replace('B2', 'B B')
                        sol = sol.replace('D2', 'D D')
                        sol = sol.replace('L2', 'L L')
                        sol = sol.split()
                        for i in range(len(sol)):
                            solution += sol[i]
                            solution += ' '
                    cube_solving = True
                    if(solve_step == len(sol)):
                        cv.putText(imgContour, "Solved", (230, 350),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    solve_cube(imgContour, sol, boxMids)


while True:
    _, frame = cap.read()
    img = frame.copy()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(hsv, (5, 5), 0)
    mask_b = cv.inRange(blur, np.array(
        [74, 81, 178]), np.array([112, 255, 255]))
    mask_g = cv.inRange(blur, np.array(
        [40, 128, 180]), np.array([78, 255, 255]))
    mask_y = cv.inRange(blur, np.array(
        [21, 159, 167]), np.array([40, 235, 255]))
    mask_o = cv.inRange(blur, np.array(
        [0, 120, 190]), np.array([20, 186, 255]))
    mask_w = cv.inRange(blur, np.array([0, 0, 190]), np.array([130, 39, 255]))
    mask_r = cv.inRange(blur, np.array(
        [165, 48, 213]), np.array([179, 170, 255]))
    mask = mask_b + mask_g + mask_o + mask_y + mask_w + mask_r
    kernel = np.ones((9, 9))
    imgDil = cv.dilate(mask, kernel, iterations=1)
    # cv.imshow("mask", imgDil)
    plotCube(imgDil, img)
    cv.imshow("Cube", img)
    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()
