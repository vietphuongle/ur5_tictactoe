import cv2
import numpy as np
import time

import socket



from math import inf as infinity
from random import choice
import platform
import time
from os import system

count_img=0

####################################
dd=0.025

dx=0.50938 + 0.018

poses=[[0.50938, -0.19284, 0.44251, 0.563, 1.479, 0.548], #0
       [0.50938, -0.29194, 0.44251, 0.563, 1.479, 0.548], #1
       [0.50938, -0.38723, 0.44251, 0.563, 1.479, 0.548], #2
       [0.50938, -0.19288, 0.34702, 0.563, 1.479, 0.548],# 3
       [0.50938, -0.29553, 0.34702, 0.563, 1.479, 0.548],# 4
       [0.50938, -0.38994, 0.34702, 0.563, 1.479, 0.548],# 5
       [0.50938, -0.19288, 0.25648, 0.563, 1.479, 0.548],# 6
       [0.50938, -0.29671, 0.25648, 0.563, 1.479, 0.548],# 7
       [0.50938, -0.38934, 0.25648, 0.563, 1.479, 0.548],# 8
]
pose_mid = [0.46234, -0.07934, 0.25644, 0.563, 1.478, 0.548] # trung gian truoc moi lan danh/ danh xong rut ve
#pose_home =  [0.48007, -0.07875, 0.08284, 0.939, 2.905, -0.138]# home 



HOST = "192.168.1.102"
PORT = 30002
s =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

def move_to_pose(pose):
    s.send(("movel(p[" + str(pose[0]) +"," + str(pose[1])+"," + str(pose[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=5)" + "\n").encode("utf8"))
    data = s.recv(1024)
    time.sleep(7) 

def draw_X(pose):
    move_to_pose(pose)
    x = pose[0]
    y = pose[1]
    z = pose[2]
    point1 = [x, y + dd, z + dd]
    point2 = [x, y - dd, z - dd]
    point3 = [x, y - dd, z + dd]
    point4 = [x, y + dd, z - dd]
        
    s.send(("movel(p[" + str(point1[0]) +"," + str(point1[1])+"," + str(point1[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2) 

    s.send(("movel(p[" + str(dx) +"," + str(point1[1])+"," + str(point1[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2)

    s.send(("movel(p[" + str(dx) +"," + str(point2[1])+"," + str(point2[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2)

    s.send(("movel(p[" + str(point2[0]) +"," + str(point2[1])+"," + str(point2[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2)

    s.send(("movel(p[" + str(point3[0]) +"," + str(point3[1])+"," + str(point3[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2) 

    s.send(("movel(p[" + str(dx) +"," + str(point3[1])+"," + str(point3[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2) 

    s.send(("movel(p[" + str(dx) +"," + str(point4[1])+"," + str(point4[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2)
    
    s.send(("movel(p[" + str(point4[0]) +"," + str(point4[1])+"," + str(point4[2])+", " + str(pose[3])+", " + str(pose[4])+", " + str(pose[5])+"], a=1.10,t=1)" + "\n").encode("utf8"))
    time.sleep(2)

    move_to_pose(pose_mid)

####################################


HUMAN = -1
COMP = +1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

#ROIs = []
#old_ROIs = []

old_img = []
curr_img = []

# get default corners
corners = np.array([[838, 572], [1023, 764]])
#points = calculate_points(corners)

# Initialize the video capture object
cap = cv2.VideoCapture(0)


not_found_corners = True
is_human_turn = True
is_first_run = True

######################## tinh nuoc di ######################


def evaluate(state):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score


def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False


def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, HUMAN) or wins(state, COMP)


def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells


def valid_move(x, y):
    """
    A move is valid if the chosen cell is empty
    :param x: X coordinate
    :param y: Y coordinate
    :return: True if the board[x][y] is empty
    """
    if [x, y] in empty_cells(board):
        return True
    else:
        return False


def set_move(x, y, player):
    """
    Set the move on board, if the coordinates are valid
    :param x: X coordinate
    :param y: Y coordinate
    :param player: the current player
    """
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False


def minimax(state, depth, player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best


def clean():
    """
    Clears the console
    """
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')


def render(state, c_choice, h_choice):
    """
    Print the board on console
    :param state: current state of the board
    """

    chars = {
        -1: h_choice,
        +1: c_choice,
        0: ' '
    }
    str_line = '---------------'

    print('\n' + str_line)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + str_line)


def ai_turn(c_choice, h_choice):
    """
    It calls the minimax function if the depth < 9,
    else it choices a random coordinate.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    clean()
    print(f'Computer turn [{c_choice}]')
    render(board, c_choice, h_choice)

    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, COMP)
        x, y = move[0], move[1]

    set_move(x, y, COMP)
    time.sleep(1)

    return x*3+y


def human_turn(c_choice, h_choice, pos):
    """
    The Human plays choosing a valid move.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    # Dictionary of valid moves
    move = -1
    moves = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    clean()
    print(f'Human turn [{h_choice}]')
    render(board, c_choice, h_choice)

    while move < 1 or move > 9:
        try:
            move = pos  #int(input('Use numpad (1..9): '))
            coord = moves[move]
            can_move = set_move(coord[0], coord[1], HUMAN)

            if not can_move:
                print('Bad move')
                move = -1
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

######################## het tinh nuoc di ######################



d=12

def compare2ROIs(img1,img2):
    # Convert to Gray
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Choose a threshold value (127 in this case)
    thresh_value = 127

    # Threshold the grayscale image
    _, b1 = cv2.threshold(img1, thresh_value, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(img2, thresh_value, 255, cv2.THRESH_BINARY)

    # Check if the sizes of the two images are the same
    if b1.shape != b2.shape:
        # Resize the images to the same dimensions
        b2 = cv2.resize(b2, (b1.shape[1], b1.shape[0]))

    # Apply the XOR operation to the two images
    diff = cv2.bitwise_xor(b1, b2)

    # Count the number of white pixels
    num_white_pixels = cv2.countNonZero(diff)

    # Calculate the total number of pixels
    total_pixels = diff.shape[0] * diff.shape[1]

    # Calculate the ratio of white pixels
    white_pixel_ratio = num_white_pixels / total_pixels
    #print(white_pixel_ratio*100)

    return white_pixel_ratio
    # Save the difference image
    #cv2.imwrite('diff'+str(idx)+'.png', diff)

def detect_2coners_of_board(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    # Convert corner coordinates to integers
    corners = np.round(corners).astype(int)

    if ids is not None:
        # print(sorted_indices)
        # print(corners)

        if len(ids) > 2:
            print("nhieu hon 2 aruco")
            return False, None
        elif len(ids) < 2:
            return False, None
        else:
            # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
            # frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if corners[0][0][0][0] < corners[1][0][0][0]:
                return True, np.array([corners[0][0][0], corners[1][0][0]])
            else:
                return True, np.array([corners[1][0][0], corners[0][0][0]])
    return False, None

def calculate_points(corners):
    points = np.zeros((16,2),dtype=int)
    if corners[0][0] < corners[1][0]:
        points[0] = np.array([int(corners[0][0]),int(corners[0][1])])
        points[15] = np.array([int(corners[1][0]),int(corners[1][1])])
    else:
        points[0] = np.array([int(corners[1][0]),int(corners[1][1])])
        points[15] = np.array([int(corners[0][0]),int(corners[0][1])])

    points[3] = np.array([points[15][0],points[0][1]])
    points[12] = np.array([points[0][0],points[15][1]])

    points[1] = np.array([int((points[3][0]-points[0][0])/3 + points[0][0]),points[0][1]])
    points[13] = np.array([int((points[15][0]-points[12][0])/3 + points[12][0]),points[15][1]])

    points[2] = np.array([int((points[3][0]-points[0][0])/3 * 2 + points[0][0]),points[0][1]])
    points[14] = np.array([int((points[15][0]-points[12][0])/3 * 2 + points[12][0]),points[15][1]])

    points[4] = np.array([points[0][0],int((points[12][1]-points[0][1])/3 + points[0][1])])
    points[7] = np.array([points[15][0],int((points[15][1]-points[3][1])/3 + points[0][1])])

    points[8] = np.array([points[0][0],int((points[12][1]-points[0][1])/3 *2 + points[0][1])])
    points[11] = np.array([points[15][0],int((points[15][1]-points[3][1])/3 *2 + points[0][1])])

    points[5] = np.array([points[1][0],points[4][1]])
    points[6] = np.array([points[2][0], points[4][1]])
    points[9] = np.array([points[1][0],points[8][1]])
    points[10] = np.array([points[2][0], points[8][1]])
    #print(points)
    return points

def draw_line(image,point1,point2):
    cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 2)
    return image

def draw_board(image, points):
    '''
    #box
    draw_line(image,points[0],points[3])
    draw_line(image, points[0], points[12])
    draw_line(image,points[3],points[15])
    draw_line(image, points[12], points[15])

    draw_line(image, points[1], points[13])
    draw_line(image, points[2], points[14])

    draw_line(image, points[4], points[7])
    draw_line(image, points[8], points[11])

    draw_line(image,points[5],points[6])
    draw_line(image, points[6], points[10])
    draw_line(image,points[10],points[9])
    draw_line(image, points[9], points[5])

    #cv2.rectangle(frame, (int(points[0][0]), int(points[0][1])), (int(points[15][0]), int(points[15][1])), (255, 0, 0), 2)


    #cv2.rectangle(frame, (int(points[0][0]), int(points[0][1])), (int(points[15][0]), int(points[15][1])), (255, 0, 0), 2)
    '''

    draw_line(image, points[0], points[5])
    draw_line(image, points[1], points[6])
    draw_line(image, points[2], points[7])
    draw_line(image, points[4], points[9])
    draw_line(image, points[5], points[10])
    draw_line(image, points[6], points[11])
    draw_line(image, points[8], points[13])
    draw_line(image, points[9], points[14])
    draw_line(image, points[10], points[15])

    return image

def extract_cells(img):
    #ret, corners = detect_2coners_of_board(img)
    # Define the coordinates of the top-left and bottom-right corners of the board
    ret, board_coords = detect_2coners_of_board(img)

    #print(board_coords)
    # Calculate the width and height of each cell
    cell_width = int((board_coords[1][0] - board_coords[0][0]) / 3)
    cell_height = int((board_coords[1][1] - board_coords[0][1]) / 3)

    # Define the coordinates of the top-left and bottom-right corners of each cell
    cell_coords = []
    cell_imgs = []
    for j in range(3):
        for i in range(3):
            x1 = board_coords[0][0] + i * cell_width
            y1 = board_coords[0][1] + j * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            cell_coords.append(((x1, y1), (x2, y2)))

    # Extract each cell ROI and save it as a separate image
    for i, coords in enumerate(cell_coords):
        (x1, y1), (x2, y2) = coords
        #print(coords)
        cell_img = img[y1+d:y2-d, x1+d:x2-d]
        #cv2.imwrite('cell_'+str(n)+str(i)+'.jpg', cell_img)
        cell_imgs.append(cell_img)
    return cell_imgs

def detect_human_turn(old_img, curr_img, board_vector):
    #board = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    for n in range(10, 18):
        #print('xet cap ', n, ' ', n + 1)
        #old_img = cv2.imread('img_' + str(n) + '.png')
        #curr_img = cv2.imread('img_' + str(n + 1) + '.png')
        old_cells = extract_cells(old_img)
        curr_cells = extract_cells(curr_img)
        max = 0
        p = -1
        for i in range(len(old_cells)):
            if board_vector[i] == 0:
                r = compare2ROIs(old_cells[i], curr_cells[i])
                if r > max:
                    max = r
                    p = i
        #board[p] = 1
        #print(p, ' ', max)
    return p


#chuong trinh chinh
frame =[]
print("dang tim 2 aruco")
while not_found_corners:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Detect the corner of board
    ret1, corners_detected = detect_2coners_of_board(frame)
    # If markers are detected
    if ret1:
        corners = corners_detected
        #print(points[0],"-",points[15])
        not_found_corners = False

count_img += 1
cv2.imwrite("save_"+str(count_img)+".png",frame)

print("tim thay 2 aruco")
#tinh 15 points

input('Enter de lay anh dau tien va robot ve mid')
ret, old_img = cap.read()
count_img += 1
cv2.imwrite("save_"+str(count_img)+".png",old_img)
move_to_pose(pose_mid)

"""
Main function that calls all functions
"""
clean()
h_choice = 'O'  # X or O

c_choice = 'X'  # X or O
first = 'Y'  # if human is the first

'''
# Human chooses X or O to play
while h_choice != 'O' and h_choice != 'X':
    try:
        print('')
        h_choice = input('Choose X or O\nChosen: ').upper()
    except (EOFError, KeyboardInterrupt):
        print('Bye')
        exit()
    except (KeyError, ValueError):
        print('Bad choice')

# Setting computer's choice
if h_choice == 'X':
    c_choice = 'O'
else:
    c_choice = 'X'

# Human may starts first
clean()
while first != 'Y' and first != 'N':
    try:
        first = input('First to start?[y/n]: ').upper()
    except (EOFError, KeyboardInterrupt):
        print('Bye')
        exit()
    except (KeyError, ValueError):
        print('Bad choice')
'''
# Main loop of this game
while len(empty_cells(board)) > 0 and not game_over(board):
    if first == 'N':
        ai_turn(c_choice, h_choice)
        first = ''
    input('Enter khi human turn xong')
    ret, curr_img = cap.read()
    count_img += 1
    cv2.imwrite("save_"+str(count_img)+".png",curr_img)
    pos_human = detect_human_turn(old_img,curr_img,[board[0][0],board[0][1],board[0][2],board[1][0],board[1][1],board[1][2],board[2][0],board[2][1],board[2][2]])
    print('human: ',pos_human)
    human_turn(c_choice, h_choice,pos_human+1)

    pos_ai=ai_turn(c_choice, h_choice)
    print('ai: ',pos_ai)
    #input("Enter de robot di chuyen")
    if pos_ai != None:
        draw_X(poses[pos_ai])
        #input("Enter khi AI danh xong")
        #time.sleep(3)
        ret, old_img = cap.read()
        count_img += 1
        cv2.imwrite("save_"+str(count_img)+".png",old_img)

# Game over message
if wins(board, HUMAN):
    clean()
    print(f'Human turn [{h_choice}]')

    render(board, c_choice, h_choice)
    
    print('YOU WIN!')
elif wins(board, COMP):
    clean()
    print(f'Computer turn [{c_choice}]')
    render(board, c_choice, h_choice)
    print('YOU LOSE!')
else:
    clean()
    render(board, c_choice, h_choice)
    print('DRAW!')

exit()


'''
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if detect_corners:
        # Detect the corner of board
        ret1, corners_detected = detect_2coners_of_board(frame)
        # If markers are detected
        if ret1:
            corners = corners_detected
            points = calculate_points(corners)
            #print(points[0],"-",points[15])
            detect_corners = False

    if is_first_run:
        old_ROIs=extract_ROI(frame,points)
        is_first_run = False


    if is_human_turn:
        old_ROIs = extract_ROI(frame, points)
        time.sleep(10)
        ROIs=extract_ROI(frame,points)
        pos_human = detect_human_position(ROIs, old_ROIs, [0,0,0,0,0,0,0,0,0])
        print(pos_human)
        is_human_turn=False
        #if pos_human>0:
            #is_human_turn = False
            #print(pos_human)
    #frame = draw_board(frame, points)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit the loop if the 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        #for i in range(len(ROIs)):
            #roi = np.array(ROIs[i])
            #cv2.imwrite('roi'+str(i)+'.jpg', roi)
        break
    if key == ord('n'):
        is_human_turn = True
'''

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
s.close()