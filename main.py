import cv2
import mediapipe as mp
import time
import numpy as np
import sys

# open camera, close program if camera doesnt/cant open.
cap = cv2.VideoCapture(0)
if not cap:
    print("Cannot open camera")
    exit()

start_time = time.time()
button_radius = 35

# set up mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

def cursor_inside_button(cursor, button_pos, button_radius):
    x, y = cursor
    bx, by = button_pos

    dx = x - bx
    dy = y - by

    distance = (dx**2 + dy**2) ** 0.5

    return distance < button_radius

def get_connector_points(center, button_pos, center_radius, button_radius, gap=8):
    cx, cy = center
    bx, by = button_pos

    dx = bx - cx
    dy = by - cy

    distance = (dx**2 + dy**2) ** 0.5

    # start slightly outside the center circle
    start_distance = center_radius + gap

    start_x = int(cx + (dx / distance) * start_distance)
    start_y = int(cy + (dy / distance) * start_distance)

    # stop at the edge of the button circle
    end_x = int(bx - (dx / distance) * button_radius)
    end_y = int(by - (dy / distance) * button_radius)

    return (start_x, start_y), (end_x, end_y)

def draw_menu(frame, center, cursor):
    cx, cy = center

    # button positions
    draw_pos = (cx - 70, cy - 100)
    undo_pos = (cx + 70, cy - 100)
    erase_pos = (cx + 130, cy)
    size_pos = (cx + 70, cy + 100)
    color_pos = (cx - 70, cy + 100)
    clear_pos = (cx - 130, cy)

    # black dot
    cv2.circle(frame, (cx, cy), 20, (0, 0, 0), -1)

    center_radius = 20

    draw_start, draw_end = get_connector_points((cx, cy), draw_pos, center_radius, button_radius)
    undo_start, undo_end = get_connector_points((cx, cy), undo_pos, center_radius, button_radius)
    erase_start, erase_end = get_connector_points((cx, cy), erase_pos, center_radius, button_radius)
    size_start, size_end = get_connector_points((cx, cy), size_pos, center_radius, button_radius)
    color_start, color_end = get_connector_points((cx, cy), color_pos, center_radius, button_radius)
    clear_start, clear_end = get_connector_points((cx, cy), clear_pos, center_radius, button_radius)

    # connector lines
    cv2.line(frame, draw_start, draw_end, (255, 150, 0), 3)
    cv2.line(frame, undo_start, undo_end, (200, 150, 255), 3)
    cv2.line(frame, erase_start, erase_end, (0, 0, 255), 3)
    cv2.line(frame, size_start, size_end, (0, 165, 255), 3)
    cv2.line(frame, color_start, color_end, (0, 255, 0), 3)
    cv2.line(frame, clear_start, clear_end, (180, 180, 180), 3)

    # draw button
    if cursor_inside_button(cursor, draw_pos, button_radius):
        cv2.circle(frame, draw_pos, button_radius, (255, 150, 0), -1)
    else:
        cv2.circle(frame, draw_pos, button_radius, (255, 150, 0), 3)
    cv2.putText(
        frame,
        "draw",
        (draw_pos[0] - 22, draw_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # undo button
    if cursor_inside_button(cursor, undo_pos, button_radius):
        cv2.circle(frame, undo_pos, button_radius, (200, 150, 255), -1)
    else:
        cv2.circle(frame, undo_pos, button_radius, (200, 150, 255), 3)
    cv2.putText(
        frame,
        "undo",
        (undo_pos[0] - 22, undo_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # erase button
    if cursor_inside_button(cursor, erase_pos, button_radius):
        cv2.circle(frame, erase_pos, button_radius, (0, 0, 255), -1)
    else:
        cv2.circle(frame, erase_pos, button_radius, (0, 0, 255), 3)
    cv2.putText(
        frame,
        "erase",
        (erase_pos[0] - 24, erase_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # size button
    if cursor_inside_button(cursor, size_pos, button_radius):
        cv2.circle(frame, size_pos, button_radius, (0, 165, 255), -1)
    else:
        cv2.circle(frame, size_pos, button_radius, (0, 165, 255), 3)
    cv2.putText(
        frame,
        "size",
        (size_pos[0] - 18, size_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # color button
    if cursor_inside_button(cursor, color_pos, button_radius):
        cv2.circle(frame, color_pos, button_radius, (0, 255, 0), -1)
    else:
        cv2.circle(frame, color_pos, button_radius, (0, 255, 0), 3)
    cv2.putText(
        frame,
        "color",
        (color_pos[0] - 20, color_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # clear button
    if cursor_inside_button(cursor, clear_pos, button_radius):
        cv2.circle(frame, clear_pos, button_radius, (180, 180, 180), -1)
    else:
        cv2.circle(frame, clear_pos, button_radius, (180, 180, 180), 3)
    cv2.putText(
        frame,
        "clear",
        (clear_pos[0] - 20, clear_pos[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    return {
        "draw": draw_pos,
        "undo": undo_pos,
        "erase": erase_pos,
        "size": size_pos,
        "color": color_pos,
        "clear": clear_pos
    }

def draw_help(frame):
    height, width, _ = frame.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    cv2.putText(frame, "V  Open/Close Menu",
                (20, height - 70), font, 0.5, color, 1)

    cv2.putText(frame, "Pinch to Select",
                (20, height - 45), font, 0.5, color, 1)

    cv2.putText(frame, "Q to Quit",
                (20, height - 20), font, 0.5, color, 1)

def draw_size_menu(frame, center, brush_size):
    cx, cy = center

    panel_width = 110
    panel_height = 220

    left = cx - panel_width // 2
    top = cy - panel_height // 2
    right = cx + panel_width // 2
    bottom = cy + panel_height // 2

    cv2.rectangle(frame, (left, top), (right, bottom), (40, 40, 40), -1)

    slider_top = top + 20
    slider_bottom = bottom - 45
    slider_x = cx

    cv2.line(frame, (slider_x, slider_top), (slider_x, slider_bottom), (120, 120, 120), 5)

    min_brush_size = 1
    max_brush_size = 30

    knob_y = int(np.interp(brush_size, [min_brush_size, max_brush_size], [slider_bottom, slider_top]))

    cv2.circle(frame, (slider_x, knob_y), 7, (0, 165, 255), -1)

    if dragging_size:
        cv2.circle(frame, (slider_x, knob_y), 11, (255, 255, 255), 2)

    cv2.putText(frame, str(brush_size), (cx - 10, bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return slider_x, slider_top, slider_bottom, knob_y

def draw_color_menu(frame, center, cursor, draw_color):
    cx, cy = center

    panel_width = 180
    panel_height = 100

    left = cx - panel_width // 2
    top = cy - panel_height // 2
    right = cx + panel_width // 2
    bottom = cy + panel_height // 2

    cv2.rectangle(frame, (left, top), (right, bottom), (40, 40, 40), -1)

    colors = {
        "pink": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255),
        "green": (80, 200, 60),

        "blue": (255, 120, 70),
        "purple": (200, 90, 170),
        "teal": (180, 200, 40),
        "white": (255, 255, 255)
    }

    start_x = left + 25
    start_y = top + 25
    spacing = 42
    size = 16

    buttons = {}

    names = list(colors.keys())

    for i, name in enumerate(names):
        row = i // 4
        col = i % 4

        x = start_x + col * spacing
        y = start_y + row * spacing

        color = colors[name]

        # draw color square
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)

        selected = color == draw_color
        hovering = cursor_inside_button(cursor, (x, y), size)

        if selected:
            thickness = 4
        elif hovering:
            thickness = 2
        else:
            thickness = 0

        if thickness > 0:
            cv2.rectangle(frame, (x - size - 3, y - size - 3), (x + size + 3, y + size + 3), (255, 255, 255), thickness)

        buttons[name] = (x,y)

    return buttons, colors

def draw_status(frame, mode, brush_size, draw_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    mode_text = "None"
    if mode == "draw":
        mode_text = "Draw"
    elif mode == "erase":
        mode_text = "Erase"

    cv2.putText(frame, f"Tool: {mode_text}", (20, 30), font, 0.6, white, 2)    

    cv2.putText(frame, f"Size: {brush_size}", (20, 60), font, 0.6, white, 2)

    cv2.circle(frame, (30, 90), 12, draw_color, -1)

    cv2.circle(frame, (30, 90), 12, white, 2)

    cv2.putText(frame, "Color", (50, 97), font, 0.6, white, 2)

def save_history(canvas, canvas_history, limit=20):
    canvas_history.append(canvas.copy())

    if len(canvas_history) > limit:
        canvas_history.pop(0)

def clamp_menu_center(center, frame, margin_x, margin_y):
    x, y = center

    height, width, _ = frame.shape

    x = max(margin_x, min(width - margin_x, x))
    y = max(margin_y, min(height - margin_y, y))

    return (x, y)

# start of line line variables
prev_x = None
prev_y = None

# menu variables
menu_open = False
menu_center = None
peace_active = False
wait_for_release = False

brush_size = 5

pinch_active = False

canvas_history = []

size_menu_open = False
size_menu_center = None
color_menu_open = False
color_menu_center = None

draw_color = (0, 0, 255)

# app state variables
mode = None
dragging_size = False

# create canvas to draw on
ret, frame = cap.read()
canvas = np.zeros_like(frame)

# if no frame thenclose program
if not ret:
    print("Cannot capture frame")
    sys.exit()

# loop that reads frames 1 by 1 and processes them
while cap.isOpened():
    ret, frame = cap.read()

    # mirror the image/video
    frame = cv2.flip(frame, 1)

    # convert bgr to rgb, opencv uses bgr while mediapipe uses rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert opencv image to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # mediapipe video mode requires a timestamp
    timestamp_ms = int((time.time() - start_time) * 1000)

    # detect the hand, detect_for_video is a object that contains 3 things hand_landmarks, hand_world_landmarks, and handedness
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # if hand detected, set hand vairable to the first hand detected. result.hand_landmarks is a list of all hands detected so 0 is the first one.
    # each hand is also a list so hand is now a list of 21 points, each point is an object that contains x,y,z coordinates
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        thumb_tip = hand[4]

        index_tip = hand[8]
        index_pip = hand[6]

        middle_tip = hand[12]
        middle_pip = hand[10]

        ring_tip = hand[16]
        ring_pip = hand[14]

        pinky_tip = hand[20]
        pinky_pip = hand[18]

        # check which fingers are up
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y

        # calculate distance between index and thumb
        dx = index_tip.x - thumb_tip.x
        dy = index_tip.y - thumb_tip.y
        distance = (dx**2 + dy**2) ** 0.5

        # pinching threshold
        pinching = distance < 0.10

        # midpoint between index tip and thumb tip for cursor
        cursor_x = (index_tip.x + thumb_tip.x) / 2
        cursor_y = (index_tip.y + thumb_tip.y) / 2

        # convert mediapipe landmark coordinates to opencv pixel coordinates
        height, width, _ = frame.shape
        x = int(cursor_x * width)
        y = int(cursor_y * height)

        # define peace sign
        peace_sign = (index_up and middle_up and not ring_up and not pinky_up)

        # menu toggle
        if peace_sign and not peace_active:
            if size_menu_open and not dragging_size:
                size_menu_open = False
                size_menu_center = None

            elif color_menu_open:
                color_menu_open = False
                color_menu_center = None

            else:
                menu_open = not menu_open
                if menu_open:
                    menu_center = clamp_menu_center((x, y), frame, 165, 140)

            peace_active = True

        elif not peace_sign:
            peace_active = False

        # draw menu
        buttons = {}
        if menu_open and menu_center is not None:
            buttons = draw_menu(frame, menu_center, (x,y))

        pinch_clicked = False
        if pinching and not pinch_active:
            pinch_clicked = True
            pinch_active = True
        elif not pinching:
            pinch_active = False

        # draw size menu
        if size_menu_open and size_menu_center is not None:
            slider_x, slider_top, slider_bottom, knob_y = draw_size_menu(frame, size_menu_center, brush_size)

            dx = x - slider_x
            dy = y - knob_y
            knob_distance = (dx**2 + dy**2) ** 0.5

            if pinch_clicked and knob_distance < 20:
                dragging_size = True

            if dragging_size and pinching:
                raw_size = np.interp(y, [slider_top, slider_bottom], [30, 1])

                brush_size = round(raw_size / 2) * 2
                brush_size = max(1, min(30, brush_size))

            if dragging_size and not pinching:
                dragging_size = False
                size_menu_open = False

        # draw color menu
        if color_menu_open and color_menu_center is not None:
            color_buttons, colors = draw_color_menu(frame, color_menu_center, (x,y), draw_color)

        if color_menu_open and pinch_clicked:
            for name, pos in color_buttons.items():
                if cursor_inside_button((x,y), pos, 18):
                    draw_color = colors[name]
                    color_menu_open = False
                    color_menu_center = None
                    wait_for_release = True
                    break

        # button/mode selector
        if menu_open and pinch_clicked:
            if cursor_inside_button((x,y), buttons["draw"], button_radius):
                mode = "draw"
                menu_open = False
                wait_for_release = True

            elif cursor_inside_button((x, y), buttons["undo"], button_radius):
                if canvas_history:
                    canvas = canvas_history.pop()
                menu_open = False
                wait_for_release = True

            elif cursor_inside_button((x, y), buttons["erase"], button_radius):
                mode = "erase"
                menu_open = False
                wait_for_release = True

            elif cursor_inside_button((x, y), buttons["size"], button_radius):
                menu_open = False
                size_menu_open = True
                size_menu_center = clamp_menu_center((x,y), frame, 65, 120)
                wait_for_release = True

            elif cursor_inside_button((x, y), buttons["color"], button_radius):
                menu_open = False
                color_menu_open = True
                color_menu_center = clamp_menu_center((x,y), frame, 100, 60)
                wait_for_release = True

            elif cursor_inside_button((x, y), buttons["clear"], button_radius):
                save_history(canvas, canvas_history)
                canvas[:] = 0
                menu_open = False
                wait_for_release = True

        if not pinching:
            wait_for_release = False

        # draw the cursor
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        if pinching and not menu_open and not size_menu_open and not color_menu_open and not wait_for_release:
            if prev_x is not None and prev_y is not None:
                smooth_x = int(prev_x * 0.7 + x * 0.3)
                smooth_y = int(prev_y * 0.7 + y * 0.3)

                if mode == "draw":
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, brush_size)

                elif mode == "erase":
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (0,0,0), brush_size * 3)

                prev_x = smooth_x
                prev_y = smooth_y

            else:
                # new storke is beginning
                save_history(canvas, canvas_history)
                prev_x = x
                prev_y = y

        else:
            prev_x = None
            prev_y = None

    display_frame = cv2.add(frame, canvas)

    draw_help(display_frame)
    draw_status(display_frame, mode, brush_size, draw_color)

    cv2.imshow("Webcam Feed", display_frame)

    # exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()