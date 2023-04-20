import cv2
import mediapipe as mp
import numpy as np
from Move import Move
from Player import Player


def apply_filters(image):
    matriz = np.ones(image.shape, dtype="uint8") * 5
    brightness_frame = cv2.add(image, matriz)

    blurred_frame = cv2.medianBlur(brightness_frame, 5)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)

    hsv = cv2.cvtColor(opening, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 254], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    return cv2.bitwise_and(image, image, mask=mask)


def process_players_hands(player1, player2, hands):
    left_most_hand_center_x = None
    right_most_hand_center_x = None

    for hand_landmarks in hands:
        center_x = get_hand_center_x(hand_landmarks)

        if left_most_hand_center_x is None or center_x < left_most_hand_center_x:
            left_most_hand_center_x = center_x
            player1.hand = hand_landmarks

        if right_most_hand_center_x is None or center_x > right_most_hand_center_x:
            right_most_hand_center_x = center_x
            player2.hand = hand_landmarks


def get_hand_center_x(hand_landmarks):
    x = [landmark.x for landmark in hand_landmarks.landmark]
    return sum(x) / len(hand_landmarks.landmark)


def draw_hand_rectangle(image, player):
    image_height, image_width, _ = image.shape

    x = [
        landmark.x * image_width
        for landmark in player.hand.landmark
    ]

    y = [
        landmark.y * image_height
        for landmark in player.hand.landmark
    ]

    xmin, xmax = int(min(x)), int(max(x))
    ymin, ymax = int(min(y)), int(max(y))

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), player.color, 2)


def get_fingertip_positions(hand_landmarks):
    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_finger, middle_finger, ring_finger, pinky_finger


def get_hand_base_position(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]


def get_distance_between_landmarks(landmark1, landmark2):
    return np.linalg.norm(
        np.array([landmark1.x, landmark1.y])
        - np.array([landmark2.x, landmark2.y])
    )


def process_player_move(player):
    hand_landmarks = player.hand

    index_finger, middle_finger, ring_finger, pinky_finger = get_fingertip_positions(
        hand_landmarks)

    base_of_hand = get_hand_base_position(hand_landmarks)

    index_middle_distance = get_distance_between_landmarks(
        index_finger, middle_finger)

    middle_ring_distance = get_distance_between_landmarks(
        middle_finger, ring_finger)

    ring_pinky_distance = get_distance_between_landmarks(
        ring_finger, pinky_finger)

    index_distance_from_base = get_distance_between_landmarks(
        index_finger, base_of_hand)

    pinky_distance_from_base = get_distance_between_landmarks(
        pinky_finger, base_of_hand)

    play = None

    if (
        index_middle_distance < 0.03
        and middle_ring_distance < 0.03
        and ring_pinky_distance < 0.03
        and index_distance_from_base < 0.1
        and pinky_distance_from_base < 0.1
    ):
        play = Move.ROCK
    elif (
        index_middle_distance < 0.04
        and middle_ring_distance < 0.04
        and ring_pinky_distance < 0.04
        and index_distance_from_base > 0.1
        and pinky_distance_from_base > 0.1
    ):
        play = Move.PAPER
    elif (
        index_middle_distance > 0.03
        and middle_ring_distance > 0.1
        and ring_pinky_distance > 0.02
        and index_distance_from_base > 0.1
        and pinky_distance_from_base < 0.1
    ):
        play = Move.SCISSORS
    else:
        player.current_move = None

    if play is not None:
        player.last_move = player.current_move
        player.current_move = play


def draw_players_move(image, player1, player2):
    player1_display_position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    text = f"player1: {player1.current_move.value}"
    cv2.putText(
        image,
        text,
        player1_display_position,
        font,
        font_scale,
        player1.color,
        thickness,
    )

    text = f"player2: {player2.current_move.value}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = image.shape[1] - text_size[0] - 50

    player2_display_position = (x, 50)
    cv2.putText(
        image,
        text,
        player2_display_position,
        font,
        font_scale,
        player2.color,
        thickness,
    )


def process_round(player1, player2):
    zero_score = player1.points == 0 and player2.points == 0
    move_changed = (player1.current_move != player1.last_move
                    or player2.current_move != player2.last_move)

    if zero_score or move_changed:
        is_the_same_move = player1.current_move == player2.current_move

        if is_the_same_move:
            return "Empate"
        elif (
            (
                player1.current_move == Move.ROCK
                and player2.current_move == Move.SCISSORS
            )
            or (
                player1.current_move == Move.PAPER
                and player2.current_move == Move.ROCK
            )
            or (
                player1.current_move == Move.SCISSORS
                and player2.current_move == Move.PAPER
            )
        ):
            winner = "Player 1"
            player1.points += 1
        else:
            winner = "Player 2"
            player2.points += 1

        return f"{winner} ganhou {player1.current_move.value} x {player2.current_move.value}"
    else:
        return round_result


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.0
)

cap = cv2.VideoCapture("pedra-papel-tesoura.mp4")

player1 = Player(color=(0, 255, 0))
player2 = Player(color=(0, 0, 255))

round_result = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = apply_filters(image)

    results = hands.process(res)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if results.multi_hand_landmarks:
        process_players_hands(
            player1, player2, hands=results.multi_hand_landmarks)

        if player1.hand != player2.hand:
            for player in [player1, player2]:
                if player.hand is not None:
                    draw_hand_rectangle(image, player)
                    process_player_move(player)

        if player1.current_move and player2.current_move:
            draw_players_move(image, player1, player2)
            round_result = process_round(player1, player2)

    cv2.putText(
        image,
        round_result,
        (int(image.shape[1] * 0.5), 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
    text = f"{player1.points} x {player2.points}"
    cv2.putText(
        image,
        text,
        (int(image.shape[1] * 0.5), 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )

    cv2.imshow("Checkpoint", image)
    if cv2.waitKey(25) & 0xFF == ord("r"):
        break

cap.release()
cv2.destroyAllWindows()


# mp_drawing.draw_landmarks(
#                         image,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style(),
#                     )
