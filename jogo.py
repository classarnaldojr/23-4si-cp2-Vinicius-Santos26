import cv2
import mediapipe as mp
import numpy as np
from enum import Enum


class Move(Enum):
    ROCK = "Pedra"
    PAPER = "Papel"
    SCISSORS = "Tesoura"


class Player:
    hand = None
    current_move = None
    last_move = None
    points = 0
    color = (0, 0, 0)

    def __init__(self, color):
        self.color = color


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.0
)

cap = cv2.VideoCapture("pedra-papel-tesoura.mp4")

player1 = Player(color=(0, 255, 0))
player2 = Player(color=(0, 0, 255))


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    matriz = np.ones(image.shape, dtype="uint8") * 5

    img2 = cv2.add(image, matriz)

    blurred_frame = cv2.medianBlur(img2, 5)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)

    hsv = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)

    # Definindo os valores mínimos e máximos para as cores branco
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 254], dtype=np.uint8)

    # Criando a máscara para as cores branco e preto
    mask = cv2.inRange(hsv, lower_white, upper_white)

    #  Aplicando a máscara à imagem original
    res = cv2.bitwise_and(image, image, mask=mask)

    results = hands.process(res)

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if results.multi_hand_landmarks:
        leftmost_hand_center_x = None
        rightmost_hand_center_x = None

        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]

            center_x = sum(x) / len(hand_landmarks.landmark)

            if leftmost_hand_center_x is None or center_x < leftmost_hand_center_x:
                leftmost_hand_center_x = center_x
                player1.hand = hand_landmarks

            if rightmost_hand_center_x is None or center_x > rightmost_hand_center_x:
                rightmost_hand_center_x = center_x
                player2.hand = hand_landmarks

        if player1.hand != player2.hand:
            for player in [player1, player2]:
                if player.hand is not None:
                    hand_landmarks = player.hand
                    # Extract the x and y coordinates of the hand landmarks
                    x = [
                        landmark.x * image.shape[1]
                        for landmark in hand_landmarks.landmark
                    ]
                    y = [
                        landmark.y * image.shape[0]
                        for landmark in hand_landmarks.landmark
                    ]

                    # Calculate the bounding box coordinates
                    xmin, xmax = int(min(x)), int(max(x))
                    ymin, ymax = int(min(y)), int(max(y))

                    mp_drawing.draw_landmarks(
                        image,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    # Get position of fingertips
                    index_finger = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    middle_finger = hand_landmarks.landmark[
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ]
                    ring_finger = hand_landmarks.landmark[
                        mp_hands.HandLandmark.RING_FINGER_TIP
                    ]
                    pinky_finger = hand_landmarks.landmark[
                        mp_hands.HandLandmark.PINKY_TIP
                    ]

                    # Get position of base of hand
                    base_of_hand = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Calculate distances between fingers
                    index_middle_distance = np.linalg.norm(
                        np.array([index_finger.x, index_finger.y])
                        - np.array([middle_finger.x, middle_finger.y])
                    )

                    middle_ring_distance = np.linalg.norm(
                        np.array([middle_finger.x, middle_finger.y])
                        - np.array([ring_finger.x, ring_finger.y])
                    )
                    ring_pinky_distance = np.linalg.norm(
                        np.array([ring_finger.x, ring_finger.y])
                        - np.array([pinky_finger.x, pinky_finger.y])
                    )

                    # Calculate distances between fingertips and base of hand
                    index_distance_from_base = np.linalg.norm(
                        np.array([index_finger.x, index_finger.y])
                        - np.array([base_of_hand.x, base_of_hand.y])
                    )

                    pinky_distance_from_base = np.linalg.norm(
                        np.array([pinky_finger.x, pinky_finger.y])
                        - np.array([base_of_hand.x, base_of_hand.y])
                    )

                    play = None

                    # Classify gesture
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
                      print('play', play)

                    # Draw a rectangle around the bounding box coordinates
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), player.color, 2)

        if player1.current_move and player2.current_move:

            cv2.putText(
                image,
                f"player1: {player1.current_move.name}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            text = f"player2: {player2.current_move.name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)
            thickness = 2
        # Get text size
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate x-coordinate for right alignment
            x = image.shape[1] - text_size[0] - 50

            cv2.putText(image, text, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if (player1.points == 0 and player2.points == 0) or (
                player1.current_move != player1.last_move
                or player2.current_move != player2.last_move
            ):
                print("--Jogou--")
                if player1.current_move == player2.current_move:
                    winner = "tie"
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

                text = (
                    f"{winner} wins with {player1.current_move.name} vs. {player2.current_move.name}"
                    if winner != "tie"
                    else "Tie"
                )

                cv2.putText(
                    image,
                    text,
                    (int(image.shape[1] * 0.5), 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

    text = f"{player1.points} x {player2.points}"
    cv2.putText(
        image,
        text,
        (int(image.shape[1] * 0.5), 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(25) & 0xFF == ord("r"):
        break

cap.release()
cv2.destroyAllWindows()