import cv2
import mediapipe as mp
import time
import os
import pyautogui

class SimpleGestureControl:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.finger_colors = {
            'thumb': (255, 0, 0),
            'index': (0, 255, 0),
            'middle': (0, 0, 255),
            'ring': (255, 255, 0),
            'pinky': (255, 0, 255),
            'palm': (128, 128, 128)
        }

        self.drawing_specs = {
            finger: self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            for finger, color in self.finger_colors.items()
        }

        self.prev_gesture = None
        self.gesture_start_time = None
        self.last_action_time = None
        self.confirmation_time = 1.0
        self.repeat_interval = 2.0

        self.feedback_message = ""
        self.feedback_time = None
        self.feedback_duration = 2.0

        self.actions = {
            "open_palm": self.panic_button,
            "fist": self.lock_system,
            "point": self.open_youtube,
            "peace": self.open_calculator,
            "yoyo": self.open_workspace,
            "thumbs_up": self.volume_up,
            "thumbs_down": self.volume_down,
            "three": self.take_screenshot,
            "four": self.next_song,
            "call": self.pause_song
        }


    def detect_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        fingertips_up = [
            index_tip.y < index_mcp.y,
            middle_tip.y < middle_mcp.y,
            ring_tip.y < ring_mcp.y,
            pinky_tip.y < pinky_mcp.y
        ]
        thumb_out = thumb_tip.x < thumb_mcp.x

        if all(fingertips_up) and thumb_out:
            return "open_palm"
        if not any(fingertips_up) and not thumb_out:
            return "fist"
        if fingertips_up[0] and not any(fingertips_up[1:]) and not thumb_out:
            return "point"
        if fingertips_up[:1] and not any(fingertips_up[2:]) and not thumb_out:
            return "peace"
        if all(fingertips_up[:3]) and not fingertips_up[3] and not thumb_out:
            return "naamam"
        if all(fingertips_up[1:]) and not fingertips_up[0] and not thumb_out:
            return "three"
        if fingertips_up[0] and fingertips_up[3] and not any(fingertips_up[1:2]) and not thumb_out:
            return "yoyo"
        if fingertips_up[3] and thumb_out and not any(fingertips_up[0:3]) :
            return "call"
        if all(fingertips_up) and not thumb_out:
            return "four"
        if thumb_out and thumb_tip.y < wrist.y and not any(fingertips_up):
            return "thumbs_up"
        if thumb_out and thumb_tip.y > wrist.y and not any(fingertips_up):
            return "thumbs_down"

        return "unknown"

    def nothing(self):
        return "No action assigned"

    def panic_button(self):
        try:
            os.startfile("panic_button.ahk")
            return "Panic Button Triggered"
        except FileNotFoundError:
            return "panic_button.ahk not found"

    def lock_system(self):
        try:
            os.startfile("lock_system.ahk")
            return "Screen Locked"
        except FileNotFoundError:
            return "lock_system.ahk not found"

    def volume_up(self):
        pyautogui.press("volumeup", presses=3)
        return "Volume Up"

    def volume_down(self):
        pyautogui.press("volumedown", presses=3)
        return "Volume Down"

    def take_screenshot(self):
        path = f"screenshot_{int(time.time())}.png"
        pyautogui.screenshot(path)
        return f"Screenshot saved as {path}"

    def open_youtube(self):
        try:
            os.startfile("open_youtube.ahk")
            return "Opened YouTube"
        except FileNotFoundError:
            return "open_youtube.ahk not found"

    def open_calculator(self):
        os.system("calc")
        return "Opened Calculator"

    def open_workspace(self):
        try:
            os.startfile("open_workspace.ahk")
            return "Opened Workspace"
        except FileNotFoundError:
            return "open_workspace.ahk not found"
        
    def next_song(self):
        pyautogui.press("nexttrack")
        return "Next Song"

    def pause_song(self):
        pyautogui.press("playpause")
        return "Play/Pause Toggled"



    def draw_finger(self, frame, hand_landmarks, finger_points, color):
        height, width = frame.shape[:2]
        prev_point = None
        for landmark in finger_points:
            point = hand_landmarks.landmark[landmark]
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), 2, color, -1)
            if prev_point:
                px, py = int(prev_point.x * width), int(prev_point.y * height)
                cv2.line(frame, (px, py), (x, y), color, 2)
            prev_point = point

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        current_time = time.time()
        height, width = frame.shape[:2]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_specs['palm'],
                    self.drawing_specs['palm']
                )

                self.draw_finger(frame, hand_landmarks, [
                    self.mp_hands.HandLandmark.THUMB_CMC,
                    self.mp_hands.HandLandmark.THUMB_MCP,
                    self.mp_hands.HandLandmark.THUMB_IP,
                    self.mp_hands.HandLandmark.THUMB_TIP
                ], self.finger_colors['thumb'])

                self.draw_finger(frame, hand_landmarks, [
                    self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ], self.finger_colors['index'])

                self.draw_finger(frame, hand_landmarks, [
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                ], self.finger_colors['middle'])

                self.draw_finger(frame, hand_landmarks, [
                    self.mp_hands.HandLandmark.RING_FINGER_MCP,
                    self.mp_hands.HandLandmark.RING_FINGER_PIP,
                    self.mp_hands.HandLandmark.RING_FINGER_DIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP
                ], self.finger_colors['ring'])

                self.draw_finger(frame, hand_landmarks, [
                    self.mp_hands.HandLandmark.PINKY_MCP,
                    self.mp_hands.HandLandmark.PINKY_PIP,
                    self.mp_hands.HandLandmark.PINKY_DIP,
                    self.mp_hands.HandLandmark.PINKY_TIP
                ], self.finger_colors['pinky'])

                current_gesture = self.detect_gesture(hand_landmarks)
                cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if current_gesture == "unknown":
                    self.prev_gesture = None
                    self.gesture_start_time = None
                    self.last_action_time = None
                elif current_gesture == self.prev_gesture:
                    if (self.gesture_start_time and self.last_action_time is None and
                        current_time - self.gesture_start_time > self.confirmation_time):
                        if current_gesture in self.actions:
                            self.feedback_message = self.actions[current_gesture]()
                            self.feedback_time = current_time
                            self.last_action_time = current_time
                    elif (self.last_action_time and
                          current_time - self.last_action_time > self.repeat_interval):
                        if current_gesture in self.actions:
                            self.feedback_message = self.actions[current_gesture]()
                            self.feedback_time = current_time
                            self.last_action_time = current_time
                else:
                    self.prev_gesture = current_gesture
                    self.gesture_start_time = current_time
                    self.last_action_time = None

        if self.feedback_time and current_time - self.feedback_time < self.feedback_duration:
            cv2.putText(
                frame,
                self.feedback_message,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame

    def run(self):
        print("Starting Simple Gesture Control")
        print("Available gestures:")
        for gesture in self.actions:
            print(f"- {gesture}: {self.actions[gesture].__name__}")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)
            cv2.imshow("Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    control = SimpleGestureControl()
    control.run()
