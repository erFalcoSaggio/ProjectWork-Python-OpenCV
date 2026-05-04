# Falcari
import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read() # prendo il frame
        if not ret:
            break
        cv2.imshow("Camera - premi Q per uscire", frame)
        # per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
