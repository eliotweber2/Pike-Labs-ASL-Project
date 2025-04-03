import cv2
from video_loader import StreamInterface
if __name__ == '__main__':
    from video_loader import read_and_process

def camera_stream_factory():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    def read(cap):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        return frame
    
    return StreamInterface(lambda: cv2.VideoCapture(0), lambda cap: read(cap), lambda x: print(x), lambda cap: cap.release)

if __name__ == '__main__':

    def process(frame):
        cv2.imshow('frame', frame)
    
    read_and_process(camera_stream_factory, process)
    cv2.destroyAllWindows()