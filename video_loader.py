import cv2

class StreamInterface:
    def __init__(self,open,read_frame,error,close):
        self.open_fn = open
        self.read_frame_fn = read_frame
        self.error_fn = error
        self.close_fn = close

    def open(self):
        self.video_input = self.open_fn()

    def read_frame(self):
        return self.read_frame_fn(self.video_input)
    
    def error(self, error):
        self.error_fn(error)
    
    def close(self):
        self.close_fn(self.video_input)

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

def read_and_process(stream_src, process_fn, stop_key='q'):
    stream = stream_src()
    stream.open()
    while True:
        frame = stream.read_frame()
        process_fn(frame)

        if stop_key and cv2.waitKey(1) & 0xFF == ord(stop_key):
            break
    stream.close()

if __name__ == '__main__':
    def process(frame):
        cv2.imshow('frame', frame)
        
    read_and_process(camera_stream_factory, process)
    cv2.destroyAllWindows()
