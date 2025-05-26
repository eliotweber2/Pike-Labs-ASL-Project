import numpy as np
from cv2 import waitKey

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

def read_and_process(stream_src, process_fn, stop_key='q', n_skip=0):
    stream = stream_src()
    stream.open()
    while True:
        for _ in range(n_skip):
            stream.read_frame()
        frame = stream.read_frame()
        if frame[1] is None:
            return
        process_fn(frame[1])
        if stop_key and waitKey(1) & 0xFF == ord(stop_key):
            break
    stream.close()