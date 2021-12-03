'''
Static wrapper class to generate video readers / writer from opencv

:author: Matthew Schofield
'''
import cv2

class VideoManager(object):

    @staticmethod
    def createVideoReader(configuration):
        # Open an opencv2 video reader to read the input video frame by frame
        input_video_reader = cv2.VideoCapture(configuration.get_input_video_path())

        # If video file is not found
        if not input_video_reader.isOpened():
            print("Cannot find video: " + configuration.get_input_video_path())
            exit(1)

        return input_video_reader

    @staticmethod
    def createVideoWriter(configuration, fps):
        # Init opencv2 video writer to output styled frames
        return cv2.VideoWriter(
            configuration.get_output_video_path(),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            fps,
            configuration.get_rev_resolution(),
            isColor=True
        )

