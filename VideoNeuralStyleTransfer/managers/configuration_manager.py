'''
Manages and organizes the configuration settings for the program

:author: Matthew Schofield
'''
import os
import torch
import configparser

class Configuration:

    def __init__(self, configuration_file_path, absolute_path):
        '''
        Initialize with path to the configuration file and absolute path being excuted within

        :param configuration_file_path: relative path to configuration file
        :param absolute_path:  path script is executing within
        '''
        # Boolean flag as to whether the configuration has been initialized
        self.ready = False

        # If a configuration file has been specified use that
        if configuration_file_path != "":
            # Read in configuration
            parser = configparser.ConfigParser()
            parser.read(configuration_file_path)
            configuration_file_settings = {section: dict(parser.items(section)) for section in parser.sections()}["SETTINGS"]

            # Parse resolution string to tuple
            resolution_str = configuration_file_settings["resolution"][1:-1].split(",")
            resolution = (int(resolution_str[0]), int(resolution_str[1]))

            style_transfer_models = configuration_file_settings["style_transfer_models"][1:-1].split(",")
            if len(style_transfer_models) == 1 and style_transfer_models[0] == '':
                style_transfer_models = []

            self.set_configuration(
                absolute_path,
                configuration_file_settings["input_video"],
                style_transfer_models,
                configuration_file_settings["segmenter_model"],
                resolution,
                bool(configuration_file_settings["use_cuda"])
            )


    def set_configuration(self, absolute_path, input_video, style_transfer_models, segmenter_model, resolution, use_cuda):
        '''
        Initialize configurations

        :param absolute_path: current execution directory path
        :param input_video: relative path to input video
        :param style_transfer_models: list of paths to style transfer model weights
        :param segmenter_model: relative path to segmenter model
        :param resolution: tuple for resolution of video
        :param use_cuda: whether to utilize cuda speed-ups (if gpu is present)
        '''
        # Path to input video
        self.input_video_path = os.path.join(absolute_path, "raw_data/video_style_transfer/base_videos", input_video)

        # Path to output video
        output_video_path = os.path.join(absolute_path, "outputs", "output_" + input_video)
        # Replace output video path extension
        self.output_video_path = os.path.splitext(output_video_path)[0] + ".avi"

        # Path to style transfer model
        self.style_transfer_weights = []
        for style_transfer_model in style_transfer_models:
            self.style_transfer_weights.append(
                os.path.join(absolute_path, "models/style_transfer", style_transfer_model.strip())
            )

        # Path to instance segmenter model
        self.instance_segmenter_path = os.path.join(absolute_path, "models/instance_segmentation/yolact_models", segmenter_model)

        # Set GPU utilization
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                print("Could not find CUDA device, defaulting to CPU")
        else:
            self.device = torch.device("cpu")

        # Redefine resolution, leave room for editing in the future
        self.resolution = resolution
        self.ready = True

    '''
    GETTERS for configuration settings
    '''
    def get_input_video_path(self):
        return self.input_video_path

    def get_output_video_path(self):
        return self.output_video_path

    def get_style_transfer_weights(self):
        return self.style_transfer_weights

    def get_instance_segmenter_path(self):
        return self.instance_segmenter_path

    def get_resolution(self):
        return self.resolution

    def get_rev_resolution(self):
        return (self.resolution[1], self.resolution[0])

    def get_device(self):
        return self.device

    def is_ready(self):
        return self.ready