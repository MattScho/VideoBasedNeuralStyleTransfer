'''
Main application to style a video using style transfer and instance segmentation

:author: Matthew Schofield
'''
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from torchvision import transforms
from style_transfer.transformer_net import TransformerNet
from instance_segmentation.image_segmenter import ImageSegmenter
from managers.video.video_manager import VideoManager
import time

def main(configuration):
    """
    Main function to control the program's execution given a configuration

    :param configuration: configuration_manager object to manage and map config values
    :return: no return value, though an output video will be found in the outputs directory after successful execution
    """

    '''
    Set up video managers
    '''
    # Init reader to process video frames
    input_video_reader = VideoManager.createVideoReader(configuration)

    # Get frames per second, to avoid choppiness and apply to output
    fps = input_video_reader.get(cv2.CAP_PROP_FPS)

    # Init writer to write created frames to output video file
    output_video_writer = VideoManager.createVideoWriter(configuration, fps)

    # Init style models
    style_models = []
    for weight_path in configuration.get_style_transfer_weights():
        style_models.append(
            TransformerNet(weight_path, configuration.get_device(), True)
        )

    # Init instance segementer model
    image_segmenter = ImageSegmenter(configuration.get_instance_segmenter_path())

    '''
    Process video
    '''
    # Frame counter to track progress
    frame_num = 0
    # Init for timing
    start_time = time.time()

    # Step through video
    while input_video_reader.isOpened():
        # Print frame number, to track progress / speed
        print(frame_num)
        frame_num += 1

        # Read next frame
        is_reading, original_frame = input_video_reader.read()
        if not is_reading:
            break

        '''
        Configure original frame
        '''
        original_frame_pil = Image.fromarray(original_frame)
        original_frame_pil = original_frame_pil.resize(configuration.get_resolution())
        original_frame_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        original_frame_tensor = original_frame_tensor_transform(original_frame_pil)
        original_frame_tensor = original_frame_tensor.unsqueeze(0).to(configuration.get_device())
        original_frame = cv2.resize(original_frame, configuration.get_rev_resolution())

        '''
        Style frames
        '''
        styled_frames = []
        for style_model in style_models:
            # Get styled frame and format
            styled = style_model(original_frame_tensor).cpu()[0]

            styled = styled.clone().clamp(0, 255).detach().numpy()
            styled = styled.transpose(1, 2, 0).astype("uint8")
            styled_resized = cv2.resize(styled, configuration.get_rev_resolution())
            # Dilute the styled frame with the original frame 50/50
            styled_frames.append(cv2.addWeighted(original_frame, 0.5, styled_resized, 0.5, 0))


        # Use the first style frame as the background style
        background_style = cv2.addWeighted(original_frame, 0.5, styled_frames.pop(0), 0.5, 0)

        '''
        Create masks
        '''
        # Uses an instance segmentation model to return a list of masks
        masks = image_segmenter.segment(original_frame)

        # Init a background mask of 0's to add to as we receive foreground masks and then later invert
        background_mask = tf.dtypes.cast(np.zeros((configuration.get_resolution()[0], configuration.get_resolution()[1], 3)), tf.float32)

        # Foreground output to attach images to as styles and masks come in
        fg_output = tf.dtypes.cast(np.zeros((configuration.get_resolution()[0], configuration.get_resolution()[1], 3)), tf.float32)

        # Iterate through masks and apply styles to instances
        iteration = 0
        for mask in masks:
            # Add the styled instance to the foreground output
            inv_mask = tf.add(mask, -1)
            inv_mask = tf.multiply(inv_mask, inv_mask)
            fg_output = tf.multiply(inv_mask, fg_output)
            fg_output = tf.add(tf.multiply(mask, styled_frames[iteration]), fg_output)
            # Add the current foreground mask to the background mask accumulator, this will later be inverted

            background_mask = tf.multiply(inv_mask, background_mask)
            background_mask = tf.add(background_mask, mask)
            # Step to use the next style type for the following frame, or reset to 1st style frame
            iteration += 1
            iteration %= len(styled_frames)

        # Invert current 'background mask' to make it the actual background mask
        background_mask = tf.add(background_mask, -1)
        background_mask = tf.multiply(background_mask, background_mask)

        # Create the background by applying the background mask to the background style
        bg_output = tf.multiply(background_mask, background_style)

        # Merge the background and foreground frames into the final output
        output_frame = np.uint8(tf.add(bg_output, fg_output))

        # Display frame and break if user hits q
        output_video_writer.write(output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup file input and output streams and shutdown open cv cleanly
    # Without this extra file handles may be left open
    input_video_reader.release()
    output_video_writer.release()

    cv2.destroyAllWindows()
    print("Completed " + str(frame_num) +  " frames in " + str(int(time.time()-start_time)) + " seconds")
