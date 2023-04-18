import argparse
import io
import time

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

import config

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_id", type=str)
    parser.add_argument("--publish_iteration_name", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()
    
    # Set the Custom Vision Project ID
    project_id = args.project_id
    publish_iteration_name = args.publish_iteration_name

    # Authenticate with Azure Custom Vision Client
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": config.PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(config.PREDICTION_ENDPOINT, prediction_credentials)

    # Set the font to use for the annotation
    font = ImageFont.truetype("Arial Narrow Italic.ttf", 32)
    # Configure the colour map from tag to boundary colour
    tag_colour_map = {"bus":"Green","car":"Blue","truck":"Red"}

    KPS = 1 # Target Keyframes Per Second
    VIDEO_PATH = args.video_path #"path/to/video/folder"
    OUTPUT_PATH = args.output_path #"path/to/output/folder"

    # Initialise the OpenCV input and output streamer
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 1, (3840,2160))

    # Get the actual video frames per second
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # Calculate the ratio of FPS to KPS
    hop = round(fps / KPS)
    # Initialise the current frame number
    curr_frame = 0
    print("Reading video file")
    # Create a loop to iterate over frames
    while(True):
        # Get the next frame
        ret, frame = cap.read()
        # If OpenCV can't read the frame (e.g. video ended) then exit out of the loop
        if not ret: break
        # Check if current frame number is a multiple of the hop ratio
        if curr_frame % hop == 0:
            # Get the bytes of the current frame image
            frameBytes = cv2.imencode(".jpg", frame)[1].tobytes()
            # Send image contents to the Custom Vision Service
            results = predictor.detect_image(project_id, publish_iteration_name, frameBytes)
            
            # Create a Pillow Image
            image = Image.open(io.BytesIO(frameBytes))
            # Create a layer to draw on top of the Image
            image_draw = ImageDraw.Draw(image)
            # Loop over each prediction
            for prediction in results.predictions:
                if prediction.probability>0.25:
                    # Draw the bounding box
                    rectangle = [3840*prediction.bounding_box.left,
                                2160*prediction.bounding_box.top,
                                3840*(prediction.bounding_box.left+prediction.bounding_box.width),
                                2160*(prediction.bounding_box.top+prediction.bounding_box.height)]
                    image_draw.rectangle(rectangle, outline=tag_colour_map[prediction.tag_name], width=4)
                    # Draw Text
                    image_draw.text(
                        (3840*prediction.bounding_box.left, 2160*prediction.bounding_box.top-32),
                        f"Name: {prediction.tag_name}, Confidence: {prediction.probability*100:.2f}%",
                        fill="black",
                        font=font
                    )
            # Convert the annotated image back into a format OpenCV can read
            new_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Write the image to the video output
            out.write(new_frame)
            # Wait for 1 second to avoid excessive calls to the prediction API
            print("Processing...")
            time.sleep(1)
        # Increment the current frame number
        curr_frame += 1

    # After the video has ended, close the OpenCV connection
    cap.release()
    out.release()
    print("Object Detection Successful!")

if __name__ == '__main__':
    main()
