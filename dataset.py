import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os


class VideoProcessor:
    def __init__(self, num_frames=30, image_size=224):
        """
        num_frames: The number of frames to extract from each video to represent the clip.
        image_size: The target dimensions (224x224 is the standard for hybrid models).
        """
        self.num_frames = num_frames

        # Standard transformations expected by pre-trained ImageNet models
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path):
        """
        Opens the video, samples frames at equal intervals, and prepares them as a Tensor.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at path: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("The video is empty or cannot be read.")

        # Calculate uniform intervals for frame sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            # Jump to the specific frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # OpenCV reads in BGR format, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image to apply torchvision transforms
                pil_img = Image.fromarray(frame)
                tensor_img = self.transform(pil_img)
                frames.append(tensor_img)
            else:
                # If reading fails, duplicate the last successful frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    # Fallback to a zero tensor if the first frame fails
                    frames.append(torch.zeros(3, 224, 224))

        cap.release()

        # Stack the list of frames into a single tensor
        # Resulting shape: (Frames, Channels, Height, Width)
        video_tensor = torch.stack(frames)

        return video_tensor


# ==========================================
# Testing the Processor
# ==========================================
if __name__ == "__main__":
    # Place a sample MP4 video in your directory or specify the path here
    # video_path = "sample_video.mp4"

    processor = VideoProcessor(num_frames=30)
    print("VideoProcessor initialized successfully.")

    # Uncomment the lines below to test when a video is available:
    # try:
    #     video_data = processor.extract_frames(video_path)
    #     print(f"Video processed successfully! Final shape: {video_data.shape}")
    #     # Expected shape: [30, 3, 224, 224]
    # except Exception as e:
    #     print(f"An error occurred: {e}")