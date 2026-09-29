import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset import VideoProcessor


class TVSumDataset(Dataset):
    def __init__(self, videos_dir, annotations_file, num_frames=30, return_all_annotators=False):
        """
        return_all_annotators: يُرجِع أيضاً درجات كل مُقيِّم من الـ20 على حدة
        (مستمرة، غير مُتوسَّطة)، بنفس فهارس الإطارات المُستخرَجة.
        """
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.return_all_annotators = return_all_annotators
        self.processor = VideoProcessor(num_frames=num_frames)

        self.annotations = pd.read_csv(annotations_file, sep='\t', header=None,
                                       names=['video_id', 'category', 'scores'])
        self.grouped_annotations = self.annotations.groupby('video_id')['scores'].apply(list).reset_index()

    def __len__(self):
        return len(self.grouped_annotations)

    def __getitem__(self, idx):
        row = self.grouped_annotations.iloc[idx]
        video_id = row['video_id']
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")

        video_tensor = self.processor.extract_frames(video_path)

        all_annotator_scores = []
        for score_str in row['scores']:
            scores = np.array([float(x) for x in score_str.split(',')])
            all_annotator_scores.append(scores)

        total_original_frames = len(all_annotator_scores[0])
        indices = np.linspace(0, total_original_frames - 1, self.num_frames, dtype=int)

        avg_scores = np.mean(all_annotator_scores, axis=0)
        sampled_avg = avg_scores[indices]
        avg_min, avg_max = sampled_avg.min(), sampled_avg.max()
        sampled_avg = (sampled_avg - avg_min) / (avg_max - avg_min + 1e-8)
        scores_tensor = torch.tensor(sampled_avg, dtype=torch.float32)

        if not self.return_all_annotators:
            return video_tensor, scores_tensor

        per_annotator_sampled = np.stack(
            [ann[indices] for ann in all_annotator_scores], axis=0
        )
        annotator_tensor = torch.tensor(per_annotator_sampled, dtype=torch.float32)

        return video_tensor, scores_tensor, annotator_tensor

    def get_fine_annotation(self, idx, fine_frames=150):
        """
        دقة منفصلة (افتراضياً 150) خاصة حصراً بـ KTS وحساب F1 — منفصلة تماماً
        عن num_frames التي يتوقّعها النموذج (30)، تماماً كما في مشروع SumMe.
        """
        row = self.grouped_annotations.iloc[idx]
        video_id = row['video_id']
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")

        all_annotator_scores = []
        for score_str in row['scores']:
            scores = np.array([float(x) for x in score_str.split(',')])
            all_annotator_scores.append(scores)

        total_original_frames = len(all_annotator_scores[0])
        indices = np.linspace(0, total_original_frames - 1, fine_frames, dtype=int)

        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(fine_frames):
            frame_idx = indices[i] if i < len(indices) else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()

        frames = np.array(frames, dtype=np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))
        content_tensor = torch.tensor(frames, dtype=torch.float32)

        per_annotator_fine = np.stack(
            [ann[indices] for ann in all_annotator_scores], axis=0
        )
        annotator_tensor = torch.tensor(per_annotator_fine, dtype=torch.float32)

        return content_tensor, annotator_tensor
