import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat


class SumMeDataset(Dataset):
    def __init__(self, videos_dir, annotations_dir, num_frames=30, transform=None,
                return_all_annotators=False):
        """
        Custom Dataset Loader for SumMe (Official Raw Videos + .mat Annotations)

        return_all_annotators: إن كانت True، يُرجِع __getitem__ أيضاً مصفوفة
            user_score الأصلية (كل مُقيِّم على حدة، ثنائية 0/1) بعد أخذ عينات
            بنفس فهارس الإطارات المُستخرَجة، اللازمة لبروتوكول F1 القياسي
            (أقصى F1 عبر المُقيِّمين). القيمة الافتراضية False تُبقي التوافق
            مع أي كود تدريب حالي يعتمد فقط على (video, scores).
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.num_frames = num_frames
        self.transform = transform
        self.return_all_annotators = return_all_annotators

        self.video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_filename = self.video_files[idx]
        video_name = os.path.splitext(video_filename)[0]

        video_path = os.path.join(self.videos_dir, video_filename)
        annotation_path = os.path.join(self.annotations_dir, f"{video_name}.mat")

        user_score_full = None
        try:
            mat_data = loadmat(annotation_path)
            gt_scores = mat_data['gt_score'].squeeze()
            if 'user_score' in mat_data:
                # (nFrames, nUsers) ثنائية — ملخّص كل مُقيِّم بشري على حدة
                user_score_full = mat_data['user_score'].astype(np.float32)
        except Exception as e:
            print(f"[!] Error loading annotation for {video_name}: {e}")
            gt_scores = np.zeros(self.num_frames)

        total_original_frames = len(gt_scores)

        cap = cv2.VideoCapture(video_path)
        frames = []

        if total_original_frames > 0:
            indices = np.linspace(0, total_original_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(self.num_frames)

        for i in range(self.num_frames):
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
        video_tensor = torch.tensor(frames, dtype=torch.float32)

        if total_original_frames > 0:
            sampled_scores = gt_scores[indices]
        else:
            sampled_scores = np.zeros(self.num_frames)

        max_score = sampled_scores.max()
        if max_score > 0:
            sampled_scores = sampled_scores / max_score

        scores_tensor = torch.tensor(sampled_scores, dtype=torch.float32)

        if not self.return_all_annotators:
            return video_tensor, scores_tensor

        if user_score_full is not None:
            # (nFrames, nUsers) -> أخذ عينات بنفس الفهارس -> (num_frames, nUsers) -> transpose -> (nUsers, num_frames)
            sampled_users = user_score_full[indices, :]
            user_tensor = torch.tensor(sampled_users.T, dtype=torch.float32)
        else:
            # احتياط إن غاب الحقل: مُقيِّم واحد فقط مبني من gt_score كبديل آمن
            user_tensor = scores_tensor.unsqueeze(0)

        return video_tensor, scores_tensor, user_tensor

    def get_fine_annotation(self, idx, fine_frames=150):
        """
        يُعيد تسلسلاً أدق زمنياً (fine_frames نقطة، افتراضياً 150) خاصاً حصراً
        بتجزئة KTS وحساب F1 على مستوى اللقطة — منفصل تماماً عن num_frames
        التي يتوقّعها النموذج ودُرِّب عليها (والتي يجب ألا تتغيّر أبداً).
        يُعيد أيضاً ملخّص كل مُقيِّم (user_score) عند هذه الدقة الأدق، بدل
        ضغطه إلى num_frames كما يفعل __getitem__ العادي.

        لا علاقة لهذه الدالة بمدخل النموذج إطلاقاً — تُستخدَم فقط لبناء
        اللقطات ومطابقتها مع درجات النموذج بعد رفعها (interpolation) لنفس
        هذه الدقة، تماماً كما يصف القسم V-A2 في الورقة (150 نقطة تقييم).
        """
        video_filename = self.video_files[idx]
        video_name = os.path.splitext(video_filename)[0]
        video_path = os.path.join(self.videos_dir, video_filename)
        annotation_path = os.path.join(self.annotations_dir, f"{video_name}.mat")

        mat_data = loadmat(annotation_path)
        gt_scores = mat_data['gt_score'].squeeze()
        user_score_full = mat_data['user_score'].astype(np.float32) if 'user_score' in mat_data else None
        total_original_frames = len(gt_scores)

        if total_original_frames > 0:
            indices = np.linspace(0, total_original_frames - 1, fine_frames, dtype=int)
        else:
            indices = np.arange(fine_frames)

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

        if user_score_full is not None:
            sampled_users = user_score_full[indices, :]
            user_tensor = torch.tensor(sampled_users.T, dtype=torch.float32)
        else:
            sampled_gt = gt_scores[indices] if total_original_frames > 0 else np.zeros(fine_frames)
            if sampled_gt.max() > 0:
                sampled_gt = sampled_gt / sampled_gt.max()
            user_tensor = torch.tensor(sampled_gt, dtype=torch.float32).unsqueeze(0)

        return content_tensor, user_tensor


if __name__ == "__main__":
    _root = os.environ.get('VS_DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    test_videos = os.path.join(_root, 'SumMe', 'videos')
    test_gt = os.path.join(_root, 'SumMe', 'GT')

    if os.path.exists(test_videos) and os.path.exists(test_gt):
        dataset = SumMeDataset(videos_dir=test_videos, annotations_dir=test_gt, num_frames=30,
                               return_all_annotators=True)
        print(f"[*] SumMe Dataset loaded successfully! Found {len(dataset)} videos.")
        vid, score, users = dataset[0]
        print(f"Video Shape: {vid.shape} | Score Shape: {score.shape} | Users Shape: {users.shape}")
    else:
        print("[!] Dataset folders not found. Please download and extract them first.")
