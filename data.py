import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from pytorchvideo.data import make_clip_sampler, LabeledVideoDataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
)
from torchaudio.transforms import MelSpectrogram, Resample

from utils import LimitDataset, get_data_from_csv, PackPathway, TextPreprocess

class DataModule(pl.LightningDataModule):
    def __init__(self, csv_path, data_split, batch_size, clip_duration, decode_audio):
        super().__init__()
        self.batch_size = batch_size
        self.do_use_ddp = do_use_ddp
        self.clip_duration = clip_duration
        self.decode_audio = decode_audio
        self.train_videos_labels, self.val_videos_labels = get_data_from_csv(csv_path, data_split)
        self.frames_to_sample = 32
        self.video_means = (0.45, 0.45, 0.45)
        self.video_stds = (0.225, 0.225, 0.225)
        self.audio_raw_sample_rate = 48000
        self.audio_resampled_rate = 16000
        self.audio_mel_window_size = 32
        self.audio_mel_step_size = 16
        self.audio_num_mels = 80
        self.audio_mel_num_subsample = 256
        self.audio_logmel_mean = -7.03
        self.audio_logmel_std = 4.66
        self.eps = 1e-10
        self.n_fft = int(
            float(self.audio_resampled_rate) / 1000 * self.audio_mel_window_size
        )
        self.hop_length = int(
            float(self.audio_resampled_rate) / 1000 * self.audio_mel_step_size
        )
        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.video_means, self.video_stds),
                            PackPathway(),
                        ]
                    ),
                ),
                ApplyTransformToKey(
                    key="audio",
                    transform=Compose(
                        [
                            Resample(
                                orig_freq=self.audio_raw_sample_rate,
                                new_freq=self.audio_resampled_rate,
                            ),
                            MelSpectrogram(
                                sample_rate=self.audio_resampled_rate,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                n_mels=self.audio_num_mels,
                                center=False,
                            ),
                            Lambda(lambda x: x.clamp(min=self.eps)),
                            Lambda(torch.log),
                            UniformTemporalSubsample(self.audio_mel_num_subsample, 1),
                            Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                            Lambda(
                                lambda x: x.view(1, x.size(0), 1, x.size(1))
                            ),  # (T, F) -> (1, T, 1, F)
                            Normalize((self.audio_logmel_mean,), (self.audio_logmel_std,)),
                        ]
                    ),
                ),
                ApplyTransformToKey(
                    key="text",
                    transform=Compose(
                        [
                            TextPreprocess()
                        ]
                    ),
                )
            ]
        )
        self.val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.video_means, self.video_stds),
                            PackPathway(),
                        ]
                    ),
                ),
                ApplyTransformToKey(
                    key="audio",
                    transform=Compose(
                        [
                            Resample(
                                orig_freq=self.audio_raw_sample_rate,
                                new_freq=self.audio_resampled_rate,
                            ),
                            MelSpectrogram(
                                sample_rate=self.audio_resampled_rate,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                n_mels=self.audio_num_mels,
                                center=False,
                            ),
                            Lambda(lambda x: x.clamp(min=self.eps)),
                            Lambda(torch.log),
                            UniformTemporalSubsample(self.audio_mel_num_subsample, 1),
                            Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                            Lambda(
                                lambda x: x.view(1, x.size(0), 1, x.size(1))
                            ),  # (T, F) -> (1, T, 1, F)
                            Normalize((self.audio_logmel_mean,), (self.audio_logmel_std,)),
                        ]
                    ),
                ),
                ApplyTransformToKey(
                    key="text",
                    transform=Compose(
                        [
                            TextPreprocess()
                        ]
                    ),
                )
            ]
        )

    def train_dataloader(self):
        self.train_dataset = LimitDataset(
            LabeledVideoDataset(
                labeled_video_paths=self.train_videos_labels,
                clip_sampler=make_clip_sampler("random", self.clip_duration),
                video_sampler=RandomSampler,
                transform=self.train_transform,
                decode_audio=self.decode_audio,
            )
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        self.val_dataset = LimitDataset(
            LabeledVideoDataset(
                labeled_video_paths=self.val_videos_labels,
                clip_sampler=make_clip_sampler("uniform", self.clip_duration),
                video_sampler=RandomSampler,
                transform=self.val_transform,
                decode_audio=self.decode_audio,
            )
        )
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)


