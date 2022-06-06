import pytorch_lightning as pl
from data import DataModule
from model import BaseModel
from argparse import ArgumentParser

def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--data_split", default=0.9, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--clip_duration", default=3.2, type=float)

    parser.add_argument("--learning_rate_video", default='3e-4', type=float)
    parser.add_argument("--learning_rate_audio", default='3e-4', type=float)
    parser.add_argument("--learning_rate_text", default='3e-4', type=float)
    parser.add_argument("--video_model", default='slowfast_resnet50', type=str)
    parser.add_argument("--audio_model", default='acoustic_resnet50', type=str)
    parser.add_argument("--text_model", default='default', type=str)

    return parser.parse_args(args)

def train(args):
    pl.seed_everything(224)
    dm = DataModule(csv_path=args.data_path, data_split=args.data_split, batch_size=args.batch_size, clip_duration=args.clip_duration, decode_audio=True)
    model = BaseModel(video_net_name=args.video_model, audio_net_name=args.audio_model, text_net_name=args.text_model, lr_v=args.learning_rate_video, lr_a=args.learning_rate_audio, lr_t=args.learning_rate_text, batch_size=args.batch_size)
    callbacks = [pl.callbacks.LearningRateMonitor(), pl.callbacks.ModelCheckpoint(monitor="val_loss")]
    trainer = pl.Trainer(callbacks=callbacks, accelerator="gpu", devices=4, strategy="ddp", max_epochs=-1, replace_sampler_ddp=False)
    trainer.fit(model, dm)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
