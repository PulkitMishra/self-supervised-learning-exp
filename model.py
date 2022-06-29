import torch
import pytorch_lightning as pl
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.resnet import create_acoustic_resnet
from pytorchvideo.models.head import create_res_basic_head
from loss import mil_nce_loss, pairwise_triplet_loss

class BaseModel(pl.LightningModule):
    def __init__(self, video_net_name, audio_net_name, text_net_name, lr_v, lr_a, lr_t, batch_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = pairwise_triplet_loss
        self.video_encoder = encode_video(video_net_name)
        self.audio_encoder = encode_audio(audio_net_name)
        self.text_encoder = encode_text(text_net_name)
        self.video_projection = project_video()
        self.audio_projection = project_audio()
        self.text_projection = project_text()
        #self.automatic_optimization = False

    def configure_optimizers(self):
        optim_video = torch.optim.Adam(self.video_encoder.parameters(), self.hparams.lr_v)
        optim_audio = torch.optim.Adam(self.audio_encoder.parameters(), self.hparams.lr_a)
        optim_text = torch.optim.Adam(self.text_encoder.parameters(), self.hparams.lr_t)
        return optim_video, optim_audio, optim_text
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        videos, audios, texts = batch["video"], batch["audio"], batch["text"]
        video_embeds = self.video_projection(self.video_encoder(videos))
        audio_embeds = self.audio_projection(self.audio_encoder(audios))
        text_embeds = self.text_projection(torch.max(self.text_encoder(texts), dim=2)[0])
        loss = self.loss(video_embeds, audio_embeds, text_embeds)
        self.log('train_loss', loss, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, audios, texts = batch["video"], batch["audio"], batch["text"]
        video_embeds = self.video_projection(self.video_encoder(videos))
        audio_embeds = self.audio_projection(self.audio_encoder(audios))
        text_embeds = self.text_projection(torch.max(self.text_encoder(texts), dim=2)[0])
        loss = self.loss(video_embeds, audio_embeds, text_embeds)
        self.log('val_loss', loss, batch_size=self.hparams.batch_size)
        return loss
    

def encode_video(net_name):
    if net_name == 'slowfast_resnet50':
        encoder = create_slowfast()
        encoder.blocks[6] = torch.nn.Identity()
        # encoder.blocks[6].dropout = torch.nn.Identity()
        # encoder.blocks[6].proj = torch.nn.Identity()
        # encoder.blocks[6].output_pool = torch.nn.Identity()
    return encoder

def encode_audio(net_name):
    if net_name == 'acoustic_resnet50':
        encoder = create_acoustic_resnet()
        encoder.blocks[5] = torch.nn.Identity()
        # encoder.blocks[5].dropout = torch.nn.Identity()
        # encoder.blocks[5].proj = torch.nn.Identity()
        # encoder.blocks[5].output_pool = torch.nn.Identity()
    return encoder

def encode_text(net_name):
    if net_name == 'default':
        layers = []
        layers.append(torch.nn.Linear(300, 2048))
        layers.append(torch.nn.ReLU())
        encoder = torch.nn.Sequential(*layers)
    return encoder

def project_video():
    return create_res_basic_head(in_features=2304, out_features=512, pool=None)

def project_audio():
    return create_res_basic_head(in_features=2048, out_features=512, pool=torch.nn.AvgPool3d, pool_kernel_size=(4, 1, 2), pool_stride=(1, 1, 1), pool_padding=(0, 0, 0), dropout_rate=0.5)

def project_text():
    return torch.nn.Linear(2048, 512)
