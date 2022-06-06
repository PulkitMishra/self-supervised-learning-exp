import torch 
def mil_nce_loss(audio_video_embed, text_embed):
    x = torch.matmul(audio_video_embed, text_embed.t())
    x = x.view(audio_video_embed.shape[0], audio_video_embed.shape[0], -1)
    numerator = x * torch.eye(x.shape[0])[:,:,None]
    numerator = numerator.sum(dim=1)
    numerator = torch.logsumexp(numerator, dim=1)
    denumerator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
    denumerator = torch.logsumexp(denumerator, dim=1)
    return torch.mean(denumerator - numerator)

def pairwise_triplet_loss(video_embeds, audio_embeds, text_embeds):
    triplet_loss = torch.nn.TripletMarginLoss()
    bs = len(video_embeds)
    loss = 0
    for i in range(bs - 1):
        loss += (triplet_loss(video_embeds[i], audio_embeds[i], audio_embeds[i+1]) +
        triplet_loss(video_embeds[i], text_embeds[i][0], text_embeds[i+1][0]) +
        triplet_loss(audio_embeds[i], text_embeds[i][0], text_embeds[i+1][0]))
    loss += (triplet_loss(video_embeds[bs-1], audio_embeds[bs-1], audio_embeds[0]) +
        triplet_loss(video_embeds[bs-1], text_embeds[bs-1][0], text_embeds[0][0]) +
        triplet_loss(audio_embeds[bs-1], text_embeds[bs-1][0], text_embeds[0][0]))
    return loss/bs
