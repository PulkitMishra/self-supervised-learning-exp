import torch
import itertools
import re
import numpy as np

class LimitDataset(torch.utils.data.Dataset):

    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def get_data_from_csv(csv_path, data_split):
    video_label_paths = []
    with open(csv_path, 'r') as f:
        for path_label in f.read().splitlines():
            path, label = path_label.split(",", 1)
            video_label_paths.append((path, {"text":label}))
    split_index = int(data_split*len(video_label_paths))
    return video_label_paths[:split_index], video_label_paths[split_index:]

class TextPreprocess(torch.nn.Module):
    """
    Transform for converting raw text to word embeddings.
    """
    def __init__(self):
        super().__init__()
        self.max_words = 16
        self.word_to_token = {}
        token_to_word = np.load('data/dict.npy')
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.word_embd = torch.nn.Embedding.from_pretrained(torch.load('data/word2vec.pth')) 

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def _split_text_to_lowercase(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        w = [word.lower() for word in w]
        #print("Sentences broken up as : ", w)
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        #print("Above words tokenized : ", words)
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            #print("Words as Padded Tensors : ", we)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text_to_lowercase(sent)) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, sentence: torch.Tensor):
        #sentence = ["Hello how are YOU!! ", 'I am fine Thanks fOR asking.']
        sentence = [sentence] #get_data_from_csv assumes only one positive sentence per video but TextPreprocessing is coded to work with mulltiple positives 
        #print("Sentences : ", sentence)
        x = self._words_to_ids(sentence)
        #print("Stacked Tensor : ", x)
        x = self.word_embd(x)
        #print("Sentences as Embeddings : ", x)
        return x
