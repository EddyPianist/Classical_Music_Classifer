from datasets import load_dataset
from transformers import ClapModel, ClapProcessor
from dataclasses import dataclass

from CLAP import build_audio_encoder, build_mlp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

import os
import time
from tqdm import tqdm


class Config:
    def __init__(self, d_model, in_channels, patchsize, layer_list, down_rate, spatial_resolution, window_size, freq_ratio, num_heads, drop_path_rate):
        self.d_model = d_model
        self.in_channels = in_channels
        self.patchsize = patchsize
        self.layer_list = layer_list
        self.down_rate = down_rate
        self.spatial_resolution = spatial_resolution
        self.window_size = window_size
        self.freq_ratio = freq_ratio,
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate

aConfig = Config(
    d_model=128, 
    in_channels=1, 
    patchsize=4, 
    layer_list=[2, 2, 12, 2], 
    down_rate=2, 
    spatial_resolution=[256, 256], 
    window_size=8, 
    freq_ratio = 4,
    num_heads=[4, 8, 16, 32],
    drop_path_rate = 0.1
)

@dataclass
class tConfig:
    n_head = 8
    vocab_length: int = 512
    tenc_layers: int = 12
    d_model: int = 768




#-------------------------------------------model defination----------------------------------------------#
clap_amodel = build_audio_encoder(aConfig)            #customized model

hf_clap = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
audio_projection = build_mlp(1024, 512, 512)

class Clap(nn.Module):
    def __init__(self, aConfig):
        super().__init__()
        #self.clap_amodel = build_audio_encoder(aConfig)
        self.clap_amodel = hf_clap.audio_model
        self.clap_tmodel = hf_clap.text_model
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        #self.audio_projection = build_mlp(1024, 512, 512)
        self.audio_projection = hf_clap.audio_projection
        self.projection_head = build_mlp(512, 512, 512)
        self.text_projection = hf_clap.text_projection
    
    
    def forward(self, audios, labels, device):
        split_audio = torch.unbind(audios, dim=0)
        audios = [self.processor(audios=a, return_tensors="pt")["input_features"] for a in split_audio]  
        audio_ft = torch.stack(audios, dim=0).squeeze(2).to(device)
        audio_embd = self.clap_amodel(audio_ft)                                     
        audio_embd_l = self.audio_projection(audio_embd.pooler_output)             #(b, d)
        audio_embd_l = F.normalize(self.projection_head(audio_embd_l))
        b , d = audio_embd_l.shape

        text = []
        for label in labels:
            text.append("Music composed by " + str(label))
        text_t = self.processor(text=text, return_tensors="pt", padding=True).to(device)
        text_t.to(device)
        text_embd = self.clap_tmodel(**text_t)                                       
        text_embd_l = F.normalize(self.text_projection(text_embd.pooler_output))                  #(b, d)

        logits = audio_embd_l @ text_embd_l.T                                          #(b ,b)
        labels = torch.arange(b).to(logits.device)
        loss_a = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.T, labels)
        loss = (loss_a + loss_t) / 2 

        return loss, logits

#=====================================================data preprocessing===================================================#
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[25]


#======================================================resume model========================================================#
pretrained_state_dict = hf_clap.audio_model.audio_encoder.state_dict()
custom_state_dict = clap_amodel.state_dict()
custom_key = {}
for key, value in pretrained_state_dict.items():
    key = key.replace("self.", "")   #replace the self. in original key for consistence
    custom_key[key] = value
# Update the custom model's state dict with the pretrained parameters
custom_state_dict.update(custom_key)
clap_amodel.load_state_dict(custom_state_dict)
audio_projection.load_state_dict(hf_clap.audio_projection.state_dict())

#t_encoder = Text_encoder(tConfig, device = device)
#print("test:", hf_clap.get_text_features("lets do a test"))

#for key, value in t_encoder.state_dict().items():
   # print(f"customized text model:{key}, value:{value.shape}")

#debug the key so that we can load hf params into our models
#for key, value in clap_amodel.state_dict().items():
#    print(f"customized key: {key}, shape:{value.shape}")


#def get_features(name):
#    def hook(model, input, output):
#        if isinstance(output, tuple):
#            features[name] = [o.shape if hasattr(o, 'shape') else type(o) for o in output]
#        else:
#            features[name] = output.shape if hasattr(output, 'shape') else type(output)
#    return hook
#
#
#
#hf_clap.audio_model.audio_encoder.patch_embed.register_forward_hook(get_features('embedding'))
#hf_clap.audio_model.audio_encoder.layers[0].register_forward_hook(get_features("layer1"))
#hf_clap.audio_model.audio_encoder.layers[1].register_forward_hook(get_features("layer2"))
#hf_clap.audio_model.audio_encoder.layers[2].register_forward_hook(get_features("layer3"))
#hf_clap.audio_model.audio_encoder.layers[3].register_forward_hook(get_features("layer4"))
#

#text = ["Sound of a bird", "Sound of a car"]
#text_inputs = processor(text, return_tensors="pt", padding=True)
#audio_inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt")
#
#hf_clap.eval()
#audio_projection.eval()
#clap_amodel.eval()
#for key, value in hf_clap.audio_projection.state_dict().items():
#    print(f"proj params: {key}, shape: {value.shape}")
#with torch.no_grad():
#    hf_text_embd = hf_clap.get_text_features(**text_inputs)
#    hf_audio_embd = hf_clap.get_audio_features(**audio_inputs)
#    audio_output = clap_amodel(audio_inputs['input_features'])
#    hf_audio_output = hf_clap.audio_model(audio_inputs['input_features'])
#    audio_embd = audio_projection(audio_output)
#    audio_embd = audio_embd.flatten(1)
#
#
#audio_embd = F.normalize(audio_embd)
#
#scores2 = audio_embd @ hf_text_embd.T
#scores = hf_audio_embd @ hf_text_embd.T
#print(scores)
#print(scores2)
#for layer_name, feature in features.items():
#    print(f"Feature from {layer_name}: {feature}")





#========================================================training script===================================================================#
from Dataloader import AudioChunkDatasetFromCSV
from torch.utils.data import DataLoader
import gc
import warnings

warnings.filterwarnings("ignore", message=".*sampling_rate*")

#====================================================configuration=============================================================#
epochs = 50
resume = False
checkpoint_path = None
device = "mps"
checkpoint_dir = "/Users/eddy/Documents/Build_from_scratch/AudioLDM/checkpoint_clap"
batchsize = 4
#--------------------------------------------------------preparing dataset------------------------------------------------------------#
csv_file = "/Users/eddy/Desktop/maestro-v1.0.0/maestro-v1.0.0.csv" 
audio_directory = "/Users/eddy/Desktop/maestro-v1.0.0"

train_dataset = AudioChunkDatasetFromCSV(csv_file=csv_file, audio_directory=audio_directory, split="train", chunk_duration=10, sample_rate=16000)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
len_train = train_dataset.__len__()
#
# Test dataset
test_dataset = AudioChunkDatasetFromCSV(csv_file=csv_file, audio_directory=audio_directory, split="test", chunk_duration=10, sample_rate=16000)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
len_test = test_dataset.__len__()
label_list = test_dataset.get_unique_labels()

prompt_list = []
for label in label_list:
    prompt_list.append("Music composed by " + str(label))


#===============================================================evaluate====================================================================#


def evaluate():
    for audio, label in test_dataloader:
        audio = processor(audios = audio.flatten(0), return_tensors="pt")
        audio_embd = hf_clap.get_audio_features(**audio)
        text = prompt_list
        text = processor(text, return_tensors="pt", padding=True)
        text_embd = hf_clap.get_text_features(**text)
        logits = audio_embd @ text_embd.T
        idx = torch.argmax(logits, dim=1)
        #if label == prompt_list[idx]:
        #    count += 1
        print(f"logits: {logits}, label: {label}, predict label: {prompt_list[idx]}")


#========================================================training configration====================================================#
clap_model = Clap(aConfig)
clap_model.train()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, clap_model.parameters()), lr=1e-4)
writer = SummaryWriter()

def train_clap(model, dataloader, optimizer, device, resume = False):
    for name, param in model.named_parameters():
        if "audio_projection" in name or "projection_head" in name:
            param.requires_grad = True
            print("params requires grad:", name)
        else:
            param.requires_grad = False

    if resume == True:
        assert checkpoint_path is not None, "checkpoint can not be emtpy when resume training"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch_start = 0

    model.to(device)

    for epoch in range(epoch_start, epochs):
        if resume == True:
            epoch += epoch_start
        epoch_start_time = time.time()

        count_sample = 0
        for audios, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs - 1}"):
            batch_start_time = time.time()
            count_sample += 1
            optimizer.zero_grad()
            loss, logits = model(audios, labels, device)
            loss.backward()
            optimizer.step()
            print(f"epochs {epoch}, {count_sample}/{(len_train // batchsize) + 1} samples, loss: {loss}")
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            print(f"estimate epoch time:{batch_time * (((len_train // batchsize) + 1) - count_sample):.2f} s.")

        if (epoch + 1) % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"clap_model_test_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        epoch_end_time = time.time()
        print("epochs time:", epoch_end_time - epoch_start_time)
        writer.add_scalar('Loss/train', loss.item(), epoch)
    
    writer.close()


train_clap(clap_model, train_dataloader, optimizer, device)

