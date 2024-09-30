# Classical Music Classifier

The Classical Music Classifier is an AI-based model designed to classify classical music via its composer, fine-tuned on the MAESTRO dataset. While the CLAP (Contrastive Language-Audio Pretraining) model, which serves as the foundation for this classifier, achieves state-of-the-art performance across various general audio classification tasks, it struggles with more specific genres like classical music. Our classifier addresses this gap by being fine-tuned specifically for classical piano works.

## Description

### Dataset
We used the MAESTRO dataset, which contains a rich collection of classical piano music from various composers and periods. The dataset is challenging due to the varying lengths of the audio files. To standardize input for training, we split the audio files into 10-second clips, discarding any clips shorter than 10 seconds as they often lack sufficient information for effective classification. Additionally, the dataset includes repeated pieces of music, with some appearing more than eight times. Although these versions are performed by different musicians, the repetition could introduce bias during training. To mitigate this, we limit each piece to a maximum of three occurrences in the dataset. 

### Model architecture
Our model is consists of an audio encoder and a text encoder which is the same as CLAP model. The overall architecture is shown in Fig.1,  we selected Roberta and HTS-AT for best performace, aligning with the findings of the original CLAP model paper. 

![Fig.1 Overall architecture](<.images/Screenshot 2024-09-30 at 10.44.46 AM.png>)

And we reimplement the audio encoder for practice purpose and reload the pretrained params from hugging face. Besides, an additional mlp layer is added to the end of audio encoder for fine-tuning. These encoders help capture rich features from the audio clips, facilitating improved performance for classical music classification. 

## Getting Started

### Dependencies

* Install all the dependencies:
```
pip install -r requirements.txt
```

* Install newest version of pytorch at: https://pytorch.org


### Executing program

* Firstly you should download the MAESTRO dataset from: https://magenta.tensorflow.org/datasets/maestro
* Preprocess the dataset following Dataset section in the Description above. 
* Then you can train your own model with main.py! Note that we use absolute dir in main.py, so you may change the dir to your own path. 


## Acknowledgments

Codes are partly borrowed from:
* [CLAP] (https://github.com/LAION-AI/CLAP)
* [Hugging face: CLAP](https://huggingface.co/laion/larger_clap_music_and_speech)


