# Classical Music Classifier

The Classical Music Classifier is an AI-based model designed to classify classical music via its composer, fine-tuned on the MAESTRO dataset. While the CLAP (Contrastive Language-Audio Pretraining) model, which serves as the foundation for this classifier, achieves state-of-the-art performance across various general audio classification tasks, it struggles with more specific genres like classical music. Our classifier addresses this gap by being fine-tuned specifically for classical piano works.

## Description

### Dataset
We used the MAESTRO dataset, which contains a rich collection of classical piano music from various composers and periods. The dataset is challenging due to the varying lengths of the audio files. To standardize input for training, we split the audio files into 10-second clips, discarding any clips shorter than 10 seconds as they often lack sufficient information for effective classification.

### Model architecture
Our model is consists of an audio encoder and a text encoder which is the same as CLAP model. The overall architecture is shown in Fig.1,  we selected Roberta and HTS-AT for best performace, aligning with the findings of the original CLAP model paper. And we reimplement the audio encoder for practice purpose and reload the pretrained params from hugging face. Besides, an additional mlp layer is added to the end of audio encoder for fine-tuning. These encoders help capture rich features from the audio clips, facilitating improved performance for classical music classification. 

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
