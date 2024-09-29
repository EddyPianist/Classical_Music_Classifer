import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio

import pandas as pd

def preprocess_csv(csv_file, min_files=8, max_repeats=3):
    """
    Preprocess the CSV file to:
    1. Remove composers with fewer than min_files entries.
    2. Keep only max_repeats copies of pieces that appear more than max_repeats times (randomly).
    
    :param csv_file: Path to the CSV file.
    :param min_files: Minimum number of files required to keep a composer.
    :param max_repeats: Maximum number of times a piece can appear (keeps only max_repeats).
    :return: Filtered DataFrame.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Step 1: Filter composers with fewer than min_files entries
    composer_counts = data['canonical_composer'].value_counts()
    valid_composers = composer_counts[composer_counts >= min_files].index
    filtered_data = data[data['canonical_composer'].isin(valid_composers)]
    
    # Step 2: Keep only max_repeats copies of pieces that appear more than max_repeats times
    def keep_max_repeats(group):
        if len(group) > max_repeats:
            return group.sample(n=max_repeats, random_state=42)  # Randomly sample max_repeats rows
        return group

    filtered_data = filtered_data.groupby('canonical_title').apply(keep_max_repeats).reset_index(drop=True)
    
    return filtered_data

class AudioChunkDatasetFromCSV(Dataset):
    def __init__(self, csv_file, audio_directory, split="train", chunk_duration=10, sample_rate=16000):
        """
        :param csv_file: Path to the CSV file containing file names and labels
        :param audio_directory: Directory where the audio files are stored
        :param split: Specify 'train' or 'test' to filter dataset
        :param chunk_duration: Duration of each chunk in seconds
        :param sample_rate: Sample rate of audio
        """
        self.audio_directory = audio_directory
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_length = chunk_duration * sample_rate
        
        # Load CSV file
        data = preprocess_csv(csv_file, min_files=8, max_repeats=3)
        
        # Filter data based on the 'split' column
        self.data = data[data['split'] == split]
        
        # Extract file names and labels
        self.file_names = self.data['audio_filename'].tolist()
        self.labels = self.data['canonical_composer'].tolist()

    def __len__(self):
        total_chunks = 0
        for file_name in self.file_names:
            file_path = os.path.join(self.audio_directory, file_name)
            audio, sample_rate = torchaudio.load(file_path)
            total_chunks += audio.shape[1] // self.chunk_length  # Total chunks per file
        return total_chunks
    #def __len__(self):
    #    # Return total number of chunks but limit to max_chunks
    #    total_chunks = 0
    #    for file_name in self.file_names:
    #        file_path = os.path.join(self.audio_directory, file_name)
    #        audio, _ = torchaudio.load(file_path)
    #        num_chunks = audio.shape[1] // self.chunk_length
    #        total_chunks += num_chunks
    #        if total_chunks >= 50:
    #            return 50
    #    return min(total_chunks, 50)
#

    def __getitem__(self, idx):
        cumulative_chunks = 0
        for file_name, label in zip(self.file_names, self.labels):
            file_path = os.path.join(self.audio_directory, file_name)
            audio, sample_rate = torchaudio.load(file_path)
            num_chunks = audio.shape[1] // self.chunk_length
            
            if cumulative_chunks + num_chunks > idx:
                chunk_idx = idx - cumulative_chunks
                start_sample = chunk_idx * self.chunk_length
                end_sample = start_sample + self.chunk_length
                
                # Extract the chunk, discard if shorter than expected
                audio_chunk = audio[:, start_sample:end_sample]
                if audio_chunk.shape[1] == self.chunk_length:
                    return audio_chunk[0], label
            
            cumulative_chunks += num_chunks

        raise IndexError(f"Index {idx} out of range")
    
    def get_unique_labels(self):
        """
        Return a list of unique labels (canonical_composer values).
        """
        return self.data['canonical_composer'].unique().tolist()

# Create the dataset and dataloader
#dataset = AudioChunkDatasetFromCSV(csv_file=csv_file, audio_directory=audio_directory, chunk_duration=10, sample_rate=16000)
#dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


