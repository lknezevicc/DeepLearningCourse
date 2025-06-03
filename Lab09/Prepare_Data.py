import pretty_midi
import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

def extract_notes_from_folder(folder_path):
  notes = []
  for root, _, files in os.walk(folder_path):
    for file in files:
      if file.endswith('.mid') or file.endswith('.midi'):
        file_path = os.path.join(root, file)
        try:
          midi = pretty_midi.PrettyMIDI(file_path)
          for instrument in midi.instruments:
            if instrument.is_drum:
              continue
            for note in instrument.notes:
              pitch = note.pitch
              duration = round(note.end - note.start, 2)
              instrument_id = instrument.program
              notes.append(f"{pitch}_{duration}_{instrument_id}")
        except Exception as e:
          print(f"Parsing failed for {file_path}: {e}")
  return notes

def prepare_sequences(notes, seq_len=50):
  inputs, targets = [], []
  for i in range(len(notes) - seq_len):
    inputs.append(notes[i:i+seq_len])
    targets.append(notes[i+seq_len])
  return inputs, targets

class MusicDataset(Dataset):
  def __init__(self, input_seqs, target_seqs, encoder):
    flat_inputs = sum(input_seqs, [])
    encoded = encoder.transform(flat_inputs)
    self.input_seqs = torch.tensor(encoded, dtype=torch.long).view(len(input_seqs), -1)
    self.target_seqs = encoder.transform(target_seqs)

  def __len__(self):
    return len(self.input_seqs)

  def __getitem__(self, idx):
    return (
      torch.tensor(self.input_seqs[idx], dtype=torch.long),
      torch.tensor(self.target_seqs[idx], dtype=torch.long)
    )