import torch
import pretty_midi
from sklearn.preprocessing import LabelEncoder
from Model import MusicModel
from Prepare_Data import extract_notes_from_folder
import os

def generate_music(model, encoder, seed_sequence, length=100, device='cpu'):
  model.eval()
  generated = seed_sequence[:]
  input_seq = seed_sequence[:]

  for _ in range(length):
    input_tensor = torch.tensor([encoder.transform(input_seq)], dtype=torch.long).to(device)
    with torch.no_grad():
      output, _ = model(input_tensor)
    probabilities = torch.softmax(output[0], dim=0).cpu()
    next_index = torch.multinomial(probabilities, 1).item()
    next_note = encoder.classes_[next_index]

    generated.append(next_note)
    input_seq = input_seq[1:] + [next_note]

  return generated

def notes_to_midi(notes_list, output_path='generated.mid'):
  pm = pretty_midi.PrettyMIDI()
  piano = pretty_midi.Instrument(program=0)

  start_time = 0
  for note_str in notes_list:
    pitch, duration, program = note_str.split('_')
    pitch = int(pitch)
    duration = float(duration)

    note = pretty_midi.Note(
      velocity=100,
      pitch=pitch,
      start=start_time,
      end=start_time + duration
    )
    piano.notes.append(note)
    start_time += duration

  pm.instruments.append(piano)
  pm.write(output_path)
  print(f"MIDI file saved to {output_path}")

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  notes = extract_notes_from_folder("data")

  encoder = LabelEncoder()
  encoder.fit(notes)

  model = MusicModel(vocab_size=len(encoder.classes_))
  model.load_state_dict(torch.load("music_model.pth", map_location=device))
  model.to(device)

  seed_seq = notes[:50]

  generated_notes = generate_music(model, encoder, seed_seq, length=200, device=device)

  notes_to_midi(generated_notes, output_path="generated_music.mid")

if __name__ == "__main__":
  main()