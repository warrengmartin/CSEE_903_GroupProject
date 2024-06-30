import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import librosa
import librosa.display
import librosa.util as librosa_util

class MelSpectogram:
  def compute_mel_spectrograms_old(self, pd_df_audio_data):
    """Read audio data and generate mel spectogram

      Parameters:
      df (pd.DataFrame): Pandas DataFrame

      Return df
    """
    # Extract audio data and sample rates
    audio_frames = pd_df_audio_data["amp"].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
    sample_rates = pd_df_audio_data["sf"].values

    # Calculate mel-spectrograms
    mel_spectrograms = []
    mel_shapes = []
    #hop_length = 128 // 4
    #fmax = 16000 // 2 * 1.5
    for audio_frame, sr in zip(audio_frames, sample_rates):
        mel = librosa.feature.melspectrogram(y=audio_frame, sr=sr, n_fft=1024, win_length=1024, hop_length=68, n_mels=64, fmax=None)
        mel_db = librosa.power_to_db(mel, ref=np.max(mel))
        mel_spectrograms.append(mel_db)
        mel_shapes.append(mel_db.shape)

    # Create new dataframe for mel-spectrograms
    mel_spec_df = pd.DataFrame({
        "mel": mel_spectrograms,
        "mel_shape": mel_shapes
    }, index=pd_df_audio_data.index)

    # Concatenate additional columns from audio files dataframe
    columns_to_add = ["key", "sf", "start", "end", "max_amp", "frame_index", "amp", "label"]
    mel_spec_df = pd.concat([mel_spec_df, pd_df_audio_data[columns_to_add]], axis=1)

    return mel_spec_df

  def compute_mel_spectrogram(self, row):
      audio_frame = np.array(row["amp"]) if not isinstance(row["amp"], np.ndarray) else row["amp"]
      sr = row["sf"]
      mel = librosa.feature.melspectrogram(y=audio_frame, sr=sr, n_fft=1024, win_length=1024, hop_length=68, n_mels=64, fmax=None)
      mel_db = librosa.power_to_db(mel, ref=np.max(mel))
      return mel_db, mel_db.shape

  def compute_mel_spectrograms(self, pd_df_audio_data):
      """Read audio data and generate mel spectrogram

      Parameters:
      df (pd.DataFrame): Pandas DataFrame

      Return df
      """
      # Compute mel spectrograms and their shapes
      mel_spec_results = pd_df_audio_data.apply(lambda row: self.compute_mel_spectrogram(row), axis=1)
      mel_spectrograms, mel_shapes = zip(*mel_spec_results)
      
      # Create new dataframe for mel-spectrograms
      mel_spec_df = pd.DataFrame({
          "mel": mel_spectrograms,
          "mel_shape": mel_shapes
      }, index=pd_df_audio_data.index)

      # Concatenate additional columns from audio files dataframe
      columns_to_add = ["key", "sf", "start", "end", "max_amp", "frame_index", "amp", "label"]
      mel_spec_df = pd.concat([mel_spec_df, pd_df_audio_data[columns_to_add]], axis=1)

      return mel_spec_df