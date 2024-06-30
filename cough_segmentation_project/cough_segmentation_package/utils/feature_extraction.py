import pandas as pd
import numpy as np
import torch

import matplotlib.pylab as plt
import librosa
import librosa.display

class Spectrogram:
  def stft(self, df):
    """Read audio data and apply short_time_fourier_transform

      Parameters:
      df (pd.DataFrame): Pandas DataFrame

      Return df
    """
    ft = []
    ft_shape = []
    for index, row in df.iterrows():
      n_fft = min(2048, len(row["amp"]))
      stft = librosa.amplitude_to_db(np.abs( librosa.stft(row["amp"], n_fft=n_fft) ), ref=np.max)
      ft.append( stft )
      ft_shape.append( stft.shape )

    df["stft"] = ft
    df["stft_shape"] = ft_shape

    return df

  def stft_optimized(self, row):
    """Read audio data and apply short_time_fourier_transform in shorter period of time

      Parameters:
      df.row (pd.DataFrame): Pandas DataFrame

      Return df.row
    """
    #stft = librosa.amplitude_to_db(np.abs( librosa.stft(row["amp"], n_fft=128, hop_length=64, win_length=64) ), ref=np.max)
    stft = librosa.amplitude_to_db(np.abs( librosa.stft(row["amp"], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length) ), ref=np.max)
    return stft

  def set_sftt_params(self, n_fft, hop_length, win_length):
    self.hop_length = hop_length
    self.n_fft = n_fft
    self.win_length = win_length
    print(f'using n_fft:{self.n_fft}, hop length:{self.hop_length}, win_length:{self.win_length}')

  def plot_spectrogram(self, file_name, S_DB, sr, hop_length=512):
    plt.figure(figsize=(10, 4))
    #librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    #linear is not used bcos it results in max y-axis value of upto 8000 instead of 4096
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram {file_name} at {sr/1000}Khz, Hop {hop_length}')
    plt.tight_layout()
    plt.show()

  def save_spectogram_data(self, df, file_name, dir='', temporary=True):
    """Save data to npy format for later reuse

      Parameters:
      df.row (pd.DataFrame): Pandas DataFrame

      Return none
    """
    # Save spectrograms and labels as NumPy arrays
    if temporary:
        dir = '/content/'

    np.save(f'{dir}{file_name}.npy', df)
    # Extract column headings
    np.save(f'{dir}{file_name}_columns.npy', df.columns.to_numpy())

  def load_spectogram_data(self, file_name, dir='', temporary=True):
    """Load data from npy format

      Parameters:
      df.row (pd.DataFrame): Pandas DataFrame

      Return df
    """
    if temporary:
        dir = '/content/'
    # Save spectrograms and labels as NumPy arrays
    data = np.load(f'{dir}{file_name}.npy', allow_pickle=True)
    col = np.load(f'{dir}{file_name}_columns.npy',allow_pickle=True)
    return pd.DataFrame(data,columns=col)

  def get_spectogram_dataset(self, frames_dic={}, frame_size=1024, limit=0, n_fft=1024, hop_length=64, win_length=1024, use_gpu=True):
    """Extract Spectogram dataset from all_frames_dic its optimized to be faster about 1m 42s for 1028

      Parameters:
      frames_dic (dict): Dictionary with key = frame_size and value = df
      frame_size (int): Frame sizes to extract spectogram
      limit (int): Limit to stop extraction after X records, useful for debugging

      Return dict
    """
    dfc = None
    if frame_size in frames_dic:
      print('starting:', frame_size )
      self.set_sftt_params(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
      if limit > 0:
        dfc = frames_dic[frame_size].copy().head(limit)
      else:
        dfc = frames_dic[frame_size].copy()
      
      if use_gpu and self.has_gpu():
        dfc['stft'] = self.use_gpu_for_stft(dfc["amp"])
      else:
        if use_gpu:
          print('Failed to find GPU')
        dfc['stft'] = dfc.apply(self.stft_optimized, axis=1)

    else:
      print('frame size not found:', frame_size )

    return dfc
  
  def has_gpu(self):
    """Test to determine if GPU is avialable

    Return torch.device
    """
    has_gpu = False
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
      has_gpu = True
    print(f'Using device: {device}')
    return device

  def use_gpu_for_stft(self, amps):
    """
    Computes the Short-Time Fourier Transform (STFT) of an audio signal on the GPU 

    Parameters:
    self (object): Reference to the class instance (likely for accessing device).
    amps (torch.Tensor): The audio signal as a n-D tensor (where n is the number of rows)

    Returns:
        list: A list containing the magnitude spectrograms (in dB) of the audio samples after performing STFT on GPU.
    """
    # Convert NP array to have float dtype
    amps_np = np.stack(amps.values)

    # Convert the NumPy array to a tensor
    amps_tensor = torch.tensor(amps_np)

    # Move tensor to GPU
    amps_tensor = amps_tensor.cuda()

    window = torch.hann_window(self.n_fft).cuda()  # Window function

    # Perform STFT
    stft_output = torch.stft(
        amps_tensor,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        window=window,
        return_complex=True
    )

    # Compute the magnitude of the STFT
    #stft_db = torch.abs(stft_output)
    magnitude = torch.abs(stft_output)

    # log to base e, natural log
    #stft_db = torch.log(stft_db)

    # log to base 10
    #stft_db = torch.log(stft_db) / torch.log(torch.tensor(2.0))

    # Convert the magnitude to decibels
    stft_db = self.amplitude_to_db_gpu(magnitude, ref=torch.max(magnitude))

    return list(stft_db.cpu().numpy())
  
  def amplitude_to_db_gpu(self, tensor, ref=1.0, amin=1e-20):
    """
    Converts an amplitude spectrogram to decibel (dB) scale in PyTorch.

    Parameters:
    spectrogram (torch.Tensor): The amplitude spectrogram with shape (..., freq, time).
    ref (float, optional): Reference value for dB conversion. Defaults to 1.0.

    Returns:
        torch.Tensor: The spectrogram converted to dB scale with the same shape as input.
    """
    # Ensure the spectrogram is non-negative (avoiding issues with log)
    spectrogram = torch.clamp(tensor, min=amin)  # Small value to prevent log(0)

    # Convert to power spectrogram (square the elements)
    power_spectrogram = spectrogram**2

    # Calculate dB relative to reference
    S_db = 20 * torch.log10(power_spectrogram / ref)

    return S_db