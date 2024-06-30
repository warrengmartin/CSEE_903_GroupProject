import os
import shutil

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

import librosa
import librosa.display

class Framing:
  def visualize_overlapping_frames(self, title, framed_df, visual_set, figsize=(12,6)):
    """Visualizes overlapping frames of audio data.

    Parameters:
        title (str): The title of the visualization.
        framed_df (pd.DataFrame): A DataFrame containing information about the framed audio data.
        visual_set (dict): A dictionary containing information about the frames to visualize.

    Returns:
        None
    """
    for i, v in visual_set.items():
      #print( framed_df[framed_df["key"] == v]["max_amp"].idxmax() )
      findex = framed_df[framed_df["key"] == v]["max_amp"].idxmax() - 1
      hop_length = (framed_df.loc[findex]["end"] - framed_df.loc[findex]["start"]) // 2
      print('Hop Length:', hop_length)
      frame1 = framed_df.loc[findex]["amp"]
      frame2 = framed_df.loc[findex+1]["amp"]
      frame3 = framed_df.loc[findex+2]["amp"]
      # Plot the overlapping frames
      plt.figure(figsize=figsize)

      # Plot first frame
      plt.plot(np.arange(len(frame1)), frame1, label='Frame 1')

      # Plot second frame
      plt.plot(np.arange(hop_length, hop_length + len(frame2)), frame2 + 1, label='Frame 2')

      # Plot third frame
      plt.plot(np.arange(2 * hop_length, 2 * hop_length + len(frame3)), frame3 + 2, label='Frame 3')

      plt.xlabel('Samples')
      plt.ylabel('Amplitude')
      plt.title(f'{title} {i} {v}')
      plt.legend()
      plt.show()

  def apply_framing(self, audio_df, frame_sizes=[256, 512, 1024, 2048]):
    """Applies framing to audio data.

    Parameters:
        audio_df (DataFrame): A DataFrame containing audio data.
        frame_sizes (list): List of frame sizes to apply

    Returns:
        dict: A dictionary containing framed audio data for different frame sizes.
    """
    def create_overlapping_frames(key, amp, label, sf, frame_size, hop_length):
      """Creates overlapping frames from audio data.

      Parameters:
          key (str): Key identifier for the audio data.
          amp (np.array): Array containing amplitude data.
          label (np.array): Array containing label data.
          sf (int): Sampling frequency of the audio data.
          frame_size (int): Size of each frame.
          hop_length (int): Hop length between frames.

      Returns:
          dict: A dictionary containing information about the created frames.
      """
      if len(amp) == len(label):
        total_frames = 1 + int((len(amp) - frame_size) / hop_length)
        dic = {"key":[], "sf":[], "start":[], "end":[], "max_amp":[], "frame_index":[], "amp":[], "label":[]}

        # Create overlapping frames
        for i in range(total_frames):
          dic["key"].append(key)
          dic["sf"].append(sf)
          dic["start"].append(i * hop_length)
          dic["end"].append(i * hop_length + frame_size)
          dic["max_amp"].append(np.max(amp[i * hop_length: i * hop_length + frame_size]))
          dic["frame_index"].append(i)
          dic["amp"].append(amp[i * hop_length: i * hop_length + frame_size])
          frames_label_raw = label[i * hop_length: i * hop_length + frame_size]

          f_label = 0
          if np.sum(frames_label_raw==1) > (len(frames_label_raw) / 2):
            f_label = 1
          dic["label"].append(f_label)

        return dic
      else:
        print('Error: non matching amp and labels', key, len(amp), len(label))

    def create_and_label_frames(audio_df):
      """Creates and labels frames from audio data.

        Parameters:
        audio_df (DataFrame): A DataFrame containing audio data.

        Returns:
            dict: A dictionary containing information about the created frames.
        """
      return create_overlapping_frames(audio_df.name, audio_df["amp"], audio_df["label"],audio_df["sf"],frame_size, hop_length)

    all_frames = {}
    for frame_size in frame_sizes:
      hop_length = frame_size // 2

      frame_df = pd.DataFrame()
      #xx = audio_df.head(1).apply(create_and_label_frames, axis=1)
      xx = audio_df.apply(create_and_label_frames, axis=1)
      for x in xx:
        if len(frame_df) > 0:
          frame_df = pd.concat([frame_df, pd.DataFrame(x)], ignore_index=True)
        else:
          frame_df = pd.DataFrame(x)

      all_frames[frame_size] = frame_df
      print(f'Frame size {frame_size}, hop_length {hop_length}, count {len(frame_df)}')
    return all_frames