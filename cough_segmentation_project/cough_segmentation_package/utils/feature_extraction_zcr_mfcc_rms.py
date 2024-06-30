import pandas as pd
import numpy as np
import librosa
from joblib import Parallel, delayed

class TimeFreqFeatures:
  def get_features(self, df_frame):
    # Apply feature extraction using pd.apply
    features_df = df_frame.apply(self.extract_features, axis=1, result_type='expand')

    # Concatenate original DataFrame with the features DataFrame
    return pd.concat([df_frame, features_df], axis=1)

  # Function to extract features
  def extract_features(self, row):
      y = row['amp']
      sr = row['sf']
      features = {}

      # Time-domain features
      features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
      features['rms'] = np.mean(librosa.feature.rms(y=y))

      # Mel-frequency cepstral coefficients (MFCC)
      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=1024)
      for i in range(mfcc.shape[0]):
          features[f'mfcc_{i+1}'] = np.mean(mfcc[i])

      return features
