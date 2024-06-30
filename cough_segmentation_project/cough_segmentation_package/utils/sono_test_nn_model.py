import numpy as np
import pandas as pd
from feature_extraction import Spectrogram
from feature_extraction_melspectogram import MelSpectogram
from sono_metrics import SonoMetrics, SonoMetricsPlot

class SonoTestNNModel:
  def mel_spectrogram_features(self, df_frame):
    mel_spectogram = MelSpectogram()
    mel_spec_df = mel_spectogram.compute_mel_spectrograms(df_frame)
    return mel_spec_df

  def spectrogram_features(self, df_frame):
    spectrogram = Spectrogram()

    frame_size = 1024

    obs = 64  #The number of frequency bins in the output (output bin size - obs)

    n_fft = (obs * 2) - 1   #sub-frame size
    win_length = n_fft  #apply windowing function to the entire length of the frame

    hop_percent = 50
    hop_length = (n_fft + 1) * hop_percent // 100

    df_1024_spectrogram = spectrogram.get_spectogram_dataset(
        frame_size=frame_size,
        frames_dic={frame_size:df_frame},
        limit=0,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        use_gpu=False
    )

    # save dataset in cache / temp storage, it helps to free up memory when fully implemented
    #spectrogram.save_spectogram_data(df=df_1024_spectrogram, file_name=frame_size, dir='', temporary=True)

    #visualize only 1 spectogram for confirmation
    if self.debug:
      key = df_1024_spectrogram.shape[0]//2
      spectrogram.plot_spectrogram( file_name=df_1024_spectrogram['key'][key], S_DB=np.array(df_1024_spectrogram['stft'][key]), sr=df_1024_spectrogram['sf'][key], hop_length=64 )
      print('shape of stft', df_1024_spectrogram['stft'][key].shape )
      print('shape of the frame', df_1024_spectrogram['amp'][key].shape )

    return df_1024_spectrogram

  def get_metrics(self, y_test, y_pred, y_pred_prob):
    SonoMetrics(y_test, y_pred, y_pred_prob)

  def prep_dataset(self, result_df):
    X_val = np.array([spec.reshape(64, 16, 1) for spec in result_df[self.key]])
    y_val = result_df['label'].values
    return X_val, y_val

  def predict_nn(self, model, df_frame, percent_proba=0.875, key='stft', metrics=True, debug=False):
    self.debug = debug
    self.key = key
    self.percent_proba = percent_proba

    if self.key == 'stft':
      df_frame = self.spectrogram_features(df_frame)
    else:
      df_frame = self.mel_spectrogram_features(df_frame)

    X_test, y_test = self.prep_dataset(df_frame)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > self.percent_proba).astype(int)

    if metrics:
      self.get_metrics(y_test, y_pred, y_pred_prob)

    return SonoMetricsPlot(df_frame, y_pred)