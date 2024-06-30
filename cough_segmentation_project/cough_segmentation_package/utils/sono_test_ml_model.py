from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_extraction_zcr_mfcc_rms import TimeFreqFeatures
from sono_metrics import SonoMetrics, SonoMetricsPlot

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

class SonoTestMlModel:
  def time_freq_features(self, df_frame):
    time_freq_features = TimeFreqFeatures()
    df_frame = time_freq_features.get_features(df_frame)
    if self.debug:
      df_frame.info()

    return df_frame

  def get_metrics(self, y_test, y_pred, y_pred_prob):
    SonoMetrics(y_test, y_pred, y_pred_prob)

  def combined_scaler(self):
    with open(self.scaler, 'rb') as f:
      loaded_scalers = pickle.load(f)

    # Extract the means and variances of all scalers
    means = []
    variances = []
    for scaler in loaded_scalers.values():
        means.append(scaler.mean_)
        variances.append(scaler.var_)

    # Average the means and variances
    mean_combined = np.mean(means, axis=0)
    var_combined = np.mean(variances, axis=0)
    std_combined = np.sqrt(var_combined)
    return CombinedScaler(mean_combined, std_combined, var_combined)

  def prep_dataset(self, result_df):
    X_test = result_df.drop(columns=['amp', 'sf', 'label', 'start', 'end', 'max_amp', 'frame_index', 'key', 'mfcc_14', 'mfcc_16', 'mfcc_15', 'mfcc_17', 'mfcc_18', 'mfcc_19', 'mfcc_20'])
    y_test = result_df['label']

    if self.debug:
      X_test.info()

    if self.scaler:
      scaler = self.combined_scaler()
      X_test_scaled = scaler.transform(X_test.values)
    else:
      X_test_scaled = X_test.values

    return X_test_scaled, y_test

  def predict_ml(self, model, df_frame, scaler, debug=False):
    self.debug = debug
    self.scaler = scaler
    df_frame = self.time_freq_features(df_frame)

    X_test, y_test = self.prep_dataset(df_frame)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    self.get_metrics(y_test, y_pred, y_pred_prob)

    return SonoMetricsPlot(df_frame, y_pred)

# Create a new scaler with the combined mean and std
class CombinedScaler(StandardScaler):
  def __init__(self, mean, std, var):
    super().__init__()
    self.mean_ = mean
    self.scale_ = std
    self.var_ = var
    self.n_features_in_ = mean.shape[0]

  def transform(self, X, copy=None):
    copy = copy if copy is not None else self.copy
    X = self._validate_data(X, reset=False, accept_sparse='csr', copy=copy, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    X -= self.mean_
    X /= self.scale_
    return X
