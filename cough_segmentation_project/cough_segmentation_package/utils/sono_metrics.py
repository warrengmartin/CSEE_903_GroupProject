from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SonoMetrics:
  def __init__(self, y_test, y_pred, y_pred_prob):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"F1: {f1 * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()

class SonoMetricsPlot:
  def __init__(self, df, y_pred):
    df["label_pred"] = y_pred

    # Group by 'B' and apply the concatenation
    merged_df = self.rebuild_audio_from_frames(df_frames=df)

    # Set the index as column 'B'
    merged_df.index.name = 'key'

    #print( merged_df.head(3) )
    self.merged_df = merged_df

    self.plot_pred( merged_df )
  
  def rebuild_single_audio_from_frames(self, df_single_audio):
    frame_size = 1024 // 2
    dfc = {"amp":[], "amp_length":[], "label":[], "label_pred":[]}
    i = 0
    for i1, r1 in df_single_audio.iterrows():
      s = i * frame_size
      e = s + frame_size
      #print(i, 'start-end', s,e)
      #print(r1["amp"][:frame_size].shape)
      dfc["amp"] += list(r1["amp"][:frame_size])

      dfc["label"] += [r1["label"]] * frame_size
      dfc["label_pred"] += [r1["label_pred"]/2] * frame_size
        
      dfc["amp_length"] = len( dfc["amp"] )
      i += 1
    return dfc

  def rebuild_audio_from_frames(self, df_frames):
    dfc = {}
    i = 0
    for dk in np.unique(df_frames["key"].values):
      dfc[dk] = self.rebuild_single_audio_from_frames(df_single_audio=df_frames[df_frames["key"] == dk])
    return pd.DataFrame(dfc).T

  def plot_pred(self, df, percent=0):
    SonoPlotAmpLabels(df=df, percent=percent)

class SonoPlotAmpLabels:
  def __init__(self, df, percent=0) -> None:
    for index, row in df.iterrows():
      if percent > 0:
        argmax = np.argmax(row["amp"])
        three_percent = len(row["amp"])*percent
        start_zoom = int(argmax - three_percent)
        end_zoom = int(argmax + three_percent)
        print(f'Max value {np.max(row["amp"])} pos {argmax} {start_zoom},{end_zoom}')
        pd.Series( row["amp"][start_zoom:end_zoom]).plot(figsize=(16, 4),lw=1, title=f'{index}')
        pd.Series( row["label"][start_zoom:end_zoom]).plot(figsize=(16, 4),lw=2)
        if "label_pred" in df.columns:
          pd.Series( row["label_pred"][start_zoom:end_zoom]).plot(figsize=(16, 4),lw=2)
      else:
        #print(len(row["label"]), 'amp', len(row["amp"]))
        pd.Series( row["amp"]).plot(figsize=(16, 4),lw=1, title=f'{index}')
        pd.Series( row["label"]).plot(figsize=(16, 4),lw=2)
        if "label_pred" in df.columns:
          pd.Series( row["label_pred"]).plot(figsize=(16, 4),lw=2)

      plt.show()