import numpy as np
import pandas as pd
from fastcore.basics import GetAttr
from fastai.tabular.data import TabularPandas
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import seaborn as sns
from sklearn.utils.fixes import loguniform
from scipy import stats
import torch
import os
import matplotlib.pylab as plt
from tabulate import tabulate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix, mean_squared_error


class TrainZCR:
  # --- Plot and Save Metrics ---
  def plot_fold_metrics(self, results, metric_names, output_dir='./'):
      """Plots individual and combined line graphs for metrics across folds, saving them as PNG files.

      Args:
          results (list): List of dictionaries containing fold results.
          metric_names (list): List of metric names to plot.
          output_dir (str, optional): Directory to save the plots. Defaults to the current directory.
      """
      num_folds = len(results)

      # Create the output directory if it doesn't exist
      os.makedirs(output_dir, exist_ok=True)

      # Individual Metric Plots
      for i, metric_name in enumerate(metric_names):
          plt.figure(figsize=(10, 6))
          train_scores = [fold['Train'][metric_name][0] for fold in results]
          valid_scores = [fold['Valid'][metric_name][0] for fold in results]

          plt.plot(range(1, num_folds + 1), train_scores, label='Train')
          plt.plot(range(1, num_folds + 1), valid_scores, label='Validation')

          plt.xlabel('Fold')
          plt.ylabel(metric_name)
          plt.title(f'{metric_name} across Folds')
          plt.legend()
          plt.grid(True)

          plt.savefig(os.path.join(output_dir, f'RF_TrainTest_{metric_name}_5-folds.png'))
          plt.close()

      # Combined Metrics Plot
      plt.figure(figsize=(15, 10))
      for metric_name in metric_names:
          train_scores = [fold['Train'][metric_name][0] for fold in results]
          valid_scores = [fold['Valid'][metric_name][0] for fold in results]

          plt.plot(range(1, num_folds + 1), train_scores, label=f'{metric_name} (Train)')
          plt.plot(range(1, num_folds + 1), valid_scores, label=f'{metric_name} (Validation)')

      plt.xlabel('Fold')
      plt.ylabel('Metric Score')
      plt.title('All Metrics across Folds')
      plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside plot
      plt.grid(True)
      plt.tight_layout()  # Adjust layout for legend outside
      plt.savefig(os.path.join(output_dir, 'RF_TrainTest_All_Metrics_5-folds.png'))
      plt.close()



  # --- Generate and Save Hyperparameter Table ---

  def round_value(self, x, decimals=3):
      """Round a value or the first element of a tuple to the specified number of decimal places."""
      if isinstance(x, tuple):
          return round(x[0], decimals)
      elif isinstance(x, (int, float)):
          return round(x, decimals)
      else:
          return x  # Return as is if it's neither a tuple nor a number

  def generate_metrics_table(self, fold_metrics, avg_train_metrics, avg_valid_metrics, metric_names):
      table_data = []
      for i, fold_metrics_row in enumerate(fold_metrics):
          train_row = [round_value(x, 3) for x in fold_metrics_row[:len(metric_names)//2]]
          test_row = [round_value(x, 3) for x in fold_metrics_row[len(metric_names)//2:]]
          row = [f"Fold {i+1}"] + train_row + test_row
          table_data.append(row)

      # Round average metrics
      avg_train_row = [round_value(avg_train_metrics[metric], 3) for metric in metric_names[:len(metric_names)//2]]
      avg_test_row = [round_value(avg_valid_metrics[metric], 3) for metric in metric_names[len(metric_names)//2:]]
      avg_row = ["Average"] + avg_train_row + avg_test_row
      table_data.append(avg_row)

      header = ["Fold"] + [f"{k} (Train)" for k in metric_names[:len(metric_names)//2]] + [f"{k} (Test)" for k in metric_names[len(metric_names)//2:]]
      return tabulate(table_data, headers=header, tablefmt="grid")

  def plot_all_metrics(self, results, metric_names, output_dir='./'):
      num_folds = len(results)
      plt.figure(figsize=(15, 10))

      # Define a list of colors and linestyles
      colors = plt.cm.tab10.colors
      linestyles = ['-', '--', '-.', ':']

      # Create a set of unique metric names
      unique_metrics = set(metric_names)

      for idx, metric_name in enumerate(unique_metrics):
          try:
              train_scores = []
              valid_scores = []
              for fold in results:
                  train_value = fold['Train'].get(metric_name)
                  valid_value = fold['Valid'].get(metric_name)

                  if train_value is not None:
                      train_scores.append(float(self.round_value(train_value, 3)))
                  if valid_value is not None:
                      valid_scores.append(float(self.round_value(valid_value, 3)))

              if train_scores and valid_scores:
                  color = colors[idx % len(colors)]
                  linestyle = linestyles[idx % len(linestyles)]
                  plt.plot(range(1, len(train_scores) + 1), train_scores, label=f'{metric_name} (Train)', marker='o', color=color)
                  plt.plot(range(1, len(valid_scores) + 1), valid_scores, label=f'{metric_name} (Validation)', marker='s', linestyle=linestyle, color=color)
              else:
                  print(f"Warning: No valid data for metric {metric_name}")
          except Exception as e:
              print(f"Error plotting metric {metric_name}: {str(e)}")

      plt.xlabel('Fold')
      plt.ylabel('Metric Score')
      plt.title('All Metrics across Folds')
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside plot
      plt.grid(True)
      plt.tight_layout()  # Adjust layout for legend outside
      plt.savefig(f'{output_dir}/All_Metrics_across_Folds.png', dpi=300, bbox_inches='tight')
      plt.close()
      
  # Function to create a grid of features
  def create_feature_grid(self, df, n_rows=10):
      grids = []
      for i in range(0, len(df), n_rows):
          grid = df.iloc[i:i+n_rows]
          grids.append(grid)
      return grids

  def save_feature_importance_table(self, grid, filename, output_dir='./'):
    """Saves a feature importance grid as a PNG image.

    Args:
        grid (pandas.DataFrame): The feature importance grid.
        filename (str): The desired filename for the PNG image.
        output_dir (str, optional): Directory to save the image. Defaults to the current directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, len(grid) * 0.5))  # Adjust size as needed

    # Hide axes
    ax.axis('off')

    # Create the table plot
    table = ax.table(cellText=grid.values,
                    colLabels=grid.columns,
                    cellLoc='center',
                    loc='center')

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust scaling as needed

    # Save the figure
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance table saved as PNG: {filepath}")


  def save_table_as_png(self, table, filename, output_dir='./'):
      """Saves a table generated with tabulate as a PNG image."""
      os.makedirs(output_dir, exist_ok=True)
      filepath = os.path.join(output_dir, filename)

      try:
          # Split the table into rows
          rows = table.split('\n')

          # Remove any empty rows
          rows = [row for row in rows if row.strip()]

          # Determine the number of columns
          num_columns = max(len(row.split('|')) - 1 for row in rows)  # Subtract 1 for the leading '|'

          # Create a figure and axis
          fig, ax = plt.subplots(figsize=(num_columns * 2, len(rows) * 0.5))

          # Hide axes
          ax.axis('off')

          # Create the table plot
          table_data = []
          for row in rows[1:]:  # Skip the header row
              # Split the row and remove any empty strings
              cells = [cell.strip() for cell in row.split('|') if cell.strip()]
              # Pad the row with empty strings if necessary
              cells += [''] * (num_columns - len(cells))
              if cells:
                  table_data.append(cells)

          # Extract headers
          headers = [header.strip() for header in rows[0].split('|') if header.strip()]
          headers += [''] * (num_columns - len(headers))

          # Create the table plot
          table = ax.table(cellText=table_data,
                          colLabels=headers,
                          cellLoc='center',
                          loc='center')

          # Adjust table style
          table.auto_set_font_size(False)
          table.set_fontsize(8)
          table.scale(1.2, 1.2)

          # Save the figure
          plt.tight_layout()
          plt.savefig(filepath, dpi=300, bbox_inches='tight')
          plt.close()

          print(f"Table saved as PNG: {filepath}")
      except Exception as e:
          print(f"Error saving table as PNG: {str(e)}")
          print("Table structure:")
          print(table)

  # Function to create hyperparameter table data
  def create_hyperparameter_table(self, results):
      hyperparameter_table_data = []
      for i, result in enumerate(results, 1):
          row = [f"Fold {i}"] + [str(result['Best Params'].get(param, '')) for param in results[0]['Best Params'].keys()]
          hyperparameter_table_data.append(row)
      return hyperparameter_table_data

  def generate_metrics_table(self, fold_metrics, avg_train_metrics, avg_test_metrics, metric_names):
      table_data = []
      train_header = metric_names[:len(metric_names)//2]
      test_header = [f"{name} (Test)" for name in metric_names[len(metric_names)//2:]]
      header = ["Fold"] + train_header + test_header

      for i, fold_metrics_row in enumerate(fold_metrics):
          train_row = fold_metrics_row[:len(metric_names)//2]
          test_row = fold_metrics_row[len(metric_names)//2:]
          row = [f"Fold {i+1}"] + train_row + test_row
          table_data.append(row)

      # Convert avg_train_metrics and avg_test_metrics dictionaries to lists
      avg_train_row = [avg_train_metrics[metric] for metric in train_header]
      avg_test_row = [avg_test_metrics[metric] for metric in metric_names[len(metric_names)//2:]]
      avg_row = ["Average"] + avg_train_row + avg_test_row
      table_data.append(avg_row)

      # Correct the column alignment here
      table = tabulate(table_data, headers=header, tablefmt="grid", colalign=("left",) * len(header))  # Align all columns to the left
      return table


  # Custom data loader
  def lightning_loader(self, X, y):
      dataset = TensorDataset(torch.from_numpy(X.values).float(), torch.from_numpy(y.values).long())
      loader = DataLoader(dataset)
      return loader


  def get_data_loaders(self, df_ZCR, cv_strat_data, shuffle=False):
      feature_names = df_ZCR.drop(['label', 'key', 'amp'], axis=1).columns.tolist()
      df_ZCR_X = df_ZCR.drop(['label', 'key', 'amp'], axis=1).values
      df_ZCR_y = df_ZCR['label'].values

      # Check for NaN values in the DataFrame before splitting
      if np.isnan(df_ZCR_X).any():
          print("Columns with NaN values before splitting:")
          nan_counts = np.isnan(df_ZCR_X).sum(axis=0)
          nan_table = pd.DataFrame({'Column': df_ZCR.drop(['label', 'key', 'amp'], axis=1).columns[nan_counts > 0], 'NaN Count': nan_counts[nan_counts > 0]})
          print(tabulate(nan_table, headers='keys', tablefmt='grid'))
      else:
          print("There are no NaN values in any column of the DataFrame before splitting.")

      table_data = []  # List to store data for the table

      for fold, (train_idx, test_idx) in enumerate(cv_strat_data):
          X_train = df_ZCR_X[train_idx]
          y_train = df_ZCR_y[train_idx]
          X_test = df_ZCR_X[test_idx]
          y_test = df_ZCR_y[test_idx]

          # Check for mismatched column names between X_train and X_test
          mismatched_columns = False

          # Calculate class balance
          train_class_balance = np.bincount(y_train, minlength=2)
          test_class_balance = np.bincount(y_test, minlength=2)

          # Append the data to the table_data list
          table_data.append([
              f"Fold {fold+1:>7}",
              str(X_train.shape).ljust(16),
              f"0: {train_class_balance[0]:>5}, 1: {train_class_balance[1]:>5}",
              str(y_train.shape).ljust(16),
              f"0: {train_class_balance[0]:>5}, 1: {train_class_balance[1]:>5}",
              str(X_test.shape).ljust(16),
              f"0: {test_class_balance[0]:>5}, 1: {test_class_balance[1]:>5}",
              str(y_test.shape).ljust(16),
              f"0: {test_class_balance[0]:>5}, 1: {test_class_balance[1]:>5}",
              "Mismatched columns" if mismatched_columns else ""
          ])

      # Define headers for the table
      headers = [
          "Fold",
          "X_train Shape", "X_train Class Balance",
          "y_train Shape", "y_train Class Balance",
          "X_test Shape", "X_test Class Balance",
          "y_test Shape", "y_test Class Balance",
          "Column Mismatch"
      ]

      # Print the table
      print(tabulate(table_data, headers=headers, tablefmt="grid"))

      return df_ZCR_X, df_ZCR_y, feature_names

  # Metrics
  def get_metrics(self, y_true, y_pred, y_pred_proba):
      conf_mat = confusion_matrix(y_true, y_pred)
      tn, fp, fn, tp = conf_mat.ravel()
      specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
      npv = tn / (tn + fn) if (tn + fn) != 0 else 0
      ppv = tp / (tp + fp) if (tp + fp) != 0 else 0

      metrics = {
          'Accuracy': accuracy_score(y_true, y_pred),
          'AUC': roc_auc_score(y_true, y_pred_proba),
          'Recall': recall_score(y_true, y_pred),
          'Specificity': specificity,
          'F1': f1_score(y_true, y_pred),
          'Precision': precision_score(y_true, y_pred),
          'NPV': npv,
          'PPV': ppv,
          'MSE': mean_squared_error(y_true, y_pred),
          'Micro F1': f1_score(y_true, y_pred, average='micro'),
          'Macro F1': f1_score(y_true, y_pred, average='macro')
      }

    # Calculate 95% confidence intervals
      confidence = 0.95
      n = len(y_true)  # Sample size
      for metric_name in ['Accuracy', 'AUC', 'Recall', 'Specificity', 'F1', 'Precision', 'NPV', 'PPV']:
          z_score = stats.norm.ppf((1 + confidence) / 2)
          std_error = np.sqrt((metrics[metric_name] * (1 - metrics[metric_name])) / n)
          margin_of_error = z_score * std_error

          # Store CI as a tuple directly with the metric value
          metrics[metric_name] = (metrics[metric_name], (metrics[metric_name] - margin_of_error, metrics[metric_name] + margin_of_error))

      return metrics, conf_mat
