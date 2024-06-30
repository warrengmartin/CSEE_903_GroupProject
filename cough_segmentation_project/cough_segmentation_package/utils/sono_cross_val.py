from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

class CrossValSplit:
    """
    A class to perform cross-validation on audio datasets, ensuring that all frames from each recording 
    are retained within specific sets (train/test).

    Attributes:
        df (pd.DataFrame): The DataFrame containing the audio dataset.
        non_cough_keys (list): List of audio file names without cough sound.
        audio_file_column_name (str): The column name in the DataFrame representing the audio file keys.
    """

    def __init__(self, df_single_frame, non_cough_keys, audio_file_column_name='key'):
        """
        Initializes the CrossValSplit with the dataset and audio file column name.

        Parameters:
            df_single_frame (pd.DataFrame): The DataFrame containing the audio dataset.
            non_cough_keys (list): List of audio file names without cough sound.
            audio_file_column_name (str): The column name in the DataFrame representing the audio file keys.
                                          Defaults to 'key'.
        """
        self.set_df(df_single_frame, non_cough_keys, audio_file_column_name)
    
    def set_df(self, df, non_cough_keys, audio_file_column_name):
        """
        Sets the DataFrame and audio file column name for the class.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the audio dataset.
            non_cough_keys (list): List of audio file names without cough sound.
            audio_file_column_name (str): The column name in the DataFrame representing the audio file keys.
        """
        self.df = df
        self.audio_file_column_name = audio_file_column_name
        self.non_cough_keys = non_cough_keys

    def get_audio_files(self):
        """
        Retrieves a DataFrame of unique audio files and their counts from the dataset.

        Returns:
            pd.DataFrame: A DataFrame with unique audio files and their corresponding counts, sorted in descending order.
        """
        unique_df = pd.DataFrame(np.unique(self.df[self.audio_file_column_name].values, return_counts=True)).T
        
        # Add labels to unique recordings (cough vs non-cough)
        unique_df[2] = 1  # Initialize with label for cough
        unique_df.loc[unique_df[0].isin(self.non_cough_keys), 2] = 0  # Set non-cough label for specified keys

        return unique_df.sort_values(by=[1], ascending=False)
    
    def get_result(self):
      """Returns information of the resultof the cross validation"""
      return self.split_audio

    def cross_val(self, stratified=False, n_splits=5, shuffle=False, plot=False, show_fold_info=False):
        """
        Performs cross-validation on the audio dataset, ensuring that all frames from each recording 
        are retained within specific sets (train/test).

        Parameters:
            stratified (bool): Whether to use StratifiedKFold cross val. Defaults to False.
            n_splits (int): Number of folds for cross-validation. Defaults to 5.
            shuffle (bool): Whether to shuffle the data before splitting into batches. Defaults to False.
            plot (bool): Whether to plot the cross-validation splits. Defaults to False.
            show_fold_info (bool): Whether to show details of each fold of the cross validation process. Defaults to False.

        Returns:
            list: List of Tuples containing train and test indexes
        """
        self.split_audio = self.split_audio_files(stratified=stratified, n_splits=n_splits, shuffle=shuffle, plot=plot)
        cv_list = []
        for index, row in self.split_audio.iterrows():
            test_df = self.df[self.df[self.audio_file_column_name].isin(row["Test Data"][:, 0])]
            train_df = self.df[self.df[self.audio_file_column_name].isin(row["Train Data"][:, 0])]

            cv_list.append((train_df.index, test_df.index))
          
            if show_fold_info:
                print('Fold: ', index+1, 
                      '\n-Train:\tFrame Size (Dataset):', train_df[self.audio_file_column_name].count(), 
                      '\t\tFrame Size (CV Split):', row["Frames Train Size"], 
                      '\t\tAudio Files:', len(row["Train Data"]))
                
                print('-Test:\tFrame Size (Dataset):', test_df[self.audio_file_column_name].count(), 
                      '\t\tFrame Size (CV Split):', row["Frames Test Size"], 
                      '\t\tAudio Files:', len(row["Test Data"]),'\n')

        return cv_list
    
    def split_audio_files(self, stratified=False, n_splits=5, shuffle=False, plot=False):
        """
        Split audio files for cross-validation .

        Parameters:
            stratified (bool): Whether to use StratifiedKFold cross val. Defaults to False.
            n_splits (int): Number of folds for cross-validation. Defaults to 5.
            shuffle (bool): Whether to shuffle the data before splitting into batches. Defaults to False.
            plot (bool): Whether to plot the cross-validation splits. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing details of each fold, including train/test sizes and indices.
        """
        data = self.get_audio_files().values

        y = np.array(data[:, 2], dtype='float')
        X = np.delete(data, 2, axis=1)

        # KFold cross-validator
        if stratified:
            if shuffle:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=6)
            else:
                kf = StratifiedKFold(n_splits=n_splits)
        else:
            if shuffle:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=6)
            else:
                kf = KFold(n_splits=n_splits)
        
        kf.get_n_splits(X, y)
        print('Sono Cross Val Split\n', kf, '\tTotal Audio Files:', X.shape[0])

        # Initialize a list to store fold information
        fold_info = []

        if plot:
            # Visualization setup
            fig, ax = plt.subplots(figsize=(16, 6))

        # Iterate through each fold and collect the details
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            fold_details = {
                'Fold': i,
                'Audio Train Size': len(train_index),
                'Audio Test Size': len(test_index),
                'Cough Audio Train Size': np.sum(y[train_index]==1),
                'Cough Audio Test Size': np.sum(y[test_index]==1),
                'Frames Train Size': sum(X[train_index][:, 1]),
                'Frames Test Size': sum(X[test_index][:, 1]),
                'Total Frames': sum(X[test_index][:, 1]) + sum(X[train_index][:, 1]),
                'Train Index': train_index,
                'Test Index': test_index,
                'Train Data': X[train_index],
                'Test Data': X[test_index]
            }
            fold_info.append(fold_details)

            if plot:
                # Bar chart visualization
                for train_idx in train_index:
                    ax.barh(i, 1, left=train_idx, color='blue', edgecolor='k', alpha=0.5, label='Train' if i == 0 and train_idx == train_index[0] else "")
                for test_idx in test_index:
                    ax.barh(i, 1, left=test_idx, color='red', edgecolor='k', alpha=0.5, label='Test' if i == 0 and test_idx == test_index[0] else "")
                

        if plot:
          # Plot class labels
          for i,v in enumerate(y):
              ax.barh(n_splits, 1, left=i, color='green' if v == 0 else 'red', alpha=0.3)
            
          # Plot settings
          ax.set_xlabel('Audio Recording Index')
          ax.set_ylabel('Fold Index')
          ax.set_title('K-Fold Cross-Validation Splits')
          ax.set_yticks(range(n_splits+1))
          ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)] + ['Class'])
          ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
          plt.grid(True)
          plt.show()

        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(fold_info)
