import os
import shutil

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

import librosa
import librosa.display

class SimpleInjestion:
  def copy_files_to_colab(self, fileName ):
    """
    Copies audio files from gdrive to colab server for fast processing

      Parameters:
          fileName (string): File Name

      Returns:
          fileName (str): File Name
    """
    contentPath = "/content/"
    dest = fileName

    fileName = contentPath + 'drive/My Drive/' + fileName
    if os.path.exists(fileName):
      dest = contentPath + fileName.split('/')[-1]
      print(f'{fileName} {dest}')
      #%cp "{fileName}" "{dest}"
      shutil.copy(fileName, dest)
      rfileName = dest
    else:
      print(f'File not found:{fileName}')
  
    return dest

  def copy_files(self, path, files):
    """
    Before running this code make sure that you have Audio Files Folder in the current directory
    and download Dataset workflow - Sheet1.csv in the current directory.

    Make sure the Dataset workflow - Sheet1.csv is the file that contains your name.

    And make sure you change your name.
    """
    filenames = []

    for j in os.listdir(os.path.join(path)):
        source = os.path.join(path,j)

        splitted_file_name = j.split(".")[0]

        # Check if the source file exists
        if splitted_file_name in files:
            # Copy the file
            shutil.copy(source, destination)
            destination_file_name = os.path.join(destination,j)

            #todo: verify that file exists in the destination directory
            filenames.append( destination_file_name )
            #print(f"File {splitted_file_name} copied successfully!")

    return filenames
