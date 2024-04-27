# CSEE 903 Group Project 

# Project Title: Noise Robust Cough Sound Segmentation Using Audio Features and Machine Learning
- Project ID: 78427; 
- Supervisor: Roneel Sharan;
- Contact: roneel.sharan@essex.ac.uk

**Team Name:**
- Sono

**Project Documentation**
- [Google Drive](https://drive.google.com/drive/folders/1vPHf0wGuo_vNYT0-DNQPbkM_VB-gTpfA?usp=drive_link)

**Student registration number(s):**
- Max Garry – Team Leader: 2323118
- Warren Martin - Secretary: 232569

- Breanne Felton: 2321566
- RANSFORD OWUSU: 2321357
- Saurav Thakur: 2322997
- Patrick Ogbuitepu: 2320824

**Project Description:**
Cough is a common symptom of respiratory diseases and the sound of cough can be indicative of the respiratory condition. Objective cough sound analysis has the potential to aid the assessment of the respiratory condition. Cough sound segmentation is an important step in objective cough sound analysis. This project focuses on developing methods to segment cough sounds from audio recordings. The students will explore various signal processing techniques, such temporal and spectral analysis, to identify distinctive features of coughs. Through machine learning algorithms, they'll create a classification model capable of distinguishing cough sounds from non-cough sounds. The students will also develop cough sound denoising algorithms for cough sound segmentation in noisy environments. Refer to https://doi.org/10.1016/j.bspc.2023.104580 and https://doi.org/10.1109/EMBC40787.2023.10340687 for more information.


# Getting Started: Github Usage Strategy
### 1. Login to Google Drive and Open the Jupyter Notebook File for the Project

#### Justification
Alternatively, the notebook could be loaded directly from GitHub, but the auto-save feature on Colab will be inactive. This could lead to a loss of changes if the user's browser crashes. To mitigate this scenario, we shall maintain only our notebook file in the Google Drive repository.

### 2. Login to GitHub on a New Browser Tab

### 3. Import Project Files from GitHub

- **Run the section at the top of the notebook titled "Load Project Files from GitHub"**:
  - The initial code will try to authenticate your GitHub credentials to download the project resources (other codes, data, etc.).
  - You’ll be prompted to enter your email address and personal access token.
  - Then the project files in GitHub will be automatically copied to the Colab environment.

### 4. Work (Write or Run Codes)

- Guidelines on writing codes will be included to ensure compliance with minimum standards and ease of readability, reuse, & collaboration.
- The notebook automatically saves to Google Drive when working.

### 5. Saving Your Work

#### Save to GitHub

- **Run the codes at the bottom of the notebook titled "Save to GitHub"**:
  - You will receive a prompt to input your email and token if it was not already provided at the first section of the notebook.
  - You will be prompted to enter a description of the feature that you are working on. 
    - Describe the feature properly. E.g., "Machine Learning Model - Random Forest Classifier" or "Feature Extraction: Principal Component Analysis."
  - The code will backup the notebook file and save all modified files to GitHub.

#### Save the Notebook to GitHub

- In the Colab notebook, click on **File** in the top menu.
- Select **Save a copy in GitHub**.
- You will be prompted to authorize Google Colab to access your GitHub account. Click on the link provided to authorize the access. Ensure to click the checkbox for the private repo.
- After authorization, a dialog box will appear where you can select the repository where you want to save the notebook.
  - Choose the private repository you set up for this purpose.
  - Leave the branch as "main" where you want to save the notebook.
  - Enter the same commit message suffixed with "notebook," e.g., "Machine Learning Model - Random Forest Classifier - Notebook."
- Click on the **OK** button to save the notebook to your GitHub repository.
- Colab will push the notebook to the specified path in your GitHub repository.

### 6. More Info

- [Check out this slideshow by Martin Fowler](https://martinfowler.com/articles/continuousIntegration.html#BuildingAFeatureWithContinuousIntegration)

