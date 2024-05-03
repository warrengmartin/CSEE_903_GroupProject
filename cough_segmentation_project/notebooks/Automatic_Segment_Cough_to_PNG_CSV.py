import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
import csv
from tqdm import tqdm
import shutil
import zipfile

sys.path.append(os.path.abspath('../src'))

all_timecodes = []

def segment_cough(x, fs, mel_spec_db, cough_padding=0.09, min_cough_len=0.02, th_l_multiplier=0.1, th_h_multiplier=1):
    mel_energy = np.sum(mel_spec_db, axis=0)
    mel_intensity = np.max(mel_spec_db, axis=0)

    # Combine original audio signal with Mel spectrogram features
    x_combined = np.concatenate((x, mel_energy, mel_intensity))

    # Rest of the cough detection algorithm remains the same
    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    # Segment coughs
    coughSegments = []
    cough_starts = []
    cough_ends = []
    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    for i, sample in enumerate(x ** 2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end + 1])
                        cough_mask[cough_start:cough_end + 1] = True
                        cough_ends.append(cough_end)
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end + 1])
                    cough_ends.append(cough_end)
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True
                cough_starts.append(cough_start)

    # Convert start and end indices to times in seconds and format with 3 decimal points
    timecodes = [f"{start / fs:.3f}, {end / fs:.3f}" for start, end in zip(cough_starts, cough_ends)]

    return coughSegments, cough_mask, timecodes


def compute_mel_spectrogram(x, sr):
    # Convert to mono if the input is stereo
    if len(x.shape) > 1:
        x = librosa.to_mono(x)

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

    # Convert to decibel scale
    mel_spec_db = librosa.power_to_db(S=mel_spec, ref=np.max)

    return mel_spec_db

def plot_segments(x, cough_mask, cough_segments, file_id):
    plt.figure(figsize=(10, 4))
    plt.plot(x)
    plt.plot(cough_mask, color='r')
    plt.title("Segmentation Output")
    
    # Define the output directory and create it if it doesn't exist
    output_dir = "/kaggle/working/output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as an image in the output directory
    image_filename = f"{file_id}_segmentation_output.png"
    image_path = os.path.join(output_dir, image_filename)
    plt.savefig(image_path)
    plt.close()
    
    return image_path

import urllib.parse
from pathlib import Path

def save_timecodes_to_csv(file_id, timecodes, image_path, output_file='timecodes.csv'):
    # Define the output directory and create it if it doesn't exist
    output_dir = "/kaggle/working/output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Update the output file path to include the output directory
    output_file_path = os.path.join(output_dir, output_file)
    
    # Convert the image path to a Google Drive URL
    image_file_id = Path(image_path).name.split('_')[0]
    g_drive_url = f"https://drive.google.com/open?id={image_file_id}"
    
    # Write to CSV with the Google Drive URL and local file path
    with open(output_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file_id] + timecodes + [g_drive_url, f"file:///Users/wgm/Downloads/output_files/{image_file_id}.png"])


# Set output directory
output_dir = "/kaggle/working/output_files"
os.makedirs(output_dir, exist_ok=True)

# List of file IDs
file_ids = [
    "0f8d80f9-700e-4069-924d-e15f29d7c19a",
    "3278e364-afdd-4c39-9126-b8b4a4caf1c5",
    "6462d540-48bc-410b-b08e-cdf0b45ee118",
    "7876c549-066a-4ea1-a82e-45772114f964",
    "ccdb32af-58f4-4718-9f6e-eec7e0020382",
    "d9b4a30f-682d-4889-b65f-b359f4ebc74d",
    "de543d13-541c-4ad7-bb3c-c5c302de3aaf",
    "f0da0dd1-a4d1-47ac-bc31-e9c6ceb05e28",
    "01567151-7bb2-45ee-9aa8-a1332b5941ea",
    "01ff40e8-63e6-4570-a463-9778ea30cad7",
    "0f8fb3e0-1a30-4bd3-982a-24342a0bdc70",
    "1dd3b212-e969-4ede-a9d9-f24b711e2028",
    "1ed9491a-4036-4308-bc44-5036fc2e9f28",
    "218f522f-d3b0-4370-a93f-e8a70d958950",
    "2cc2fd2e-6314-424a-977b-7237f935fb65",
    "2e4456dd-bb47-45f7-b2ef-7b2d2f2859c6",
    "2f651988-a4c4-4042-a123-f0cc68a961a5",
    "314cd5c3-0030-4ea1-83b3-d6845897b903",
    "31e5a9cf-6a3e-43a3-8e0f-65c99f5748e9",
    "3733c427-2c04-48ee-8e8d-82f38a184b04",
    "37ac823b-4e39-4e85-a6c1-5e1e79979a13",
    "4260e6fd-0a12-48de-9bfd-1b8644c2186b",
    "481443c9-17a9-4194-9525-8c5fc2cd15fc",
    "48b287c0-5c2b-44ef-b469-3808b3b287ea",
    "49cc8b94-838b-4b23-9fb1-8d9384520634",
    "4c2840ed-fd4b-4a86-b976-5cc040d6ea4e",
    "4dba41ae-c9c3-4bf3-a5fb-8d8d2b8809c9",
    "510640bb-d6db-483f-a754-c5fa1eaaa388",
    "51d4bc80-1f02-4ffa-8e2e-1dc3527a0979",
    "52f6838a-4033-4a71-84bb-81cbb4e64ab7",
    "56994aa1-4626-4782-9f27-f125adfd120a",
    "5b7312df-3d0b-4a64-8884-e9072acceb41",
    "5be10c89-4c9a-495e-8671-f445f51a3fe0",
    "5c53c446-4d42-4dcf-9a2a-3c8c4a7682e1",
    "5f5f4027-45d4-4867-a9e2-ecc500eaf21d",
    "620ded24-220f-4ada-b032-2b5c170b279a",
    "637d6498-1d52-4860-9976-5a7a94f2a0c3",
    "651341cb-6be7-4981-b3e0-a454dee3af64",
    "69317fad-cc19-43fe-8537-782fc415ee7c",
    "6dff10f3-5df0-4b5a-be62-613afd6115dc",
    "7ae1ffe1-2259-411f-8ead-6c107e01e824",
    "818197ee-acaf-48e1-8776-1c70942a4ced",
    "824a8cf9-1199-48bf-aad5-65539d6d2011",
    "8254b396-cceb-4644-a723-5747cade8cf9",
    "88571a5b-7d25-4373-86f8-f784dbceabec",
    "89c97828-25b3-4957-8f2c-8da426e68d74",
    "8b733837-961a-4f17-a18c-13b89b54ef69",
    "8f5531fb-5c2a-450b-b14c-4321085887d8",
    "922fd4da-e8ce-4ef5-b9a8-56921dfa9328",
    "92b22064-50ef-4244-b5fd-aa348ceb7371",
    "9333c624-ef59-44f6-9643-fa9b2485c173",
    "98fb6294-d339-4c83-8ff5-2bbcf82e35e0",
    "99d322ed-c367-4d45-b6d1-f008b47f1af9",
    "9cb306f5-47a1-441c-888a-34ad29f5983c",
    "a75adc66-7f70-4f0f-91f7-7a2a2eac61e9",
    "aa059172-1850-4afe-8809-2fe79603286a",
    "ada1def9-7e36-4d4e-8e3e-e678afee4185",
    "ae18328e-da30-4137-a717-a718ffe2412e",
    "b010f28f-e601-49de-9293-fb65fdad2dd4",
    "b05935c3-68d7-4a28-aea9-f92fbe295c6c",
    "b618923d-796d-4adb-ae9e-266ab08f1249",
    "bae0c88f-852c-4f3f-8141-d037c777459e",
    "bb9766af-74b9-4533-ac93-e0d75f7bd7ba",
    "bcf8e484-3423-4654-83e5-8188ef14e73f",
    "be48c56b-4a96-43f7-b45c-46d9c0dfd909",
    "c3382b37-9f19-4379-b488-68f636a6c684",
    "c3b9dca7-67eb-4438-b6aa-dd6b2aca1432",
    "c941e62c-e260-4ee9-959e-af91663ac58c",
    "c9fa0d4d-76d1-41c4-863c-b83bb8908140",
    "cb89e13c-f7b9-4277-8529-85385032e6b2",
    "d5a1411d-d312-44b1-a37d-a387ec13df1a",
    "d769f90f-8183-4c42-8ea6-cf992ee40a64",
    "d7e71034-14a0-4ebb-870c-11bffb7d80d4",
    "d9936aee-41cf-4917-a02e-99b76db504c9",
    "dddb1a55-d976-40bc-ad41-cdc713623e03",
    "e1b9d3d1-61f8-4957-923b-e2715d7fabb1",
    "e56b40e6-935c-41c6-a04f-cfa133d69cec",
    "e56ea219-e97c-45c4-8222-5ecf168445b5",
    "e6448323-4be3-4fac-ba98-33aee0de0817",
    "e9bf0a5a-ff96-480b-a54a-f647143e1d6f",
    "eceaa3a9-f850-4ba9-82ee-1bcd2575b99c",
    "f4eaf2ba-b2e3-4755-893e-6ddb786189c9",
    "f504d41c-13e5-4d7a-9a91-0436aab969cf",
    "f5e8d8ec-e234-4117-a4ab-2f7dba0066ef",
    "f67d8450-f76e-4c47-bbbe-85418e320acb",
    "fd7d172d-4106-427a-870b-0384c88f147f",
    "fed255ec-4829-4f4a-b22d-9bb23f2dd89f",
    "ff1234d7-7837-4ba7-842f-99fdc916baa9",
    "03d30b8b-03f8-4089-bda8-0e14deec7d82",
    "21aee478-6d13-45ea-be4d-4f29fd244798",
    "22cb791b-2eba-480f-9eb3-69018bd25a04",
    "25b750e5-8a76-4c13-9fd8-851e45d1b5ed",
    "30b282bb-affd-4449-81b3-f5bcc8877bee",
    "37347587-048a-4a1c-a100-7cf820711428",
    "3c6cfda1-3fe9-403d-b5d1-7797420debe2",
    "493632f2-537a-4a36-a932-268a8aa59486",
    "4c58cf95-9b2e-4091-adaa-0ac06a355c48",
    "55c0a763-5f42-476e-9c72-e2924b21b2b4",
    "639c6225-edd2-4f9c-8c7f-5fedd57ef73f",
    "64da4ad6-4280-49a1-bd37-e35efd7837c6",
    "6c9a5f33-9e90-4c08-8782-20b0d86abc46",
    "70dbb19c-8c7a-45d4-9d2a-cd2272cc7ccb",
    "70ea9638-f7e3-4f16-bb97-7d066a0e6cdf",
    "74536842-b3dc-4f8f-b232-0333419d7ca2",
    "760cb050-4175-4498-817d-a9976e359c52",
    "7c0c7e95-4a6b-4d75-b46c-6cb8cf280094",
    "7cd1efb7-e973-492d-8a2c-e00a42ef0d78",
    "83d26908-f355-426a-b5dc-92dee890482d",
    "8956ab9c-a0ca-4f2e-ae00-1f5d0065d50b",
    "92307087-a106-4fd5-919a-a92b57ffd316",
    "9821da28-7655-4736-abf8-4c89582e407f",
    "a1f69db9-81f4-4a07-9808-dd1c99053125",
    "a4cc4680-8bb6-4646-b9cf-d77a4e8ada21",
    "a5aeaa95-ed25-452b-b112-62c042c21b33",
    "aa78817e-f5e3-470b-b516-6006c59fb1fa",
    "abbff660-5d13-4c31-b842-bbf2189ff14e",
    "b73530b6-f66b-4599-af9a-4066253bae7f",
    "c1371e45-804a-42fc-a3f9-5e66d4dc0a23",
    "c2fc880a-7b05-4c36-affe-5ffe03dbc2b4",
    "c5a4c854-9d78-4615-9fdd-519e547e972f",
    "c67bffd4-1620-4f80-9b40-5661d595a5a0",
    "d2a55629-24e2-4460-8a0a-b7065927ee07",
    "e08e7dcf-5219-4eb7-9d74-c5605830db06",
    "e4729edf-12e5-4ba9-883e-1114c32b90f0",
    "ea15c9f5-e77d-4b3f-803d-443a69d63eed",
    "eade4e05-0fe3-4736-8a9c-35467d078113",
    "f5a661dc-8161-4842-b1c3-cd7265896101",
    "f942b766-ef1e-42e7-826d-40e053414824",
    "fb0971e2-8ebf-459c-972d-b09d28ae0ca6",
    "fd849b72-f4bf-4852-9bc2-fd9becc9571e",
    "01614a4a-947f-43ca-a609-fc2787509f6f",
    "08f05aa0-7c3d-40be-a8ae-ab6bc6d41be3",
    "09115490-33a9-461c-9437-d7f71be057b0",
    "0b2f75d7-f116-4f35-ae4c-f2018eab2794",
    "0d930dca-d29e-4b69-a211-79ce69081c97",
    "0dcae895-e7c6-4e47-a399-29f22b37e075",
    "16b16ff5-e875-4f0f-ad76-bd7ba27faca6",
    "3481b694-34fe-4b03-a529-88ef085815fd",
    "409c0b45-5adb-408b-adfd-94f7ffef6fc9",
    "47eee8d8-e1ac-49a7-8059-ad702620309a",
    "494fdf0a-fc66-4292-acc4-7a69df02405f",
    "52339446-0c26-4d66-a776-9894b8bd196f",
    "5280c666-4f3e-4d7c-a283-19041fc4cb2b",
    "5542e025-cec0-48d8-8ac2-541975793b94",
    "57263117-e5ef-4d4c-b678-cec8095c7e0d",
    "0029d048-898a-4c70-89c7-0815cdcf7391",
    "008ba489-31ad-44d8-856b-fcf72369dc46",
    "008c1c9e-aeef-40c5-846c-24f1b964f884",
    "00bf9f83-2e8f-47cf-a4f2-97f2beceebc1",
    "01f97898-9ec4-480b-9fcc-a9daa737106d",
    "0527be95-d7f1-4156-8e37-1587355661ca",
    "0640716b-e287-4181-a653-5b798e8308c8",
    "0969d0c4-34ce-4e9a-8cf1-1b18403587e8",
    "09de6967-b295-4516-8a4d-4d95c9a7b02c",
    "09e64861-61b1-4f66-a7c0-17ac1a0c60f0",
    "0a03da19-eb19-4f51-9860-78ad95fa8cb5",
    "0dfa1560-9b26-4474-a9d9-d96c05424fbc",
    "11c1e2d8-52e2-4e32-8ab8-b858268e0ca9",
    "1357788c-fd94-45ff-b31d-da0cda859731",
    "13cc16ff-2b44-4740-b7f4-e80cf7404d30",
    "14a3c119-bee8-435a-9193-d2ce7cb84a03",
    "182246b0-9f77-4c40-976e-3c9342619819",
    "191dca0e-7951-4aa3-9764-2761bd511955",
    "1c8853d4-5a0b-425e-b211-d20b2d16e35d",
    "00ce5b06-c302-4387-bbd7-86355a4a8c12",
    "005b8518-03ba-4bf5-86d2-005541442357",
    "006d8d1c-2bf6-46a6-8ef2-1823898a4733",
    "06b568b5-b9f8-4334-816c-c16009bb5de7",
    "1e3025cb-d921-485b-a08c-c2966fc4f730",
    "2e349a58-b119-4b15-8b3f-5a5c6416bca4",
    "459c949d-1074-4687-a911-e1b61753643c",
    "4f95a31d-9302-47bb-a0b6-cdd8b13c0aab",
    "562c3783-d5ac-4d30-90c6-054fbc83dbff",
    "66636b7b-3e0e-4613-9f1f-4ec16fb313c3",
    "706b1acc-0e61-499e-abb1-43de015eef46",
    "78637ec8-6570-4b6a-b8fd-a1610022c413",
    "88a9b00f-205e-4fbc-9cda-1793f2c7600b",
    "8c9122bd-5601-48aa-865a-63db152a91df",
    "9a80a95e-6d74-4d45-bfd0-62c3eb40471a",
    "9aad0bb1-3d4e-430b-964f-2a8b27a20143",
    "bb825a76-02ce-4a02-a001-7ae1a05d5bb8",
    "e247085a-cf34-45be-b3c1-3c609c7d2bf1",
    "018b40a1-c109-459a-9e31-86cbd2cb3918",
    "0569d979-384b-4a30-b0ca-2b19e8c8650b",
    "18722fbe-d240-457a-a0fc-2d9d09bfbcc8",
    "21c16c1c-46fc-4b80-b941-65d7c6e87555",
    "2295a6f8-08d5-467f-8534-8329c9083f28",
    "28d4e487-0d9d-4911-951d-5de7fcb5c986",
    "2cc30bb4-6e0f-4cb5-a7a5-f7fa03c203ba",
    "2e9f905a-697c-48db-883e-7e905873172f",
    "4735aed3-46f9-44c5-91f0-ca772c541fff",
    "56da1f04-fe81-4928-a1de-0f3b9d333c12",
    "6647b629-2246-48c9-83ad-c3ad4795c891",
    "71d4e0e2-99b1-49bc-990f-1f7b4b5de826",
    "7d1428e9-7241-482b-8dbd-95f43a57c694",
    "85950027-a5fd-4cd4-ad38-46202aa61172",
    "87afcc54-2b3c-45c1-a4c7-2e68a8e30280",
    "01820f7c-b953-4faf-aa13-978cfda6b08e",
    "02aa80ef-a83b-477f-b01d-575651364b22",
    "03c8796e-547e-4158-ae5f-7c613aaeb02f"
]



# Process each file with tqdm progress bar
for file_id in tqdm(file_ids):
    file_path = f"/kaggle/input/audio-file-dataset-roneel/{file_id}.wav"
    x, fs = librosa.load(file_path, sr=None)
    
    # Compute Mel spectrogram
    mel_spec_db = compute_mel_spectrogram(x, fs)
    
    # Call segment_cough with Mel spectrogram
    cough_segments, cough_mask, timecodes = segment_cough(x, fs, mel_spec_db)
    
    # Append timecodes to the list
    all_timecodes.append(timecodes)
    
    # Plot the segments and get the image filename
    image_filename = plot_segments(x, cough_mask, cough_segments, file_id)
    
    # Save timecodes to CSV, including the image path
    save_timecodes_to_csv(file_id, timecodes, image_filename)

# Set compression level (1 to 9, where 9 is the highest and no compression)
compression_level = 8

# Create a zip file with the specified compression level
with zipfile.ZipFile(os.path.join(output_dir, "output_files.zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.setpassword(b'password')
    zipf.writestr('timecodes.csv', '\n'.join([','.join(timecodes) for timecodes in tqdm(all_timecodes)]))
    for file_id in tqdm(file_ids):
        img_path = f"{file_id}_segmentation_output.png"
        img_data = open(os.path.join(output_dir, img_path), 'rb').read()
        zipf.writestr(img_path, img_data)
