import os
import shutil

base_dir = "/home/amirthan/PycharmProjects/signlanguage_using_landmark/MyApp/ASL_Alphabet_Dataset/asl_alphabet_train"
output_dir = "../model_library/Datasets/Dataset-1/Labels"

# Create the Labels folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each subfolder
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    csv_filename = f"{folder}_labels.csv"
    csv_path = os.path.join(folder_path, csv_filename)

    # Check if CSV exists
    if os.path.isfile(csv_path):
        dest_path = os.path.join(output_dir, csv_filename)
        shutil.copy(csv_path, dest_path)
        print(f"✅ Copied: {csv_filename}")
    else:
        print(f"⚠️ Skipped: {csv_filename} not found")
