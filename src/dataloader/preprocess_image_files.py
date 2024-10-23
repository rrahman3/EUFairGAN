import shutil
import os
import pandas as pd

train_df = pd.read_csv('../data/nihcc_chest_xray/nihcc_chest_xray_training_samples.csv')

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
       'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
       'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
       'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous_Emphysema',
       'Tortuous_Aorta', 'Calcification_of_the_Aorta', 'No_Finding']

train_df = train_df.rename(columns={'Subcutaneous Emphysema':'Subcutaneous_Emphysema', 'Pleural Thickening': 'Pleural_Thickening', 'SubcutaneousEmphysema': 'Subcutaneous_Emphysema', 'Tortuous Aorta':'Tortuous_Aorta', 'Calcification of the Aorta':'Calcification_of_the_Aorta', 'No Finding':'No_Finding'})

src_dir = "data/nihcc_chest_xray/xray_images"
dst_dir = "data/nihcc_chest_xray/images/train"
os.makedirs("data/nihcc_chest_xray/images/")
os.makedirs("data/nihcc_chest_xray/images/train/")
for class_name in classes:
    class_dir = os.path.join(dst_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
def find_classes(row):
    class_lists = []
    for class_name in classes:
        if row[class_name]:
            class_lists.append(class_name)
        
    return class_lists

for idx, row in train_df.iterrows():
    # print(row)
    img_file = row['id']
    class_lables = find_classes(row)
    source_img_path = os.path.join(src_dir, img_file)
    if os.path.exists(source_img_path):
        for class_name in class_lables:
            target_img_path = os.path.join(dst_dir, class_name, img_file)
            shutil.copy(source_img_path, target_img_path)
        print(idx, source_img_path, target_img_path)