import shutil
import os
import pandas as pd

class CopyImages:
    def __init__(self, metadata_file, src_dir, dst_dir):
        self.metadata_file = metadata_file
        self.train_df = pd.read_csv(metadata_file)

        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
            'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous_Emphysema',
            'Tortuous_Aorta', 'Calcification_of_the_Aorta', 'No_Finding']

        self.train_df = self.train_df.rename(columns={'Subcutaneous Emphysema':'Subcutaneous_Emphysema', 
                                                      'Pleural Thickening': 'Pleural_Thickening', 
                                                      'SubcutaneousEmphysema': 'Subcutaneous_Emphysema', 
                                                      'Tortuous Aorta':'Tortuous_Aorta', 
                                                      'Calcification of the Aorta':'Calcification_of_the_Aorta', 
                                                      'No Finding':'No_Finding'})

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        os.makedirs(self.dst_dir, exist_ok=True)
        for class_name in self.classes:
            class_dir = os.path.join(dst_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
                
    def find_classes(self, row):
        class_lists = []
        for class_name in self.classes:
            if row[class_name]:
                class_lists.append(class_name)
            
        return class_lists

    def copy_files(self):
        for idx, row in self.train_df.iterrows():
            # print(row)
            img_file = row['id']
            class_lables = self.find_classes(row)
            source_img_path = os.path.join(self.src_dir, img_file)

            if os.path.exists(source_img_path):
                for class_name in class_lables:
                    target_img_path = os.path.join(self.dst_dir, class_name, img_file)
                    if os.path.exists(target_img_path):
                        print("Already destination file exists.")
                        continue
                    shutil.copy(source_img_path, target_img_path)
                print(idx, source_img_path, target_img_path)

if __name__=="__main__":
    # train = CopyImages(metadata_file='data/nihcc_chest_xray/nihcc_chest_xray_training_samples.csv',
    #             src_dir="data/nihcc_chest_xray/xray_images",
    #             dst_dir="data/nihcc_chest_xray/images/train",
    #         )
    # train.copy_files()

    val = CopyImages(metadata_file='data/nihcc_chest_xray/nihcc_chest_xray_validation_samples.csv',
                src_dir="data/nihcc_chest_xray/xray_images",
                dst_dir="data/nihcc_chest_xray/images/val",
            )
    val.copy_files()
