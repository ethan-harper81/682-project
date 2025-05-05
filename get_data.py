import numpy as np
import os
import shutil
import zipfile
import csv


indices = [
    0, 1, 3, 7, 10, 12, 15, 16, 21, 23, 24, 25, 26, 27, 30, 31, 36,38, 39
]


'''
Need to extract folder and remove "Medical Imaging Dataset" from its parent directory
Then run to get data
'''
path = "Medical Imaging Dataset"

os.makedirs("data_set", exist_ok=True)

#Store Original name, split to train-test-val?d

disease_names = []

print(os.walk(path))
for root, dirs, files in os.walk(path):
    print(root)
    print(dirs)
    j = 0
    for i, file in enumerate(files):
        
        
        if i in indices:
            print("file: ", file)
            rename = f"disease_{j}.zip"
            old = os.path.join(path,file)
            new = os.path.join(path, rename)
            os.rename(old, new)
            extract_to_path = os.path.join("data_set", new).replace(".zip", "")

            disease_names.append(file.removesuffix('.zip'))
            
            j+=1
            
            with zipfile.ZipFile(new, 'r') as zipref:
                zipref.extractall(extract_to_path)
            
            # for  r,d,fs in os.walk(extract_to_path):
            #     for r_,d_,fs_ in os.walk(os.path.join(root, d[0])):
            #         for f in fs_:
            #             shutil.move(os.path.join(root,f),extract_to_path)
        
        #os.remove(new)
        
'''
Need to re extract folder and run again with this to save labels
'''
with open('disease_names.csv', 'w', newline = '') as f :
    writer = csv.writer(f)
    for i, row in enumerate(disease_names):
        writer.writerow([i, row])

        
        
    
    

