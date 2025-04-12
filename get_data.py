import numpy as np
import os
import sys
import zipfile


indices = [
    0, 1, 3, 7, 10, 12, 15, 16, 21, 23, 24, 25, 26, 27, 30, 31, 36,38, 39
]

path = "Medical Imaging Dataset"

os.makedirs("data_set", exist_ok=True)

#Store Original name, split to train-test-val?d

print(os.walk(path))
for root, dirs, files in os.walk(path):
    print(root)
    print(dirs)
    for i, file in enumerate(files):
        rename = f"disease_{i}.zip"
        old = os.path.join(path,file)
        new = os.path.join(path, rename)
        os.rename(old, new)
        if i in indices:
            
            extract_to_path = os.path.join("data_set", new).replace(".zip", "")
            
            
            
            with zipfile.ZipFile(new, 'r') as zipref:
                zipref.extractall(extract_to_path)
        
        #os.remove(new)
        
        
        
        
    
    

