import numpy as np
import pandas as pd

file_path = 'pointcloud.npy'
to_file = 'pointcloud.csv'

print(f"Converting {file_path} to {to_file}, lowering the data size from 64*64*64 to 64*4096.")

array = np.load(file_path)
reshaped_array = array.reshape(64, -1)
df = pd.DataFrame(reshaped_array)

df.to_csv(to_file, index=False, header=False)

print(f"Converted and stored.")