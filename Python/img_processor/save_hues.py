import os
import glob
from process_functions import *
from numpy import median
import json

#Get the current working directory
curr_dir = os.getcwd()
# Set output encoding for 1 on n (Y)
colors = { 'blue': {'encoding':[0,0,1]}, 'red': {'encoding':[0,1,0]}, 'green': {'encoding':[1,0,0]} }
img_data = []
already_used = []
for color in colors.keys():
    hues = []
    # Load jpg files from each color's folder
    # Changed to only jpg because png was causing issues in openCV
    for (i,image_file) in enumerate(glob.iglob(curr_dir+'/img_data/'+ color+'/*.jpg')):
        try:
            file_name = image_file.split('/')[-1]
            file_index = file_name.find('_')
            file_name = file_name[file_index+1::]
            # Check to see if there are duplicate jpg files
            if file_name in already_used:
                continue
            already_used.append(file_name)
            # Use process fuctions
            hsv_matrix = process(image_file, i)
            hue_only_matrix = hue_extract(hsv_matrix)
            for row in hue_only_matrix:
                for value in row:
                    hues.append(value)
            # Create a data entry to the database
            data_entry = {}
            data_entry['id'] = color[0] + str(i)
            data_entry['color'] = color
            data_entry['filename'] = image_file.split('/')[-1]
            data_entry['hue_matrix'] = hue_only_matrix
            data_entry['encoded_color'] = colors[color]['encoding']
            img_data.append(data_entry)
        except:
            print("Error: " + file_name)
            pass
    # Print results
    median_hue = median(hues)
    print(color +' | median: ' + str(median_hue))
    print(color +' | n: ' + str(i))
    # Save metadata
    colors[color]['n'] = i
    colors[color]['median_hue'] = median_hue

# Save database
data = { 'metadata': colors , 'img_data': img_data}
with open('final_data.json', 'w') as outfile:
    outfile.write(json.dumps(data, ensure_ascii=False))
    print('json saved')
