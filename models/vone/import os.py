import os
import shutil

# Set the source and target directories
source_dir = '/home/wcortez/Documents/tiny-imagenet-200/train'
in_path = '/home/wcortez/Documents/tiny-imagenet-200'
target_dir_base = '/home/wcortez/SurroundSuppression/Surround-Suppresion-for-VOneNet'

# Ensure the base target directory exists
if not os.path.exists(target_dir_base):
    os.makedirs(target_dir_base)



# Load wnids and class names
id_dict = {}
number_lines_wndids = 0
with open(os.path.join(in_path, 'wnids.txt'), 'r') as f:
    for index, line in enumerate(f):
        id_dict[index] = line.strip()
        number_lines_wndids += 1

print("Number of lines in wnids.txt: ", number_lines_wndids)

class_names = {}
number_lins_wrds = 0
with open(os.path.join(in_path, 'words.txt'), 'r') as f: 
    for line in f:
        wnid, label = line.split('\t')
        class_names[wnid] = label.strip()
        number_lins_wrds += 1

print("Number of lines in words.txt: ", number_lins_wrds)

# Define your search terms and find relevant wnids
search_terms = [' cat', 'teapot']
relevant_wnids = {term: [] for term in search_terms}
for wnid, label in class_names.items():
    for term in search_terms:
        if term in label:
            print(f"Found {term} in {label} (wnid: {wnid}")
            relevant_wnids[term].append(wnid)

# Extract images based on the relevant wnids and organize by search term
def extract_and_organize_images(source_dir, base_target_dir, wnids_dict):
    for term, wnids in wnids_dict.items():
        term_target_dir = os.path.join(base_target_dir, term)
        if not os.path.exists(term_target_dir):
            os.makedirs(term_target_dir)
        
        for wnid in wnids:
            class_dir = os.path.join(source_dir, wnid, 'images')
            if os.path.exists(class_dir): 
                for filename in os.listdir(class_dir):
                    source_path = os.path.join(class_dir, filename)
                    target_path = os.path.join(term_target_dir, f"{wnid}_{filename}")
                    shutil.copy(source_path, target_path)


# Execute the extraction and organization
extract_and_organize_images(source_dir, target_dir_base, relevant_wnids)
