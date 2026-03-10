import os

folder_path = "C:/Users/Alian Khan Gandapur/Desktop/Experiment/Shapes_last"
counter = 26454

# Get a list of file names with full paths sorted by modified timestamp
file_names = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))], key=os.path.getmtime)

for file_name in file_names:
    _, ext = os.path.splitext(file_name)
    if ext.lower() in (".jpg", ".jpeg", ".png", ".gif"):
        new_file_name = f"{counter:04d}{ext}"
        os.rename(file_name, os.path.join(folder_path, new_file_name))
        counter += 1
