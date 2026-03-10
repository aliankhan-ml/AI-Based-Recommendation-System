import pandas as pd

# Load the Excel file into a pandas dataframe
df = pd.read_excel("C:/Users/Alian Khan Gandapur/Desktop/Experiment/Duplicate/Users/Alian Khan Gandapur/Desktop/Experiment/Shapes_last/same.xlsx", engine='openpyxl')

# Extract the column as a list
my_list = df.iloc[:, 0].tolist()

# Print the list to verify the output
print(my_list)

# Load the Excel file into a pandas dataframe
df = pd.read_excel("C:/Users/Alian Khan Gandapur/Desktop/Experiment/Shapes_last/patterns.xlsx", engine='openpyxl')

# Load the list of values to delete

# Filter the dataframe to keep only rows whose first column values are NOT in the list
df_filtered = df[~df.iloc[:, 0].isin(my_list)]

# Save the filtered dataframe to a new Excel file
df_filtered.to_excel("C:/Users/Alian Khan Gandapur/Desktop/Experiment/duplicate.xlsx", engine='openpyxl', index=False)