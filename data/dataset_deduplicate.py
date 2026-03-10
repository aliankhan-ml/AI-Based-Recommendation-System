import pandas as pd
import os

# Load the first Excel sheet into a DataFrame
df1 = pd.read_excel('C:/Users/Alian Khan Gandapur/Desktop/Aliyan/patterns.xlsx', engine='openpyxl')

# Load the second Excel sheet into a DataFrame
df2 = pd.read_excel('C:/Users/Alian Khan Gandapur/Desktop/Ak/patterns.xlsx', engine='openpyxl')

# Loop through each row of the first DataFrame
for index1, row1 in df1.iterrows():
    # Loop through each row of the second DataFrame
    for index2, row2 in df2.iterrows():
        # If the file descriptions match
        if row1['Tags'] == row2['Tags']:
            # Get the file name and path from the second DataFrame
            Image = row2['Image']
            file_path = row2['file_path']
            # Delete the file
            os.remove(file_path)
            # Print the file name
            print(f"File '{Image}' with description '{row1['Tags']}' has been deleted.")
            # Drop the matching row from the second DataFrame
            df2.drop(index2, inplace=True)
            # Exit the inner loop since we've found a match
            break

# Save the modified second DataFrame back to the original Excel sheet
df2.to_excel('patterns.xlsx', index=False)
