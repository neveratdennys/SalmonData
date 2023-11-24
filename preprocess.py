import requests
from zipfile import ZipFile
import pandas as pd
import io

# get zip file from DFO data source
zip_url = "https://api-proxy.edh.azure.cloud.dfo-mpo.gc.ca/catalogue/records/7ac5fe02-308d-4fff-b805-80194f8ddeb4/attachments/ise-ecs.zip"

response = requests.get(zip_url)
zip_data = response.content

with ZipFile(io.BytesIO(zip_data)) as my_zip:
    # Only use englsih csv files
    file_list = [file_name for file_name in my_zip.namelist() if 'eng.csv' in file_name]
    
    data_frames = []
    for file_name in file_list:
        with my_zip.open(file_name) as file:
            # Read the CSV file as text with 'latin-1' encoding
            # Process each CSV file, replacing the en dash character (–) with a hyphen (-)
            csv_text = file.read().decode('latin-1')
            csv_text_cleaned = csv_text.replace('–', '-')
            cleaned_bytes = csv_text_cleaned.encode('utf-8')
           
            df = pd.read_csv(io.BytesIO(cleaned_bytes), skiprows = 1)

            # TODO: add new column that specify which fishing method is being used
            
            data_frames.append(df)

            #import pdb; pdb.set_trace()

       
# Combine or perform operations on the DataFrames as needed
combined_data = pd.concat(data_frames)

# Work with the combined data as needed
print("Combined Data:")
print(combined_data.shape)
