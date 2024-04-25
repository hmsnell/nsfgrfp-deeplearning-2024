import requests
import pandas as pd
import os

def download_pdf(id,count):
    url = "https://drive.google.com/uc?export=download&id="
    print(id)
    url = url+id
    try:
        # Send GET request
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Write the contents of the response to a file
        with open('data/PDFs/pdf'+str(count)+'.pdf', 'wb') as f:
            f.write(response.content)
        print(f"PDF has been successfully downloaded and saved as {filename}")
    except requests.RequestException as e:
        print(f"An error occurred: {e}")



# URL of the PDF you want to download
filename = 'data/hyperlinks.csv'
df = pd.read_csv(filename)

# Check if the directory exists
if not os.path.exists('data/PDFs'):
    # Create the directory
    os.makedirs('data/PDFs')
    print(f"Directory '{'PDFs'}' created")
else:
    print(f"Directory '{'PDFs'}' already exists")


ncount,count=0,0
for _, row in df.iterrows():
    proposal_id = row['proposal_id']
    hyperlink = row['proposal_hyperlinks']
    
    # Check if 'googledrive' is in the hyperlink
    if type(hyperlink)==str and 'drive.google' in hyperlink :
        count+=1
        download_pdf(proposal_id,count)
    else:
        ncount+=1
        print(f"Not google drive: {proposal_id}")
print(ncount)