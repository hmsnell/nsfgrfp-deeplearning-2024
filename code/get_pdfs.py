import requests
import pandas as pd
import os

def download_google_pdf(id,count):
    url = "https://drive.google.com/uc?export=download&id="
    id = str(id)
    url = url+id
    try:
        # Send GET request
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        # Write the contents of the response to a file
        with open('data/PDFs/pdf'+str(count)+'.pdf', 'wb') as f:
            f.write(response.content)
        #print(f"PDF has been successfully downloaded and saved as {filename}")
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

def download_pdf(hyperlink, count):
    url = hyperlink
    try:
        headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        }
        # Send GET request
        response = requests.get(url, headers=headers)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        # Write the contents of the response to a file
        with open('data/PDFs/pdf'+str(count)+'.pdf', 'wb') as f:
            f.write(response.content)
        #print(f"PDF has been successfully downloaded and saved as {filename}")
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

ncount,count,index=0,0,1
for _, row in df.iterrows():
    proposal_id = row['proposal_id']
    hyperlink = row['proposal_hyperlinks']
    Name = row['Name']
    index+=1
    # Check if 'googledrive' is in the hyperlink
    if type(hyperlink)==str and 'drive.google' in hyperlink :
        count+=1
        download_google_pdf(proposal_id,index)
    elif type(hyperlink)==str and '.pdf' in hyperlink: 
        count+=1
        download_pdf(hyperlink, index)
    else:
        ncount+=1
        print(f"Not accepted link: {hyperlink}")
print(ncount)
