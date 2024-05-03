import os
import pdfplumber
import pandas as pd


def extract_last_contiguous_digits(filename):
    filename=filename.split('.')[0]
    digits = []

    for char in reversed(filename):
        
        if char.isdigit():
            digits.append(char)
        else:
            break
    return ''.join(reversed(digits))





def extract_pdf_content(pdf_path):
    """Extracts content from a single PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = []
        for page in pdf.pages:
            full_text.append(page.extract_text())
        return "\n".join([text for text in full_text if text])  # Join non-empty text from all pages

def read_pdfs_and_save_to_csv(folder_path, output_csv):
    """Reads all PDF files in the specified folder and saves their content to a CSV file."""
    data = {'title': [], 'text': []}
    filenum=0
    folder_path=folder_path+'/PDFs'
    for filename in os.listdir(folder_path):
        
        idx = extract_last_contiguous_digits(filename)
        try:
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                content = extract_pdf_content(pdf_path)
                
                if content:
                    lines = content.split('\n')
                    title = idx
                    text = '\n'.join(lines[1:])  
                    text = text.replace('\n', ' ') 
                    
                    data['title'].append(title)
                    data['text'].append(text)
                
                
        except:
            pass 

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df['title'] = pd.to_numeric(df['title'])
    df = df.sort_values(by='title')
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Specify the folder containing PDFs and the output CSV file path
folder_path = 'data'
output_csv = 'data/pdf_texts.csv'
output_tsv='data/pdf_texts.tsv' 
hyperlinks= 'data/hyperlinks.csv'

# Process the PDFs and save the results
#read_pdfs_and_save_to_csv(folder_path, output_csv)
#label2tsv(output_csv,output_tsv,hyperlinks)

hyperlinks = pd.read_csv(hyperlinks)  # Assuming the first column is the index
pdf_texts = pd.read_csv(output_csv)
print(hyperlinks.head())

# Assume the first column of pdf_texts is named 'Index' (change accordingly if it has a different name)
indices = pdf_texts.iloc[:, 0]  # This selects the first column of pdf_texts

# Extract corresponding 'Success' entries
pdf_texts['Success'] = [hyperlinks.loc[idx-2, 'Success'] if idx in hyperlinks.index else None for idx in indices]
pdf_texts.to_csv('data/pdf_texts.tsv', sep='\t', index=False)

# Display the results
