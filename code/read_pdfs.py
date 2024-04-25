import os
import pdfplumber
import pandas as pd

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
    folder_path=folder_path+'/PDFs'
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            content = extract_pdf_content(pdf_path)
            if content:
                lines = content.split('\n')
                title = pdf_path.split('pdf')[1]
                text = '\n'.join(lines[1:])  
                text = text.replace('\n', '') 
                
                data['title'].append(title)
                data['text'].append(text)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df['title'] = pd.to_numeric(df['title'])
    df = df.sort_values(by='title')
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Specify the folder containing PDFs and the output CSV file path
folder_path = 'data'
output_csv = 'data/pdf_texts.csv'

# Process the PDFs and save the results
read_pdfs_and_save_to_csv(folder_path, output_csv)