import requests
import argparse
import json
import pandas as pd
import tqdm
import os

S2_API_KEY = os.environ['S2_API_KEY']

# Add command line parser for lm_metadata_path and output_path
parser = argparse.ArgumentParser()
parser.add_argument('--lm_metadata_path', type=str, default=None, help='Path to the metadata file for the LM papers', required=True)
parser.add_argument('--s2_output_path', type=str, default=None, help='Path to the S2 output df', required=True)
args = parser.parse_args()
lm_metadata_path = args.lm_metadata_path
s2_output_path = args.s2_output_path

lm_metadata = pd.read_json(lm_metadata_path, orient='records', lines=True, dtype={'id': str})
lm_arxiv_ids = [f'arXiv:{id}' for id in lm_metadata['id']]
assert len(set(lm_arxiv_ids)) == len(lm_arxiv_ids) # Check for duplicates

# Splitting the arxiv_ids into chunks of 500 due to the API limitation
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

BASE_URL = 'https://api.semanticscholar.org/graph/v1/paper/batch'
HEADERS = {
    'x-api-key': S2_API_KEY,
    'Content-Type': 'application/json'
}

# We'll loop through chunks of IDs and send batch requests
lm_papers = []
for chunk in tqdm.tqdm(list(chunks(lm_arxiv_ids, 500))):
    response = requests.post(
        BASE_URL,
        headers=HEADERS,
        params={'fields': 'paperId,title,citationCount,influentialCitationCount'},
        json={"ids": chunk}
    )
    
    if response.status_code == 200:
        papers = response.json()  # Assuming the response directly gives a list of papers
        for (i, paper) in enumerate(papers):
            if paper is None:
                print(f"Failed to fetch details for paper: {chunk[i]}")
                continue
            arxiv_id = chunk[i].split(':')[1]
            paper['id'] = arxiv_id
            lm_papers.append(paper)
    else:
        print(f"Failed to fetch details for chunk: {chunk}")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

lm_s2_df = pd.DataFrame.from_records(lm_papers)
# Move the id column to the front, and rename paperId to s2PaperId
lm_s2_df = lm_s2_df.set_index('id').reset_index().rename(columns={'paperId': 's2PaperId'})

lm_s2_df.to_json(s2_output_path, orient='records', lines=True)
    