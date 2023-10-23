import pandas as pd
import numpy as np
import os
import argparse
import torch
from InstructorEmbedding import INSTRUCTOR
from multiprocessing import Pool


'''
Example usage. Change the directories accordingly. 

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_paper_embeddings.py \
    --metadata_path=/share/pierson/raj/LLM_bibliometrics_v2/processed_data/cs_stat_metadata.json \
    --output_dir=/share/pierson/raj/LLM_bibliometrics_v2/processed_data/embeddings \
    --batch_size=128 \
    --abstract_or_title=abstract
'''

# use argparse to get the metadata directory
# metadata_path, output_dir, both required
parser = argparse.ArgumentParser()
parser.add_argument('--metadata_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--abstract_or_title', type=str, default='abstract', choices=['abstract', 'title'])

args = parser.parse_args()
metadata_path = args.metadata_path
output_dir = args.output_dir
batch_size = args.batch_size
abstract_or_title = args.abstract_or_title

metadata = pd.read_json(metadata_path, lines=True, orient='records',
                        dtype={'id': str})

# Define a function to process each batch of notes on a GPU
def process_per_gpu_set(gpu_id, per_gpu_set):
    torch.cuda.set_device(gpu_id)  # Set the CUDA device for this process
    model = INSTRUCTOR('hkunlp/instructor-xl').cuda()
    prompt = "Represent the Computer Science research abstract for clustering; Input:"
    
    # Access the DataFrame batch to get the data for this process
    paper_texts = per_gpu_set[abstract_or_title].str.strip()
    text_with_prompt = [[prompt, text] for text in paper_texts]
    
    embeddings = model.encode(text_with_prompt, 
                              convert_to_numpy=True, 
                              show_progress_bar=True,
                              batch_size=128)
    
    arxiv_ids = per_gpu_set['id'].values
    return embeddings, arxiv_ids

# Split the papers into batches
num_gpus = torch.cuda.device_count()
paper_batches = []
per_gpu_size = len(metadata) // num_gpus
print('Number of GPUs:', num_gpus)
print('Number of papers to embed per GPU:', per_gpu_size)
for i in range(num_gpus):
    # Divide the data equally among GPUs
    gpu_batch = metadata.iloc[i * per_gpu_size: (i + 1) * per_gpu_size].copy()
    paper_batches.append(gpu_batch)

# Create a multiprocessing pool and distribute the batches to different processes
with Pool(num_gpus) as pool:
    gpu_ids = list(range(num_gpus))
    args_list = zip(gpu_ids, paper_batches)
    embeddings_list, arxiv_ids_list = zip(*pool.starmap(process_per_gpu_set, args_list))

# Concatenate the embeddings from different processes
embeddings = np.concatenate(embeddings_list, axis=0)
arxiv_ids = np.concatenate(arxiv_ids_list, axis=0)

print("Embeddings shape:", embeddings.shape)
print("ArXiv IDs shape:", arxiv_ids.shape)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the embeddings as npz
size = round(len(embeddings) / 1000)
yearmonthday = pd.Timestamp.now().strftime('%Y%m%d')
np.savez_compressed(os.path.join(output_dir, f'{abstract_or_title}-{size}K-{yearmonthday}.npz'), 
                    embeddings=embeddings,
                    id=arxiv_ids,
)