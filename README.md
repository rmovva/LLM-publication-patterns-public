# LLM-publication-patterns
Describing changes in LLM research trends in 2023. Feel free to open an issue with questions/comments about the code.

## Data

### 17K language modeling papers with annotations

The primary dataframe with 17K LLM-related papers from 1 January 2018 through 7 September 2023 is in `lm_metadata_all_annotations.json.gz`. 
The dataframe can be loaded in using the following codeblock:

```python
lm_metadata = pd.read_json('lm_metadata_all_annotations.json.gz',
                              orient='records', lines=True,
                              dtype={'id': str}, compression='gzip')
lm_metadata['above_pred_female_threshold'] = lm_metadata['above_pred_female_threshold'].map({1: True, 0: False})
lm_metadata['v1_date'] = pd.to_datetime(lm_metadata['v1_date'], unit='ms')
```

The dataframe consists of 19 metadata annotations per paper. Several of these columns are base arXiv metadata, and the columns we add are described below. 
See the paper's Methods section and Appendix for additional details on how the columns are defined.

- `LM_related_terms`: a list of LLM-related keywords that appear in the paper's title or abstract, which are used to classify the paper as being LLM-related
- `cluster`: one of 40 possible paper topics identified by clustering an embedding of the paper abstract and labeling the resulting clusters with topic names
- `domains`: a list of email domains per paper, used to infer affiliations
- `industry`: whether the paper contains one of 41 identified industry affiliations that are linked to at least 10 LLM papers. Note that because affiliations are determined using emails, this column undercounts the true number of industry affiliations (not all authors list emails).
- `academic`: whether the paper contains one of 280 identified academic affiliations that are linked to at least 10 LLM papers. Note that because affiliations are determined using emails, this column undercounts the true number of academic affiliations.
- `above_pred_female_threshold`: whether the paper's list of author names have a predicted majority (>=50%) of female authors. **\*Important: see footnote.** 
- `inferred_female_frac_nqg_uncertainty_threshold_0.100`: the fraction of author names that the nomquamgender package predicts to be gendered female using an uncertainty threshold in the package of 0.1
- `citationCount`: the paper's citation count as tracked by Semantic Scholar, as of 15 Sep 2023
- `percentile_rank_in_3_month_window`: the paper's percentile ranked by citation count, only comparing to other papers in its 3-month window
- `percentile_rank_in_12_month_window`: same as above, except comparing to all papers published in the same year

*Please read the paper for further details about the majority predicted gender column. In brief, (1) names are a noisy and imperfect correlate with gender, and (2) name-gender prediction methods systematically lack data and avoid making predictions for non-Western names, so a substantial fraction of authors in the dataset lack any prediction. In our paper, we use this column to study aggregate trends, and find that they hold across different prediction methods and confidence thresholds. Any analyses using this column should explicitly note these caveats to avoid incorrect conclusions.

### All 418K arXiv cs/stat papers with clusters

This dataframe, `cs_stat_metadata_clusters.json`, is available here: https://cornell.box.com/s/x6bzme0pnnl0j4o4utv5cvxtlo0edv7x.  


These data come directly from the [arXiv dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). We subsetted the data to include only papers which list at least one cs/stat arXiv category, we cleaned the columns a little bit, and then computed embeddings + clusters for each paper. The preprocessing code is [here](https://github.com/rmovva/LLM-publication-patterns-public/blob/master/data_prep/preprocess_metadata.ipynb), and the code used to cluster the papers is [here](https://github.com/rmovva/LLM-publication-patterns-public/blob/master/data_prep/topic_clustering.ipynb).  

If you'd like to define a custom subsetting procedure (instead of using the keyword list we chose), you can start from this file. It's also used in a supplementary figure in the paper.  

## Analyses

Notebooks to replicate all analyses starting from the annotated dataframe are available in the `./analysis` folder. The notebooks contain some documentation, but please feel free to open an issue with any problems.

## Regenerating the annotated data from scratch

Using scripts in the `./data_prep` folder, one can regenerate all paper annotations starting from the arXiv metadata downloaded from Kaggle: [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
These scripts can be used to update the results on later data timepoints, for example. Documentation to be released.
