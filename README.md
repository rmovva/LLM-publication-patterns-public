# LLM-publication-patterns
Describing changes in LLM research trends in 2023. Please contact rmovva at cs.cornell.edu with questions/comments.

## Data

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
- `industry`: whether the paper contains one of 41 identified industry affiliations that are linked to at least 10 LLM papers
- `academic`: whether the paper contains one of 280 identified academic affiliations that are linked to at least 10 LLM papers
- `above_pred_female_threshold`: whether the paper's list of author names have a predicted majority (>=50%) of female authors
- `inferred_female_frac_nqg_uncertainty_threshold_0.100`: the fraction of author names that the nomquamgender package predicts to be gendered female using an uncertainty threshold in the package of 0.1
- `citationCount`: the paper's citation count as tracked by Semantic Scholar, as of 15 Sep 2023
- `percentile_rank_in_3_month_window`: the paper's percentile ranked by citation count, only comparing to other papers in its 3-month window
- `percentile_rank_in_12_month_window`: same as above, except comparing to all papers published in the same year

## Analyses

Notebooks to replicate all analyses starting from the annotated dataframe are available in the `./analysis` folder. Documentation to be released. Please feel free to open an issue with any problems, in the mean time.  
Note that the metadata file for the full arXiv sample with clusters `cs_stat_metadata_clusters.json` (used in `./analysis/full_arxiv_metadata_stats.ipynb`) is not released because it exceeds Github's size limit and is only used in one supplemental figure. However, feel free to reach out if this data would be helpful for your analysis.

## Regenerating the annotated data from scratch

Using scripts in the `./data_prep` folder, one can regenerate all paper annotations starting from the arXiv metadata downloaded from Kaggle: [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
These scripts can be used to update the results on later data timepoints, for example. Documentation to be released.
