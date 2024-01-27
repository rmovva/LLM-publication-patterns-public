import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.stats import rankdata, chi2_contingency

# Force font family to be Arial, if it's available
from matplotlib import pyplot as plt
import matplotlib
for f in matplotlib.font_manager.fontManager.ttflist:
    if 'Arial' in f.name:
        plt.rcParams['font.family'] = f.name
        # print("Found Arial font, setting as default")
        break

import sys
# if current dir is analysis/, then add ../data_prep to path
# else add ./data_prep to path
if os.path.basename(os.getcwd()) == 'analysis':
    sys.path.append('../data_prep')
else:
    sys.path.append('./data_prep')

from preprocess_utils import PROCESSED_DATA_DIR
from preprocess_utils import add_semantic_scholar_info, add_affiliation_info
from preprocess_utils import get_domain_counter, is_industry_domain, is_academic_domain, industry_domains, academic_domains

# For topic shift analysis within LLM papers, papers after this date are considered recent
ANALYSIS_START_DATE = datetime(2018, 1, 1, 0, 0, 0)
RECENT_DATE_THRESHOLD = datetime(2023, 1, 1, 0, 0, 0)


def load_annotated_lm_metadata(repo_path=True):
    if repo_path:
        file_in_repo = 'lm_metadata_all_annotations.json.gz'
        if os.path.basename(os.getcwd()) == 'analysis':
            file_in_repo = '../' + file_in_repo
        lm_metadata = pd.read_json(file_in_repo,
                                    orient='records', lines=True,
                                    dtype={'id': str},
                                    compression='gzip',
                                    )             
    else:
        lm_metadata = pd.read_json(os.path.join(PROCESSED_DATA_DIR, 'lm_metadata_all_annotations.json'),
                                    orient='records', lines=True,
                                    dtype={'id': str})
    lm_metadata['v1_date'] = pd.to_datetime(lm_metadata['v1_date'], unit='ms')
    return lm_metadata

'''
This function cannot be run natively in the public repo,
but the output is the same as the available annotated metadata file.
To run on new data, change the filepath in the pd.read_json() call.
'''
def annotate_lm_metadata():
    lm_metadata = pd.read_json(os.path.join(PROCESSED_DATA_DIR, 'lm_papers_metadata_clusters.json'),
                                 orient='records', lines=True,
                                 dtype={'id': str})
    lm_metadata['v1_date'] = pd.to_datetime(lm_metadata['v1_date'], unit='ms')
    lm_metadata = add_semantic_scholar_info(lm_metadata)
    lm_metadata = add_affiliation_info(lm_metadata)
    return lm_metadata


'''
This function cannot be run in public repo currently, because it requires data for all 400K papers.
Too large to include in repo, will figure out alternate release.
'''
def load_all_metadata(full_date_range = False):
    if not full_date_range:
        metadata = pd.read_json(os.path.join(PROCESSED_DATA_DIR, 'cs_stat_metadata_clusters.json'),
                                orient='records', lines=True,
                                dtype={'id': str})
    else:
        metadata = pd.read_json(os.path.join(PROCESSED_DATA_DIR, 'fulldaterange_cs_stat_metadata_20230910.json'),
                                orient='records', lines=True,
                                dtype={'id': str})
    metadata['v1_date'] = pd.to_datetime(metadata['v1_date'], unit='ms')
    return metadata


def convert_citations_to_ranks(lm_metadata, month_window_sizes=[3, 6, 12]):
    '''
    This function will take the 'citationCount' column and convert to percentiles in a given time window.
    New columns will be added to lm_metadata, for k = 3, 6, 12:
    - percentile_rank_in_{k}_month_window: percentile rank in a k-month window
    
    We can't use raw citationCount directly in analysis because it's not comparable across years; more recent papers have less time to accumulate citations.
    Normalizing to year may also not be great, because Sep 2023 is much closer than Jan 2023.
    We can bin into windows of length n_months and then rank within each window.

    lm_metadata: LLM papers dataframe; must already have 'citationCount' column
    '''
    for n_months in month_window_sizes:
        lm_metadata[f'percentile_rank_in_{n_months}_month_window'] = np.nan

        # Get equally spaced bins of length n_months starting from ANALYSIS_START_DATE to the last paper's date
        # e.g. if n_months = 3, bins will be [2018-01-01, 2018-04-01, 2018-07-01, ...]
        date_intervals = list(pd.interval_range(start=ANALYSIS_START_DATE, end=lm_metadata['v1_date'].max(), freq=f'{n_months}MS', closed='left'))
        # Add a final interval to capture remaining papers
        date_intervals.append(pd.Interval(date_intervals[-1].right, lm_metadata['v1_date'].max(), closed='left'))

        # For each bin, get the papers that fall within that bin, and then use lm_metadata.loc to set their citationCount percentiles
        for i, interval in enumerate(date_intervals):
            papers_in_interval = lm_metadata['v1_date'].between(interval.left, interval.right)
            n_papers_in_interval = papers_in_interval.sum()
            citation_ranks = rankdata(lm_metadata.loc[papers_in_interval, 'citationCount'], method='average', nan_policy='omit')
            lm_metadata.loc[papers_in_interval, f'percentile_rank_in_{n_months}_month_window'] = citation_ranks / n_papers_in_interval

    return lm_metadata


def get_domain_counts_with_affiliation_type(metadata):
    domain_counter = get_domain_counter(metadata)
    # Get list of domains sorted by value
    domains = [x[0] for x in domain_counter.most_common()]
    counts = [x[1] for x in domain_counter.most_common()]
    affiliation_types = []
    for domain in domains:
        if is_industry_domain(domain):
            affiliation_types.append('industry')
        elif is_academic_domain(domain):
            affiliation_types.append('academic')
        else:
            affiliation_types.append('other')
    
    return pd.DataFrame({'domain': domains, 'count': counts, 'affiliation_type': affiliation_types})


def get_experienced_llm_authors(lm_metadata, date_threshold=RECENT_DATE_THRESHOLD):
    '''
    Return a set of author names who have co-authored LLM papers before 2023.

    This can be used to annotate the experience level of a paper's authors (we consider only either the first or last author).
    '''
    lm_metadata_authors = lm_metadata.explode('authors').rename(columns={'authors': 'author'})
    print("Number of total co-authors on LLM papers:", len(lm_metadata_authors))
    pre_threshold_lm_authors = set(lm_metadata_authors[lm_metadata_authors.v1_date < date_threshold].author.unique())
    print(f"Number of total co-authors on LLM papers before {date_threshold.strftime('%Y-%m-%d')}:", len(pre_threshold_lm_authors))

    return pre_threshold_lm_authors


def get_topic_counts_by_binary_variable(lm_metadata, binary_col_of_interest, group_by='cluster', min_topic_count=25):
    """
    Given lm_metadata which has topics/subarXivs in 'group_by' column + a binary column of interest (e.g. industry affiliation), give a df with:
    - 'cluster' column with name of topic or other group
    - 'ratio' column with p(topic | col = True)/p(topic | col = False)
    - 'count_true' column with (count(topic & col = True), (count(col = True))
    - 'count_false' column with (count(topic & col = False), (count(col = False))
    - 'p_topic_if_true' column with p(topic | col = True)
    - 'p_topic_if_false' column with p(topic | col = False)
    - 'p_chi2' column with p-value of chi2 test for independence of topic and col

    If you want to group by arXiv category instead of cluster, pass group_by='category'
    """
    pos_idxs = lm_metadata[binary_col_of_interest] == True
    neg_idxs = lm_metadata[binary_col_of_interest] == False
    count_pos = pos_idxs.sum()
    count_neg = neg_idxs.sum()
    results_df = {
        'topic': [],
        'ratio': [],
        'count_true': [],
        'count_false': [],
        'p_topic_if_true': [],
        'p_topic_if_false': [],
        'p_chi2': [],
    }
    pos_topic_counts = lm_metadata[pos_idxs].groupby(group_by).size()
    neg_topic_counts = lm_metadata[neg_idxs].groupby(group_by).size()
    for k in pos_topic_counts.index:
        if k not in neg_topic_counts.index:
            neg_topic_counts[k] = 0
        if pos_topic_counts[k] + neg_topic_counts[k] < min_topic_count:
            continue
        results_df['topic'].append(k)
        results_df['count_true'].append((pos_topic_counts[k], count_pos))
        results_df['count_false'].append((neg_topic_counts[k], count_neg))
        results_df['p_topic_if_true'].append(pos_topic_counts[k] / count_pos)
        results_df['p_topic_if_false'].append(neg_topic_counts[k] / count_neg)
        results_df['ratio'].append((pos_topic_counts[k] / count_pos) / (neg_topic_counts[k] / count_neg))

        observed = [[pos_topic_counts[k], count_pos - pos_topic_counts[k]],
                    [neg_topic_counts[k], count_neg - neg_topic_counts[k]]]
        chi2, p, _, _ = chi2_contingency(observed)
        results_df['p_chi2'].append(p)
    results_df = pd.DataFrame(results_df)
    return results_df.sort_values('ratio', ascending=False)


def make_dotplot_from_topic_counts_by_binary_variable(
        df,
        bar_names_col,
        true_label,
        false_label,
        top_and_bottom_k=5,
        true_color='#3260a0',
        false_color='#bd2830',
        figsize=(2, 3),
        dpi=300,
        tickfontsize=8,
        labelfontsize=8,
        ax=None,
        legend_coords=None,
        xlabel=r'$p$(topic | group)',
        errorbars=True
    ):
    df = df.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    df = df.sort_values('ratio', ascending=True)
    if top_and_bottom_k is not None:
        df = pd.concat([df[df.ratio < 1].head(top_and_bottom_k), 
                        df[df.ratio > 1].tail(top_and_bottom_k)])

    # For row in df
    for i, row in df.iterrows():
        topic = row[bar_names_col]
        p_topic_if_true = row['p_topic_if_true']
        p_topic_if_false = row['p_topic_if_false']
        ax.scatter([p_topic_if_true], [topic], color=true_color, marker='o', s=10)
        ax.scatter([p_topic_if_false], [topic], color=false_color, marker='o', s=10)
        if errorbars:
            # bernoulli standard errors
            errorbar_true = 1.96 * np.sqrt(p_topic_if_true * (1 - p_topic_if_true)/row['count_true'][1])
            errorbar_false = 1.96 * np.sqrt(p_topic_if_false * (1 - p_topic_if_false)/row['count_false'][1])
            ax.errorbar([p_topic_if_true], [topic], xerr=errorbar_true, color=true_color, capsize=0, capthick=0.5, elinewidth=0.5)
            ax.errorbar([p_topic_if_false], [topic], xerr=errorbar_false, color=false_color, capsize=0, capthick=0.5, elinewidth=0.5)

    # Draw a dashed hline between the top and bottom k
    ax.axhline(y=top_and_bottom_k-0.5, color='grey', linestyle='--', linewidth=0.5)

    ax.tick_params(axis='x', which='major', labelsize=tickfontsize)
    ax.tick_params(axis='y', which='major', labelsize=tickfontsize)

    if legend_coords:
        ax.legend([true_label, false_label], fontsize=labelfontsize, bbox_to_anchor=legend_coords, loc='upper left')
    else:
        ax.legend([true_label, false_label], fontsize=labelfontsize)
    # ax.set_xlabel('% of LLM papers', fontsize=labelfontsize)
    ax.set_xlabel(xlabel, fontsize=labelfontsize)

    return ax


def make_pretty_ratio_plot_from_ratio_df(
        df, 
        bar_names_col, 
        neg_label, 
        pos_label, 
        ratio_col = 'ratio',
        top_and_bottom_k=5, 
        log_scale = True,
        intermediate_ticks = True,
        equal_max_min_xlim = True,
        manual_xlim = None,
        xlabel_yoffset = -1.7,
        xlabel_xoffset = 0.5,
        figsize=(3, 3),
        cmap_name='RdBu',
        fixed_color=True,
        color_offset=0.1,
        dpi=300,
        labelfontsize=8,
        tickfontsize=8,
        ax=None,
        title=None,
        filename=None,
        errorbars=True
    ):
    """
    Given a df with a column ratio and a column of bar names, plot the ratio as a horizontal bar chart. 
    Useful for plotting relative enrichment of binary variables. 

    df: dataframe with columns 'ratio' and bar_names_col
    bar_names_col: name of column in df with the labels for the bars
    neg_label: label for the unenriched class
    pos_label: label for the enriched class
    ratio_col: name of column in df with the enrichment ratio
    only_plot_top_and_bottom_k: if not None, only plot the top and bottom k bars
    log_scale: if True, plot ratios in log space (though the labels will be in linear space)
    intermediate_ticks: if True, plot intermediate ticks, not just the min and max
    equal_max_min_xlim: if True, set the left and right xlimits to be the same
    manual_xlim: if not None, set the xlimits to be this. should be in abs(log2) space
    xlabel_yoffset: y-direction offset for the xlabel annotations (neg_label and pos_label)
    xlabel_xoffset: x-direction offset for the xlabel annotations (neg_label and pos_label)
    figsize: figure size
    cmap_name: name of the colormap to use
    fixed_color: if True, use a fixed color for all bars, depending on the sign of the ratio
    color_offset: if fixed_color is True, this is the offset from the ends of the colormap to use
    dpi: dpi for the figure
    title: title for the figure
    filename: if not None, save the figure to this filename
    """
    df = df.copy()

    assert ratio_col in df.columns
    if ratio_col != 'ratio':
        df = df.rename(columns={ratio_col:'ratio'})

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    df = df.sort_values('ratio', ascending=True)
    if top_and_bottom_k is not None:
        df = pd.concat([df[df.ratio < 1].head(top_and_bottom_k), 
                        df[df.ratio > 1].tail(top_and_bottom_k)])
    if log_scale: 
        df['log_ratio'] = np.log2(df['ratio'])
        # use colormap for bars
        my_cmap = plt.cm.get_cmap(cmap_name)
        if fixed_color:
            # Only set the color depending on the sign of the log ratio
            df['color'] = df['log_ratio'].map(lambda y: my_cmap(color_offset) if y < 0 else my_cmap(1-color_offset))
        else:
            rescale = lambda y: (y - df['log_ratio'].min()) / (df['log_ratio'].max() - df['log_ratio'].min())
            df['color'] = df['log_ratio'].map(lambda y: my_cmap(rescale(y)))
        # plot bars
        df.plot.barh(x=bar_names_col, y='log_ratio', ax=ax, color=df['color'], legend=False)
        if errorbars:
            # errorbar on risk ratio. https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals8.html
            errorbars = []
            for i in range(len(df)):
                X1, N1 = df.iloc[i]['count_true'] # pos and total count for positive class
                X2, N2 = df.iloc[i]['count_false'] # pos and total count for negative class
                # chi2 test
                _, chi_p, _, _ = chi2_contingency([[X1, N1-X1], [X2, N2-X2]])
                print("Chi2 test for %s: p = %2.3e (not multiple hypothesis corrected)" % (df.iloc[i][bar_names_col], chi_p))
                assert np.allclose((X1/N1)/(X2/N2), df.iloc[i]['ratio'])
                log_se = np.sqrt((N1-X1)/(X1 * N1) + (N2 - X2)/(X2 * N2))
                assert log_se > 0
                upper_ci = np.exp(np.log(df.iloc[i]['ratio']) + 1.96 * log_se)
                lower_ci = np.exp(np.log(df.iloc[i]['ratio']) - 1.96 * log_se)
                assert upper_ci >= lower_ci
                errorbars.append([np.log2(df.iloc[i]['ratio']) - np.log2(lower_ci), np.log2(upper_ci) - np.log2(df.iloc[i]['ratio'])])
            errorbars = np.array(errorbars).T
            ax.errorbar(df['log_ratio'], df[bar_names_col], xerr=errorbars, color='black', capsize=0, capthick=0.5, elinewidth=0.5, ls='none')
            
        if manual_xlim:
            log_xmin, log_xmax = manual_xlim
            ax.set_xlim(-log_xmin, log_xmax)
        elif equal_max_min_xlim:
            # Gets 2 to the power of the max log ratio, rounds up, and multiplies by 1.05 to get a little extra space
            max_xlim_in_real_space = round(np.ceil(np.exp2(max(df['log_ratio'].abs())) * 1.2))
            log_xmax = log_xmin = np.log2(max_xlim_in_real_space)
            ax.set_xlim(-log_xmin, log_xmax)
        else:
            log_xmin = np.abs(np.floor(df['log_ratio'].min()))
            log_xmax = np.ceil(df['log_ratio'].max())
            ax.set_xlim(-log_xmin, log_xmax)
        if intermediate_ticks:
            # tick locs should be from -log_xlim to log_xlim in increments of 1
            # working in log2 space so ticks are whole numbers
            tick_locations = np.arange(-np.floor(log_xmin), np.floor(log_xmax) + 1, 1)
            tick_labels = ['%dx' % x for x in np.exp2(abs(tick_locations))]
            ax.set_xticks(tick_locations, tick_labels, fontsize=tickfontsize)
            # Add neg_label and pos_label text below tick labels, at -log_xlim/2 and +log_xlim/2
            ax.text(-log_xmin + xlabel_xoffset, xlabel_yoffset, neg_label, 
                    horizontalalignment='center', verticalalignment='top', fontsize=labelfontsize)
            ax.text(log_xmax - xlabel_xoffset, xlabel_yoffset, pos_label, 
                    horizontalalignment='center', verticalalignment='top', fontsize=labelfontsize)
        else:
            ax.set_xticks([-log_xmin, log_xmax], 
                          ['%2.1fx %s' % (np.exp2(log_xmin), neg_label), '%2.1fx %s' % (np.exp2(log_xmax), pos_label)], 
                          fontsize=tickfontsize)
        ax.axvline(0, color='#888888', linestyle='--', alpha=0.5)
    else:
        df.plot.barh(x=bar_names_col, y='ratio', ax=ax, legend=False)
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([xmin, xmax], ['%2.1fx %s' % (xmin, neg_label), '%2.1fx %s' % (xmax, pos_label)], fontsize=tickfontsize)
        ax.axvline(1, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', which='major', labelsize=tickfontsize)
    if title:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return ax


def enrichment_plot_and_dotplot(
        df,
        neg_label,
        pos_label,
        true_label,
        false_label,
        bar_names_col='topic',
        top_and_bottom_k=5,
        total_figsize=(5, 2.1), 
        dpi=200,
        width_ratios=(3, 2),
        left_plot='enrichment',
        xlabel_xoffset = 0.5,
        xlabel_yoffset = -1.7,
        legend_coords=None,
        manual_enrichment_xlim=None,
        dotplot_xlabel=r'$p$(topic | group)',
        labelfontsize=8,
        tickfontsize=8,
    ):
    f, (ax0, ax1) = plt.subplots(1, 2, 
                                 figsize=total_figsize, 
                                 dpi=dpi, 
                                 width_ratios=width_ratios,
                                 sharey=True,
                                 )

    if left_plot == 'enrichment':
        enrich_ax = ax0
        dot_ax = ax1
    else:
        enrich_ax = ax1
        dot_ax = ax0

    # Use shorthand abbreviations for the cluster names
    df = df.copy()
    
    # Check that this is for the LLM subset, not all of arXiv
    if "Applications of LLMs/ChatGPT" in df[bar_names_col].unique() or "Applications and Benchmark Evaluations" in df[bar_names_col].unique():
        df[bar_names_col] = df[bar_names_col].map(cluster_abbrevs_for_plotting)
    
    enrich_ax = make_pretty_ratio_plot_from_ratio_df(
        df,
        bar_names_col=bar_names_col,
        neg_label=neg_label,
        pos_label=pos_label,
        ratio_col='ratio',
        ax=enrich_ax,
        top_and_bottom_k=top_and_bottom_k,
        manual_xlim=manual_enrichment_xlim,
        xlabel_xoffset=xlabel_xoffset,
        xlabel_yoffset=xlabel_yoffset,
        labelfontsize=labelfontsize,
        tickfontsize=tickfontsize,
    )

    dot_ax = make_dotplot_from_topic_counts_by_binary_variable(
        df,
        bar_names_col=bar_names_col,
        true_label=true_label,
        false_label=false_label,
        ax=dot_ax,
        legend_coords=legend_coords,
        top_and_bottom_k=top_and_bottom_k,
        xlabel=dotplot_xlabel,
        labelfontsize=labelfontsize,
        tickfontsize=tickfontsize,
    )

    return f, (enrich_ax, dot_ax)


cluster_abbrevs_for_plotting = {
    "Biases & Harms": "Biases & Harms",
    "Efficiency & Performance": "Efficiency & Performance",
    "Representations, Syntax, Semantics": "BERT & Embeddings",
    "Entity Extraction & RecSys": "Entity Extraction & RecSys",
    "Pretrained LMs & Text Classification": "Pretrained LMs & Text Classification",
    "LLMs, Reasoning, Chain-of-Thought": "Reasoning & Chain-of-Thought",
    "Natural Sciences": "Natural Sciences",
    "Summarization and Evaluation": "Summarization & Evaluation",
    "Video & Multimodal Models": "Video & Multimodal Models",
    "Knowledge Graphs and Commonsense": "Knowledge Graphs & Commonsense",
    "Software, Planning, Robotics": "Software, Planning, Robotics",
    "Knowledge Distillation": "Knowledge Distillation",
    "Privacy & Adversarial Risks": "Privacy & Adversarial Risks",
    "Multilingual Transfer Learning" : "Multilingual Transfer Learning",
    "Translation & Low-Resource Languages" : "Translation & Low-Resource",
    "Applications of LLMs/ChatGPT" : "Applications of LLMs/ChatGPT",
    "Dialogue & Conversational AI" : "Dialogue & Conversational AI",
    "Legal & Scientific Document Analysis" : "Legal & Scientific Documents",
    "Question Answering & Retrieval" : "Question Answering & Retrieval",
    "Emotion/Sentiment Analysis" : "Emotion & Sentiment Analysis",
    "Speech Recognition" : "Speech Recognition",
    "Applications and Benchmark Evaluations" : "Applications & Benchmark Evals",
    "Prompts & In-Context Learning" : "Prompts & In-Context Learning",
    "Transformers, RNNs, Attention" : "Transformer/RNN Architectures",
    "Fine-Tuning & Domain Adaptation" : "Fine-Tuning & Domain Adaptation",
    "Text Generation" : "Text Generation",
    "NLP for Healthcare" : "NLP for Healthcare",
    "Social Media & Misinformation" : "Social Media & Misinformation",
    "Societal Implications of LLMs" : "Societal Implications of LLMs",
    "Toxicity & Hate Speech" : "Toxicity & Hate Speech",
    "Search, Ranking, Retrieval" : "Search & Retrieval",
    "Visual Foundation Models" : "Visual Foundation Models",
    "Audio and Music Modeling" : "Audio & Music Modeling",
    "Finance Applications" : "Finance Applications",
    "Code Generation" : "Code Generation",
    "Datasets & Benchmarks" : "Datasets & Benchmarks",
    "Spelling & Grammar Correction" : "Spelling & Grammar Correction",
    "Vision-Language Models" : "Vision-Language Models",
    "Human Feedback & Interaction" : "Human Feedback & Interaction",
    "Interpretability & Reasoning" : "Interpretability & Reasoning",
}


SUBARXIV_NAMES_DICT = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Comp. Eng, Finance, and Sci",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases", 
    "cs.DC": "Distributed Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structs and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "CS and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in CS",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural Computing",
    "cs.NI": "Networking and Internet",
    "cs.OH": "CS: Other",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Info. Networks",
    "cs.SY": "Systems and Control",
    "stat.AP": "Stat - Applications",
    "stat.CO": "Stat - Computation",
    "stat.ME": "Stat - Methodology",
    "stat.ML": "Stat - ML",
    "stat.OT": "Stat - Other"
}


domain_to_abbreviated_name = {
    "microsoft.com": "Microsoft",
    "google.com": "Google",
    "cmu.edu": "CMU",
    "stanford.edu": "Stanford",
    "tsinghua.edu.cn": "Tsinghua",
    "amazon.com": "Amazon",
    "fb.com": "Meta",
    "pku.edu.cn": "Peking",
    "washington.edu": "UW",
    "alibaba-inc.com": "Alibaba",
    "ac.cn": "Chinese Academy of Sci.",
    "mit.edu": "MIT",
    "tencent.com": "Tencent",
    "ibm.com": "IBM",
    "illinois.edu": "UIUC",
    "sjtu.edu.cn": "SJTU",
    "huawei.com": "Huawei",
    "nyu.edu": "NYU",
    "berkeley.edu": "UC Berkeley",
    "zju.edu.cn": "Zhejiang",
    "allenai.org": "AllenAI",
    "gatech.edu": "Georgia Tech",
    "fudan.edu.cn": "Fudan",
    "ust.hk": "HKUST",
    "usc.edu": "USC",
    "ntu.edu.sg": "NTU Singapore",
    "cam.ac.uk": "Cambridge",
    "ucla.edu": "UCLA",
    "ucsd.edu": "UCSD",
    "utexas.edu": "UT Austin",
    "hit.edu.cn": "Harbin Inst. of Tech.",
    "ustc.edu.cn": "U. of Sci. and Tech. of China",
    "ethz.ch": "ETH Zurich",
    "ed.ac.uk": "Edinburgh",
    "columbia.edu": "Columbia",
    "jhu.edu": "JHU",
    "nus.edu.sg": "NUS",
    "ucsb.edu": "UCSB",
    "umass.edu": "UMass",
    "uva.nl": "U. Amsterdam",
    "ox.ac.uk": "Oxford",
    "cuhk.edu": "Chinese U. of Hong Kong",
    "umich.edu": "UMich",
    "kaist.ac.kr": "KAIST",
    "ruc.edu.cn": "Renmin U. of China",
    "uwaterloo.ca": "Waterloo",
    "seas.upenn.edu": "UPenn",
    "buaa.edu.cn": "Beihang",
    "princeton.edu": "Princeton",
    "navercorp.com": "Naver",
    "cornell.edu": "Cornell",
    "umd.edu": "UMD",
    "salesforce.com": "Salesforce",
    "adobe.com": "Adobe",
    "bytedance.com": "ByteDance",
    "snu.ac.kr": "SNU",
    "ucl.ac.uk": "UCL",
    "hku.hk": "HKU",
    "monash.edu": "Monash",
    "isi.edu": "ISI",
    "baidu.com": "Baidu",
    "epfl.ch": "EPFL",
    "inria.fr": "Inria",
    "nvidia.com": "NVIDIA",
    "ubc.ca": "UBC",
    "ntu.edu.tw": "NTU",
    "di.ku.dk": "KU",
    "unc.edu": "UNC",
    "osu.edu": "OSU",
    "cis.lmu.de": "LMU Munich",
    "u.nus.edu": "NUS",
    "sheffield.ac.uk": "Sheffield",
    "bupt.edu.cn": "BUPT",
    "westlake.edu.cn": "Westlake",
    "samsung.com": "Samsung",
    "northeastern.edu": "Northeastern",
    "asu.edu": "ASU",
    "pjlab.org.cn": "PJ Lab",
    "utoronto.ca": "U of T",
    "whu.edu.cn": "WHU",
    "mila.quebec": "Mila",
    "yale.edu": "Yale",
    "psu.edu": "Penn State",
    "uci.edu": "UCI",
    "mail.sysu.edu.cn": "SYSU",
    "mail2.sysu.edu.cn": "SYSU",
    "nd.edu": "Notre Dame",
    "jd.com": "Jingdong",
    "unimelb.edu.au": "UniMelb",
    "mcgill.ca": "McGill",
    "nju.edu.cn": "Nanjing",
    "smu.edu.sg": "SMU",
    "deepmind.com": "DeepMind",
    "uic.edu": "UIC",
    "sutd.edu.sg": "SUTD",
    "umontreal.ca": "UdeM",
    "wisc.edu": "UW-Madison",
    "hbku.edu.qa": "HBKU",
    "tu-darmstadt.de": "TU Darmstadt",
    "harvard.edu": "Harvard",
    "ttic.edu": "TTIC",
    "openai.com": "OpenAI",
    "huggingface.co": "HuggingFace",
}


domain_to_very_short = {
    "microsoft.com": "Microsoft",
    "google.com": "Google",
    "cmu.edu": "CMU",
    "stanford.edu": "Stanford",
    "tsinghua.edu.cn": "Tsinghua",
    "amazon.com": "Amazon",
    "fb.com": "Meta",
    "pku.edu.cn": "PKU",
    "washington.edu": "UW",
    "alibaba-inc.com": "Alibaba",
    "ac.cn": "CAS",
    "mit.edu": "MIT",
    "tencent.com": "Tencent",
    "ibm.com": "IBM",
    "illinois.edu": "UIUC",
    "sjtu.edu.cn": "SJTU",
    "huawei.com": "Huawei",
    "nyu.edu": "NYU",
    "berkeley.edu": "Berkeley",
    "zju.edu.cn": "ZJU",
    "allenai.org": "AllenAI",
    "gatech.edu": "Georgia Tech",
    "fudan.edu.cn": "Fudan",
    "ust.hk": "HKUST",
    "usc.edu": "USC",
    "ntu.edu.sg": "NTU",
    "cam.ac.uk": "Cambridge",
    "ucla.edu": "UCLA",
    "ucsd.edu": "UCSD",
    "utexas.edu": "UT",
    "hit.edu.cn": "HIT",
    "ustc.edu.cn": "USTC",
    "ethz.ch": "ETH Zurich",
    "ed.ac.uk": "Edinburgh",
    "columbia.edu": "Columbia",
    "jhu.edu": "JHU",
    "nus.edu.sg": "NUS",
    "ucsb.edu": "UCSB",
    "umass.edu": "UMass",
    "uva.nl": "UvA",
    "ox.ac.uk": "Oxford",
    "cuhk.edu.hk": "CUHK",
    "umich.edu": "UMich",
    "kaist.ac.kr": "KAIST",
    "ruc.edu.cn": "RUC",
    "uwaterloo.ca": "Waterloo",
    "seas.upenn.edu": "UPenn",
    "buaa.edu.cn": "BUAA",
    "princeton.edu": "Princeton",
    "navercorp.com": "Naver",
    "cornell.edu": "Cornell",
    "umd.edu": "UMD",
    "salesforce.com": "Salesforce",
    "adobe.com": "Adobe",
    "bytedance.com": "ByteDance",
    "snu.ac.kr": "SNU",
    "ucl.ac.uk": "UCL",
    "hku.hk": "HKU",
    "monash.edu": "Monash",
    "isi.edu": "ISI",
    "baidu.com": "Baidu",
    "epfl.ch": "EPFL",
    "inria.fr": "Inria",
    "nvidia.com": "NVIDIA",
    "ubc.ca": "UBC",
    "ntu.edu.tw": "NTU Taiwan",
    "di.ku.dk": "KU",
    "unc.edu": "UNC",
    "osu.edu": "OSU",
    "cis.lmu.de": "LMU Munich",
    "u.nus.edu": "NUS",
    "sheffield.ac.uk": "Sheffield",
    "bupt.edu.cn": "BUPT",
    "westlake.edu.cn": "Westlake",
    "samsung.com": "Samsung",
    "northeastern.edu": "Northeastern",
    "asu.edu": "ASU",
    "pjlab.org.cn": "Shanghai AI",
    "utoronto.ca": "U of T",
    "whu.edu.cn": "WHU",
    "mila.quebec": "Mila",
    "yale.edu": "Yale",
    "psu.edu": "Penn State",
    "uci.edu": "UCI",
    "sysu.edu.cn": "SYSU",
    "nd.edu": "Notre Dame",
    "jd.com": "Jingdong",
    "unimelb.edu.au": "UniMelb",
    "mcgill.ca": "McGill",
    "nju.edu.cn": "Nanjing",
    "smu.edu.sg": "SMU",
    "deepmind.com": "DeepMind",
    "uic.edu": "UIC",
    "sutd.edu.sg": "SUTD",
    "umontreal.ca": "UdeM",
    "wisc.edu": "UW-Madison",
    "hbku.edu.qa": "HBKU",
    "tu-darmstadt.de": "TU Darmstadt",
    "harvard.edu": "Harvard",
    "ttic.edu": "TTIC",
    "openai.com": "OpenAI",
    "huggingface.co": "HuggingFace",
    "lmu.de": "LMU",
    "tau.ac.il": "TAU",
    "huji.ac.il": "HUJI",
    "technion.ac.il": "Technion",
    "ucdavis.edu": "UCD",
    "u-tokyo.ac": "UTokyo",
    "purdue.edu": "Purdue",
    "uzh.ch": "UZH",
    "iiit.ac.in": "IIIT Hyderabad",
    "qmul.ac.uk": "QMUL",
    "ualberta.ca": "UAlberta",
    "rutgers.edu": "Rutgers",
    "uga.edu": "UGA",
    "whu.edu.cn": "WHU",
    "auckland.ac.nz": "Auckland",
    "uq.edu.au": "UQ",
    "apple.com": "Apple",
    "mbzuai.ac.ae": "MBZUAI",
    "cuhk.edu": "CUHK",
}


top_10_us_institution_domains = [
    "cmu.edu",
    "stanford.edu",
    "washington.edu",
    "illinois.edu",
    "berkeley.edu",
    "nyu.edu",
    "mit.edu",
    "usc.edu",
    "ucla.edu",
    "ucsd.edu",
]

top_10_chinese_institution_domains = [
    "tsinghua.edu.cn",
    "pku.edu.cn",
    "ac.cn",
    "sjtu.edu.cn",
    "zju.edu.cn",
    "fudan.edu.cn",
    "hit.edu.cn",
    "ustc.edu.cn",
    "ruc.edu.cn",
    "buaa.edu.cn",
]