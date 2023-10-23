import os
import re
import pandas as pd
import pdftotext
import multiprocessing
import time
from tqdm import tqdm
from collections import Counter

'''
If regenerating metadata, these paths will need to be changed to wherever you download the data.
'''
BASE_DATA_DIR = '/share/pierson/raj/LLM_bibliometrics_v2/base_data'
PAPER_PDF_DIR = '/share/pierson/raj/arXiv_LLM_bibliometrics/arxiv-data-out/tarpdfs'
PROCESSED_DATA_DIR = '/share/pierson/raj/LLM_bibliometrics_v2/processed_data'
EMBEDDINGS_DIR = '/share/pierson/raj/LLM_bibliometrics_v2/processed_data/embeddings'
S2_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 's2_data_lm_papers.json')
GENDER_PATH = os.path.join(PROCESSED_DATA_DIR, 'nqg_author_predicted_genders.json')

'''
This is a fairly wide list of language model-related terms, to capture a broad set of relevant papers.
'''
LM_WORD_LIST_UNCASED = [
    'language model',
    'pretrained language model',
    'large language model',
    'foundation model',
]
LM_WORD_LIST_CASED = [
    'BERT',
    'XLNet',
    'GPT-2', 'GPT-3', 'GPT-4', 'GPT-Neo', 'GPT-J', 
    'ChatGPT',
    'PaLM', 
    'LLaMA',
]

def get_lm_terms(title_or_abstract):
    '''
    Get a list of language model-related terms in a string.
    Meant to be used on a title or abstract.
    '''
    lm_terms = []
    for word in LM_WORD_LIST_UNCASED:
        if word in title_or_abstract.lower():
            lm_terms.append(word)
    for word in LM_WORD_LIST_CASED:
        if word in title_or_abstract:
            lm_terms.append(word)
    return lm_terms


def add_semantic_scholar_info(lm_metadata):
    '''
    Adds semantic scholar citationCount to the lm_metadata dataframe.

    Ignore 'influentialCitationCount' for now, since it's mostly 0.
    '''
    s2_data = pd.read_json(S2_DATA_PATH, lines=True, orient='records', dtype={'id': str})

    # Print number of papers which we have S2 data for
    papers_with_s2_data = len(lm_metadata[lm_metadata.id.isin(s2_data.id)].id.tolist())
    print(f"Papers with S2 data: {papers_with_s2_data} out of {len(lm_metadata)} ({papers_with_s2_data/len(lm_metadata)*100:.1f}%)")

    lm_metadata = lm_metadata.merge(s2_data[['id', 'citationCount']], on='id', how='left')
    return lm_metadata
    

def add_affiliation_info(metadata):
    '''
    Function to add affiliation info to metadata dataframe.

    Takes the dataframe 'metadata' with a column 'emails' and returns a new dataframe with three new columns:
    'domains': the set of unique domains extracted from the emails
    'industry': TRUE if the paper has an industry domain listed, FALSE otherwise
    'academic': TRUE if the paper has an academic domain listed, FALSE otherwise
    '''
    newdf = metadata.copy()
    newdf['domains'] = newdf.apply(lambda x: emails_to_unique_domains(x['emails']), axis=1) # Get unique domains in a list, consolidating some common duplicates
    domain_dict = get_domain_dict(newdf, 10) # Get a dict that maps domains onto parent domains
    newdf['domains'] = newdf.apply(lambda x: clean_domains(x['domains'], domain_dict), axis=1) # Remove generic domains
    newdf['industry'] = newdf.apply(lambda x: contains_industry(x['domains']), axis=1)
    newdf['academic'] = newdf.apply(lambda x: contains_academic(x['domains']), axis=1)
    newdf = newdf.drop(columns=['emails'])

    # Print number of papers with either industry or academic affiliation
    papers_with_affiliation = len(newdf[(newdf.industry == True) | (newdf.academic == True)].id.tolist())
    print(f"Papers with an identified affiliation: {papers_with_affiliation} out of {len(newdf)} ({papers_with_affiliation/len(newdf)*100:.1f}%)")
    
    return newdf


def add_majority_female_annotation(metadata, 
                                   female_prop_threshold=0.5, 
                                   gender_col='inferred_female_frac_nqg_uncertainty_threshold_0.100', 
                                   verbose=False):
    """
    Given a female_prop_threshold \in [0, 1], annotate the metadata with a column indicating whether the paper is above or below the threshold in terms of its inferred author female frac.  
    """
    gender_df = pd.read_json(GENDER_PATH, lines=True, orient='records', dtype={'id': str})

    ids_in_metadata = set(metadata['id'])
    assert ids_in_metadata.issubset(set(gender_df['id']))

    # Below line requires both gender_df and metadata to be sorted by id
    gender_df = gender_df.loc[gender_df['id'].map(lambda x: x in ids_in_metadata)]

    # Print the number of papers where all predicted genders are unknown          
    if verbose:
        print(f"gender_df shape after subsetting to ids in metadata: {gender_df.shape}")
        print("Proportion of missing values in gender col %2.2f" % gender_df[gender_col].isnull().mean())

    print("Labeling papers using a predicted-female proportion threshold of %2.2f" % female_prop_threshold)
    gender_df['above_pred_female_threshold'] = None
    gender_df.loc[gender_df[gender_col] >= female_prop_threshold, 'above_pred_female_threshold'] = True
    gender_df.loc[gender_df[gender_col] < female_prop_threshold, 'above_pred_female_threshold'] = False

    metadata = pd.merge(metadata, gender_df[['id', 'above_pred_female_threshold', gender_col]], 
                        left_on='id', right_on='id', how='inner', validate='one_to_one')

    print("%2.1f%% of papers with data are above female threshold; %2.1f%% of papers have at least one name prediction" 
          % (100*metadata['above_pred_female_threshold'].mean(), 100*(~pd.isnull(metadata['above_pred_female_threshold'])).mean()))

    return metadata


def fix_author_name_list(authors):
    """
    Fix author names in the metadata DataFrame.

    Convert each list of lists into a single list, where each sublist is converted to a string
    The string should be the concatenation of the first and last names of each author, i.e.
    [[Streinu, Ileana, ], [Theran, Louis, ]] --> ['Ileana Streinu', 'Louis Theran']

    This function also needs to handle edge cases like "Zhizhang (David)Chen", which gets parsed incorrectly and treated as two separate authors.

    :param authors: list of lists of author names
    :return: list of author names as [first last, etc]
    """
    output = []
    i = 0
    while i < len(authors):
        if len(authors[i][0]) > 0 and len(authors[i][1]) > 0:
            # Normal case
            output.append(f"{authors[i][1]} {authors[i][0]}")
            i += 1
        elif len(authors[i][0]) == 0:
            # Empty last name, so this element should just be skipped
            i += 1
            continue
        elif len(authors[i][1]) == 0 and len(authors[i]) == 4 and i+1 < len(authors) and len(authors[i+1][1]) == 0 and len(authors[i+1]) == 3: 
            # This is the weird edge case: use the next element to construct the name
            output.append(f"{authors[i][0]} {authors[i][3]} {authors[i+1][0]}")
            i += 2
        elif len(authors[i][1]) == 0:
            # This is another edge case where the author only has a last name listed
            # Just take the last name
            output.append(f"{authors[i][0]}")
            i += 1
        else:
            raise ValueError(f"Unexpected case: {authors[i]}")

    return output


def get_paper_filenames(row):
    """
    Get the path to the fulltext PDF for a given paper.
    We can just take the v1 of each paper, since we are only using fulltexts for affiliation extraction.

    :param row: row of the metadata DataFrame
    :return: a list of paths to the fulltext PDFs version 1's for each paper
    """
    arxiv_id = row['id']
    monthyear = row['id'].split('.')[0]
    return (f"{arxiv_id}v1", monthyear)


def convert_pdf_to_text(pdf_path, txt_path):
    try:
        if not os.path.exists(txt_path):
            # Filter out the specific Poppler warning
            with open(pdf_path, "rb") as pdf_file:
                pdf_text = pdftotext.PDF(pdf_file)
                text_content = "\n".join(pdf_text)
                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(text_content)
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")


def convert_pdfs_serial(pdf_paths, txt_paths):
    for i in tqdm(range(len(pdf_paths)), miniters=100):
        convert_pdf_to_text(pdf_paths[i], txt_paths[i])


def convert_pdfs_in_parallel(pdf_paths, txt_paths):
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Start time
    start_time = time.time()

    # Use pool.starmap to apply the conversion function in parallel
    for pdf_path, txt_path in zip(pdf_paths, txt_paths):
        pool.apply_async(convert_pdf_to_text, args=(pdf_path, txt_path))

    # Close the pool
    pool.close()
    pool.join()

    # Print total elapsed time
    total_time = time.time() - start_time
    print(f"All {len(pdf_paths)} PDFs processed in {total_time:.2f} seconds")


def create_symlinks_to_papers(lm_metadata):
    """
    Take a dataframe of papers, and create symlinks in the desired directory to the fulltext PDFs for each paper.
    This is necessary because we have to download all the fulltext PDFs from GCP, so for better organization, we create symlinks to the papers we need.
    """
    # PAPER_PDF_DIR is where you have all arXiv fulltexts downloaded
    llm_pdf_dir = os.path.join(BASE_DATA_DIR, 'llm_paper_pdfs')

    # Make all the monthyear subdirs in the destination directory
    all_monthyears = lm_metadata['id'].apply(lambda x: x.split('.')[0]).unique()
    for monthyear in all_monthyears:
        os.makedirs(f'{llm_pdf_dir}/{monthyear}', exist_ok=True)

    # Create symlink from os.path.join(llm_pdf_dir, monthyear, filename) pointing to os.path.join(download_pdf_dir, monthyear, filename)
    paper_filenames = lm_metadata.apply(get_paper_filenames, axis=1).tolist()
    for filename, monthyear in paper_filenames:
        # If file doesn't exist in PAPER_PDF_DIR, skip (some PDFs are missing)
        if not os.path.exists(os.path.join(PAPER_PDF_DIR, monthyear, f'{filename}.pdf')):
            print(f'Missing paper: {os.path.join(PAPER_PDF_DIR, monthyear, f"{filename}.pdf")}')
            continue
        else:
            os.symlink(os.path.join(PAPER_PDF_DIR, monthyear, f'{filename}.pdf'), os.path.join(llm_pdf_dir, monthyear, f'{filename}.pdf'))


email_ignore_list = ['permissions@acm.org', 'pubs-permissions@ieee.org', 'reprints@ieee.org', 'info@vldb.org']
def get_all_emails(row, fulltext_dir, n_lines_to_check=100):
    '''
    Look for email addresses in the first N lines of paper fulltexts.
    Parameters:
    - row: row of metadata dataframe
    - n_lines_to_check: top N number of lines to check for email addresses
    - fulltext_dir: directory where fulltext files are stored
    '''
    # Get the path to all possible fulltext files
    fulltext_paths = []
    for version_dict in row.versions:
        version = version_dict['version']
        fulltext_path = os.path.join(fulltext_dir,
                                     row.id.split('.')[0], 
                                     f'{row.id}{version}.txt')
        fulltext_paths.append(fulltext_path)
    fulltext_paths = fulltext_paths[::-1]

    # Open the fulltext file
    i = 0
    while i < len(fulltext_paths):
        if os.path.exists(fulltext_paths[i]):
            break
        else: i += 1
    if i == len(fulltext_paths):
        return None
    
    with open(fulltext_paths[i], 'r') as f:
        # Read the first n_lines_to_check lines
        all_matches = []
        for i in range(n_lines_to_check):
            line = f.readline()
            match_on_line = False
            # Do a regex match to find an email address, following the spec outlined above
            # Make sure the final string after the period does not only contain digits
            matches = re.finditer(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}(?!\d)', line)
            for match in matches:
                if match and match.group().lower() not in email_ignore_list:
                    all_matches.append(match.group())
                    match_on_line = True
            
            # Also, if there's a match to the {author1, author2}@domain pattern, return it
            matches = re.finditer(r'\{[\w\.,| ]+\}@[\w\.-]+\.[a-zA-Z]{2,}(?!\d)', line)
            for match in matches:
                if match and match.group().lower() not in email_ignore_list:
                    all_matches.append(match.group())
                    match_on_line = True
            
            # Finally, check for any matches to @domain, but only in the first 25 lines.
            # This is to catch cases where the "@gmail.com", for example, is hanging
            if not match_on_line and i < 25:
                matches = re.finditer(r'@[\w\.-]+\.[a-zA-Z]{2,}(?!\d)', line)
                for match in matches:
                    if match and match.group().lower() not in email_ignore_list:
                        all_matches.append(match.group())

        return all_matches
    

# For a dataframe with a field giving a list of unique domains, get a Counter() that records 
# the number of papers from each domain
def get_domain_counter(df):    
    domain_counter = Counter()

    for row in df['domains']:
        for domain in row:
            domain_counter[domain] += 1
            
    return domain_counter


# Returns all "superdomains" of a given string 
# (e.g., 'a.cs.cornell.edu' has superdomains ['a.cs.cornell.edu', 'cs.cornell.edu', 'cornell.edu']
def get_superdomains(domain):
    superdomains = []
    split_string = domain.split('.')
    
    # Iterate over the parts of the string after each period
    for i in range(len(split_string)):
        substring = '.'.join(split_string[i:])
        superdomains.append(substring)
    
    return superdomains


# Given a dataframe with a 'domains' column, produces a dict that maps domains onto parent domains 
# (e.g., 'cs.cornell.edu' -> 'cornell.edu')
def get_domain_dict(df, threshold):    
    domain_counter = get_domain_counter(df)  
    full_domain_list = domain_counter.keys()
    
    domain_dict = {}
    for domain in full_domain_list:
        parent_domain = domain
        for substring in get_superdomains(parent_domain):
            if substring in full_domain_list and domain_counter[substring]>=threshold:
                parent_domain = substring
        domain_dict[domain] = parent_domain
    
    return domain_dict


# Cleans a list of domains by removing generic email addresses (such as gmail.com) and mapping domains onto their parent domains
# according to domain_dict
def clean_domains(domains, domain_dict):
    clean_domains = set()
    domains_generic = ['gmail', 'hotmail', '163.com', 'ieee.org', 'acm.org', 'outlook.com', 'foxmail.com', '126.com']
    
    if domains:
        for domain in domains:
            if not any(substr in domain for substr in domains_generic):
                clean_domains.add(domain_dict[domain])
    
    return list(clean_domains)                
        

# Takes a list of emails and returns a list of unique domains
def emails_to_unique_domains(emails):
    domains = set()
    if emails:
        for x in emails:
            # First, manually add some parent domains that are missed
            if 'ac.cn' in x:
                domain = 'ac.cn'
            elif 'uni-muenchen.de' in x:
                domain = 'uni-muenchen.de'
            elif 'u-tokyo.ac' in x:
                domain = 'u-tokyo.ac'
            elif 'lmu.de' in x:
                domain = 'lmu.de'
            elif 'harvard.edu' in x:
                domain = 'harvard.edu'
            elif 'bits-pilani.ac.in' in x:
                domain = 'bits-pilani.ac.in'
            elif 'a-star.edu.sg' in x:
                domain = 'a-star.edu.sg'
            elif 'bosch.com' in x:
                domain = 'bosch.com'
            elif 'xjtu.edu.cn' in x:
                domain = 'xjtu.edu.cn'
            elif 'sysu.edu.cn' in x:
                domain = 'sysu.edu.cn'
            elif 'cuhk.edu' in x:
                domain = 'cuhk.edu'
            # Handle special cases where multiple domains refer to same institution
            elif 'meta.com' in x:
                domain = 'fb.com'
            elif 'naverlabs.com' in x:
                domain = 'navercorp.com'
            elif 'uw.edu' in x:
                domain = 'washington.edu'
            elif 'u.nus.edu' in x:
                domain = 'nus.edu.sg'
            elif 'isi.edu' in x:
                domain = 'usc.edu'
            elif 'ucdconnect.ie' in x:
                domain = 'ucd.ie'
            elif 'tsinghua.org.cn' in x:
                domain = 'tsinghua.edu'
            elif 'cislmu.org' in x:
                domain = 'lmu.de'
            elif 'kaist.edu' in x:
                domain = 'kaist.ac.kr'
            elif 'toronto.edu' in x:
                domain = 'utoronto.ca'
            # Handle departmental subdomains, only want the parent domain
            elif '@cs.' in x: 
                domain = x.split('@cs.')[1]
            elif '@cse.' in x:
                domain = x.split('@cse.')[1]
            else:
                domain = x.split('@')[1]
            domains.add(domain)
            
    return list(domains)

    
industry_domains = [
    'microsoft.com', 'google.com', 'fb.com', 'amazon.com', 'alibaba-inc.com', 'ibm.com', 'tencent.com', 'huawei.com',
    'salesforce.com', 'navercorp.com', 'baidu.com', 'adobe.com', 'nvidia.com', 'bytedance.com', 'samsung.com',
    'intel.com', 'deepmind.com', 'jd.com', 'apple.com', 'iflytek.com', 'bosch.com', 'openai.com', 'qq.com',
    'meituan.com', 'antgroup.com', 'yahoo.com', 'pingan.com.cn', 'huggingface.co', 'corp.netease.com',
    'sensetime.com', 'anthropic.com', 'peopletec.com', 'shannonai.com', 'sap.com',
    # Below are added for arXiv v2
    'bloomberg.net', 'asapp.com', 'lgresearch.ai', 'sony.com', 'taobao.com', 'siemens.com', 'ai21.com',
]

academic_domains = [
    'cmu.edu', 'stanford.edu', 'tsinghua.edu.cn', 'washington.edu', 'pku.edu.cn', 'mit.edu', 'ac.cn', 'usc.edu',
    'illinois.edu', 'nyu.edu', 'sjtu.edu.cn', 'berkeley.edu', 'gatech.edu', 'cam.ac.uk', 'ust.hk', 'fudan.edu.cn',
    'utexas.edu', 'ucsd.edu', 'ntu.edu.sg', 'ed.ac.uk', 'nus.edu.sg', 'zju.edu.cn', 'jhu.edu', 'columbia.edu',
    'ethz.ch', 'ucla.edu', 'hit.edu.cn', 'umass.edu', 'ucsb.edu', 'ustc.edu.cn', 'harvard.edu', 'upenn.edu',
    'ox.ac.uk', 'umich.edu', 'uva.nl', 'kaist.ac.kr', 'uwaterloo.ca', 'princeton.edu', 'cuhk.edu.hk', 'ucl.ac.uk',
    'umd.edu', 'buaa.edu.cn', 'snu.ac.kr', 'ruc.edu.cn', 'lmu.de', 'cornell.edu', 'ubc.ca', 'epfl.ch', 'osu.edu',
    'monash.edu', 'unc.edu', 'sysu.edu.cn', 'ntu.edu.tw', 'umontreal.ca', 'toronto.edu', 'hku.hk', 'sheffield.ac.uk',
    'inria.fr', 'uci.edu', 'westlake.edu.cn', 'uic.edu', 'unimelb.edu.au', 'yale.edu', 'tu-darmstadt.de', 'mila.quebec',
    'whu.edu.cn', 'uni-saarland.de', 'nd.edu', 'iiit.ac.in', 'dfki.de', 'psu.edu', 'bupt.edu.cn', 'mff.cuni.cz',
    'northeastern.edu', 'stonybrook.edu', 'purdue.edu', 'asu.edu', 'helsinki.fi', 'virginia.edu', 'hbku.edu.qa',
    'iitb.ac.in', 'korea.ac.kr', 'ttic.edu', 'imperial.ac.uk', 'rwth-aachen.de', 'nju.edu.cn', 'sutd.edu.sg',
    'uchicago.edu', 'duke.edu', 'vt.edu', 'mcgill.ca', 'uzh.ch', 'utoronto.ca', 'cardiff.ac.uk', 'manchester.ac.uk',
    'qmul.ac.uk', 'iiitd.ac.in', 'smu.edu.sg', 'tau.ac.il', 'suda.edu.cn', 'wisc.edu', 'hust.edu.cn', 'ecnu.edu.cn',
    'uit.edu.vn', 'technion.ac.il', 'idiap.ch', 'pjlab.org.cn', 'ims.uni-stuttgart.de', 'iitkgp.ac.in', 'rochester.edu',
    'umn.edu', 'ucdavis.edu', 'bit.edu.cn', 'arizona.edu', 'uni-mannheim.de', 'sydney.edu.au', 'dartmouth.edu',
    'huji.ac.il', 'ualberta.ca', 'surrey.ac.uk', 'rug.nl', 'iitk.ac.in', 'uts.edu.au', 'ucsc.edu', 'georgetown.edu',
    'rutgers.edu', 'gmu.edu', 'sc.edu', 'anu.edu.au', 'mpi-inf.mpg.de', 'unsw.edu.au', 'uni-heidelberg.de',
    'northwestern.edu', 'uu.nl', 'uio.no', 'brown.edu', 'hse.ru', 'ufl.edu', 'bu.edu', 'ut.ac.ir', 'rice.edu', 'kyoto-u.ac.jp',
    'utah.edu', 'tudelft.nl', 'aalto.fi', 'tamu.edu', 'tohoku.ac.jp', 'itu.dk', 'liverpool.ac.uk', 'emory.edu',
    'cuhk.edu.cn', 'bgu.ac.il', 'uq.edu.au', 'phystech.edu', 'uni.lu', 'uh.edu', 'kuleuven.be', 'tju.edu.cn',
    'buffalo.edu', 'iust.ac.ir', 'pitt.edu', 'ijs.si', 'iitm.ac.in', 'tum.de', 'ucf.edu', 'iitd.ac.in', 'upf.edu',
    'nii.ac.jp', 'unipi.it', 'queensu.ca', 'glasgow.ac.uk', 'kcl.ac.uk', 'xjtu.edu.cn', 'biu.ac.il', 'aueb.gr',
    'uestc.edu.cn', 'uni-hamburg.de', 'mq.edu.au', 'ehu.eus', 'uni-konstanz.de', 'uml.edu', 'xmu.edu.cn',
    'hanyang.ac.kr', 'kaust.edu.sa', 'usp.br', 'ugent.be', 'tuwien.ac.at', 'auckland.ac.nz', 'polyu.edu.hk',
    'tu-berlin.de', 'kit.edu', 'liacs.leidenuniv.nl', 'ntua.gr', 'adelaide.edu.au', 'nankai.edu.cn',
    'iu.edu', 'skoltech.ru', 'warwick.ac.uk', 'neu.edu.cn', 'cityu.edu.hk', 'uni-bielefeld.de', 'msu.edu',
    'scut.edu.cn', 'unitn.it', 'nudt.edu.cn', 'fri.uni-lj.si', 'utdallas.edu', 'unibocconi.it',
    'goa.bits-pilani.ac.in', 'seu.edu.cn', 'mbzuai.ac.ae', 'uga.edu', 'deakin.edu.au',
    'polymtl.ca', 'um.edu.mt', 'adaptcentre.ie', 'udc.es', 'dal.ca', 'bjtu.edu.cn', 'uni-tuebingen.de',
    'tib.eu', 'fbk.eu', 'kth.se', 'temple.edu', 'l3s.de', 'ucalgary.ca', 'vu.nl', 'rpi.edu', 'gu.se',
    'jku.at', 'univ-grenoble-alpes.fr', 'rit.edu', 'ru.nl', 'hw.ac.uk', 'vutbr.cz',
    'buet.ac.bd', 'uni-wuppertal.de', 'tecnico.ulisboa.pt', 'upb.ro', 'shanghaitech.edu.cn', 'sdu.edu.cn',
    'uottawa.ca', 'telecom-paris.fr', 
    # Below are added for arXiv v2
    'di.ku.dk', 'umbc.edu', 'stevens.edu', 'yorku.ca', 'ucr.edu', 'sustech.edu.cn',
    'uni-bonn.de', 'iastate.edu', 'connect.polyu.hk', 'um.edu.mo', 'jlu.edu.cn', 'nwpu.edu.cn',
    'yonsei.ac.kr', 'lehigh.edu', 'ucd.ie', 'gwu.edu', 'idea.edu.cn', 'vanderbilt.edu', 'ku.edu.tr',
    'research.gla.ac.uk', 'ncsu.edu', 'is.naist.jp', 'uoregon.edu', 'njust.edu.cn', 'fau.de',
    'leeds.ac.uk', 'uconn.edu', 'colorado.edu', 'bristol.ac.uk', 'tue.nl', 'uni-passau.de',
    'usf.edu', 'auburn.edu', 'uth.tmc.edu', 'iisc.ac.in', 'drexel.edu', 'newcastle.edu.au',
]


# returns TRUE if the domain is an industry domain
def is_industry_domain(domain):
    return any(substr in domain for substr in industry_domains)


# returns TRUE if the domain is an academic domain
def is_academic_domain(domain):
    return any(substr in domain for substr in academic_domains)


# returns TRUE if the list 'domains' contains an industry domain and FALSE otherwise
def contains_industry(domains):
    return any(substr in domain for domain in domains for substr in industry_domains)


# returns TRUE if the list 'domains' contains an academic domain and FALSE otherwise
def contains_academic(domains):
    return any(substr in domain for domain in domains for substr in academic_domains)