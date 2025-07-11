'''
Functions for filtering original dataset and generating dataframes to store results
'''
import pandas as pd
import numpy as np
import gzip
import pickle
from pandarallel import pandarallel
from utils import prinT
import os
from sqlalchemy import create_engine
    
    
def get_paper_df(chunksize: int=1e+6):
    '''
    1. Filter all papers, drop unuseful columns,
    2. only keep those 1) with a valid year of publish; 2)from journals and conferences, and
    3. save the dataframe as a .pkl file.
    '''
    
    prinT('generating \'paper_df\' from the original file...')
    pandarallel.initialize(progress_bar=False)
    col_index_list = [0, 3, 7, 11, 12]
    data_types = {'PaperID': np.uint32,
                  'DocType': 'category',
                  'Year': 'datetime64[ns]',
                  'JournalID': np.uint32,
                  'ConferenceSeriesID': np.uint32,
                  }
    col_name_list = list(data_types.keys())
    paper_df = pd.DataFrame()
    
    with gzip.open('/mnt/data/MAG/MAG-20211206/mag/Papers.txt.gz', mode="rt") as paper_file:
        prinT("start loading a chunk")
        for chunk in pd.read_csv(paper_file, 
                                 names=col_name_list,
                                 usecols=col_index_list,
                                 chunksize=chunksize, 
                                 sep='\t',
                                 low_memory=False,
                                 error_bad_lines=False):
            # these two steps delete rows with non-standard 'Year' values
            chunk['Year'] = pd.to_datetime(chunk['Year'], format='%Y', errors='coerce')
            chunk.dropna(subset=['Year'], inplace=True)
            
            chunk.loc[:,col_name_list] = chunk.loc[:,col_name_list].fillna(0)
            chunk = chunk[col_name_list].astype(data_types)
            prinT("finish loading a chunk, start filtering it")
            
            # drop rows which are not from Journals or Conferences
            chunk.loc[:,'VenueID'] = chunk[['DocType','JournalID','ConferenceSeriesID']].parallel_apply(lambda x: 
                                                            x['JournalID'] if x['DocType'] == 'Journal'
                                                            else x['ConferenceSeriesID'] if x['DocType'] == 'Conference'
                                                            else np.nan, 
                                                            axis=1)
            chunk.dropna(subset=['VenueID'], inplace=True)
            
            # keep 'PaperID', 'DocType', 'Year', and "VenueID" only, and append the chunk into paper_df
            paper_df = pd.concat([paper_df, chunk[['PaperID','DocType', 'Year','VenueID']]], ignore_index=True)
            prinT("finishing filtering %d rows out of %d rows, current total number of selected rows: %d" %(len(chunk), chunksize, len(paper_df)))
            
    paper_df = paper_df.set_index('Year')
    paper_df = paper_df.sort_index()
    paper_df.to_pickle('/media/sdb/p2v/pickles/paper.pkl')
    prinT("The file \'paper.pkl\' is ready")
    return paper_df


# def get_paper_set(paper_df: pd.DataFrame):
#     '''
#     Reuturn a ndarray containing a set of unique papers from paper_df, and store it into a .pkl file
#     '''
    
#     prinT('generating \'paper_set\', start counting unique PIDs...')
#     paper_set = pd.unique(paper_df['PaperID'])
#     prinT("finish counting unique PIDs, start writing into the file")
#     # Save paper_set into .pkl file
#     with open("/media/sdb/p2v/pickles/paper_set.pkl", "wb") as set_file:
#         pickle.dump(paper_set,set_file)
#     prinT("The file \'paper_set.pkl\' is ready")
    
#     return paper_set
    
    
def get_ref_df(paper_df: pd.DataFrame, chunksize: int=1e+8):
    '''
    1. Filter all citations, keep only journal paper and conference paper, and
    2. save the dataframe as a .pkl file
    '''
    
    prinT('generating \'paper_set\', start counting unique PIDs...')
    paper_set = pd.unique(paper_df['PaperID'])
    prinT("finish")
    
    prinT('generating \'ref_df\' from the original file...')
    PaperRef_col_names = ['PaperID','PaperReferenceID']
    data_types = {'PaperID': np.uint32, 'PaperReferenceID': np.uint32}
    ref_df = pd.DataFrame()

    prinT("start loading a chunk")
    with gzip.open('/mnt/data/MAG/MAG-20211206/mag/PaperReferences.txt.gz', mode="rt") as ref_file:
        for chunk in pd.read_csv(ref_file, 
                                 names=PaperRef_col_names,
                                 chunksize=chunksize, 
                                 low_memory=False,
                                 sep='\t'):
            prinT("finish loading a chunk, start filtering it")
            chunk = chunk[['PaperID','PaperReferenceID']].astype(data_types)
            # Drop rows which are not from Journals or Conferences
            chunk.query('PaperID in @paper_set and PaperReferenceID in @paper_set', inplace=True)
            
            # keep 'PaperID', 'Year', and "VenueID" only, and append the chunk into paper_df
            ref_df = pd.concat([ref_df, chunk], ignore_index=True)
            
            prinT("finishing filtering %d rows out of %d rows, current total number of selected rows: %d" %(len(chunk), chunksize, len(ref_df)))
    ref_df.set_index('PaperID', inplace=True, drop=True)
    ref_df.to_pickle('/media/sdb/p2v/pickles/ref.pkl')
    prinT("file \'ref.pkl\' is ready.")
    
    return ref_df


def get_data_for_a_decade(start_year: str, end_year: str, paper_df: pd.DataFrame, ref_df: pd.DataFrame):
    prinT("----------------------------------")
    prinT("filtering papers published between {start_year}-{end_year}...".format(start_year=start_year, end_year=end_year))
    paper_df = paper_df[start_year:end_year]
    prinT("finish. %d papers in this decade" %len(paper_df))
    paper_set = pd.unique(paper_df['PaperID'])
    prinT("querying reference records from ref_df...")
    ref_df = ref_df.query('PaperID in @paper_set and PaperReferenceID in @paper_set')
    prinT("finish. %d reference records in this decade" %len(ref_df))
    
    dirs = '/media/sdb/p2v/pickles/decades/%s_to_%s/' %(start_year, end_year)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    paper_df.to_pickle('/media/sdb/p2v/pickles/decades/%s_to_%s/paper.pkl' %(start_year, end_year))
    ref_df.to_pickle('/media/sdb/p2v/pickles/decades/%s_to_%s/ref.pkl' %(start_year, end_year))
    

def make_MAG_venue_info_df():
    '''
    1. Create a DataFrame which contains 'VenueType', 'VenueID', and 'OriginalVenue' information, and 
    2. store it into 'MAG_venue_info_df.pkl' 
    '''
    
    prinT("start loading paper_detailed.pkl")
    with open('/media/sdb/p2v/pickles/paper_detailed.pkl', 'rb') as df_file:
        MAG_paper_detailed_df = pickle.load(df_file)
    prinT("finish")
    
    pandarallel.initialize(progress_bar=False)
    prinT("preparing VID for papers...")
    MAG_venue_info_df = MAG_paper_detailed_df[['DocType','OriginalVenue']]
    MAG_venue_info_df.loc[:,'VenueID'] = MAG_paper_detailed_df[['DocType','JournalID','ConferenceSeriesID']].parallel_apply(lambda x: 
                                                            x['JournalID'] if x['DocType'] == 'Journal'
                                                            else x['ConferenceSeriesID'] if x['DocType'] == 'Conference'
                                                            else np.nan, 
                                                            axis=1)
    MAG_venue_info_df['VenueID'] = pd.to_numeric(MAG_venue_info_df['VenueID'], errors='coerce')
    MAG_venue_info_df.dropna(subset=['VenueID'], inplace=True)
    MAG_venue_info_df['VenueID'] = MAG_venue_info_df['VenueID'].astype(np.int64)
    MAG_venue_info_df.drop_duplicates(subset=['VenueID'], inplace=True)
    
    MAG_venue_info_df.set_index('VenueID', inplace=True)
    MAG_venue_info_df.rename(columns={'DocType': 'VenueType'}, inplace=True)

    MAG_venue_info_df[['VenueType','OriginalVenue']].to_pickle('/media/sdb/p2v/pickles/MAG_venue_info_df.pkl')
    prinT("finish. MAG_venue_info_df saved")
    return MAG_venue_info_df


def make_labeled_journal_info_df():
    '''
    1. Create a DataFrame containing: 
       'VenueType', 'VenueID', 'OriginalVenue', 'ScopusName', 'ScopusCategory', 'VenueID', and 'InceptionYear' information, and 
    2. store it into 'labeled_journal_info_df.pkl'
    '''
    from dotenv import load_dotenv
    import os

    load_dotenv()
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_PORT = os.getenv("MYSQL_PORT")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DB = os.getenv("MYSQL_DB")
    engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8'
                           % (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB))
    sql = 'SELECT * FROM container WHERE scopus_cat IS NOT NULL AND mag_journal_id IS NOT NULL'

    scopus_df = pd.read_sql(sql, engine)[['scopus_name', 'scopus_cat', 'mag_journal_id', 'inception']]
    scopus_df.columns = ['ScopusName', 'ScopusCategory', 'VenueID', 'InceptionYear']
    scopus_df = scopus_df.astype({'ScopusName': 'object', 
                                  'ScopusCategory': 'object', 
                                  'VenueID': 'int', 
                                  'InceptionYear': 'int'},
                                   errors='ignore')
    scopus_df.drop_duplicates(subset=['VenueID'], inplace=True)
    scopus_df.set_index('VenueID', inplace=True)
    
    with open('/media/sdb/p2v/pickles/MAG_venue_info_df.pkl', 'rb') as df_file:
        MAG_venue_info_df = pickle.load(df_file)
    labeled_journal_info_df = MAG_venue_info_df.merge(scopus_df, left_index=True, right_index=True)
    labeled_journal_info_df.to_pickle('/media/sdb/p2v/pickles/labeled_journal_info_df.pkl')
    prinT("finish. labeled_journal_info_df saved")
    
    return labeled_journal_info_df


