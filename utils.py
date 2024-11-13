'''
tool functions
'''
import math
import datetime
import numpy as np
import pandas as pd
import gzip
from pandarallel import pandarallel


def prinT(str_to_print: str):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "--", str_to_print)
    

def pow10ceil(x):
    # 101, 500 -> 1000
    return 10**math.ceil(math.log(x, 10))


def quick_look(num_row: int=1e+4):
    prinT('take a quick look (%d rows) at Paper.txt file...' %num_row)
    col_index_list = [0, 3, 7, 11, 12, 21]
    data_types = {'PaperID': np.uint32,
                  'DocType': 'category',
                  'Year': np.uint16,
                  'JournalID': np.uint32,
                  'ConferenceSeriesID': np.uint32,
                  'OriginalVenue': 'object'}
    col_names = list(data_types.keys())
    with gzip.open('/mnt/data/MAG/MAG-20211206/mag/Papers.txt.gz', mode="rt") as paper_file:
        prinT("start loading a chunk")
        chunk = pd.read_csv(paper_file, 
                            names=col_names,
                            usecols=col_index_list,
                            nrows=num_row, 
                            sep='\t', 
                            low_memory=False,
                            error_bad_lines=False)
        prinT("finished")
        return chunk
    
    
def get_paper_detailed_df(chunksize: int=1e+6):
    '''
    1. Filter all papers, drop unuseful columns,
    2. only keep those 1) with a broken date; 2)from journals and conferences, and
    3. save the dataframe as a .pkl file.
    4. keep more infomation (than the above function), like titles, etc.
    '''
    pandarallel.initialize(progress_bar=False)
    
    prinT('generating \'paper_detailed_df\' from the original file...')
    col_index_list = [0, 2, 3, 4, 7, 11, 12, 18, 19, 21]
    data_types = {'PaperID': np.uint32, 
                  'Doi': 'object',
                  'DocType': 'category',
                  'PaperTitle': 'object',
                  'Year': 'datetime64[ns]',
                  'JournalID': np.uint32,
                  'ConferenceSeriesID': np.uint32,
                  'ReferenceCount': np.uint32, 
                  'CitationCount': np.uint32,
                  'OriginalVenue': 'object'}
    col_name_list = list(data_types.keys())
    col_name_list_without_DocType = list(data_types.keys())
    col_name_list_without_DocType.remove('DocType')
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
            
            chunk.loc[:,['DocType']] = chunk.loc[:,['DocType']].fillna('NoID')
            chunk.loc[:,col_name_list_without_DocType] = chunk.loc[:,col_name_list_without_DocType].fillna(0)
            
            chunk = chunk[col_name_list].astype(data_types)
            prinT("finish loading a chunk, start filtering it")
            
            # drop rows which are not from Journals or Conferences
            chunk.loc[:,'VenueID'] = chunk[['DocType','JournalID','ConferenceSeriesID', 'OriginalVenue']].apply(lambda x: 
                                                            x['JournalID'] if x['DocType'] == 'Journal'
                                                            else x['ConferenceSeriesID'] if x['DocType'] == 'Conference'
                                                            else x['OriginalVenue'] if (x['DocType'] == 'NoID') & (x['OriginalVenue'] != 0)
                                                            else np.nan, 
                                                            axis=1)
            chunk.dropna(subset=['VenueID'], inplace=True)
            
            # append the chunk into paper_df
            paper_df = pd.concat([paper_df, chunk], ignore_index=True)
            prinT("finishing filtering %d rows out of %d rows, current total number of selected rows: %d" %(len(chunk), chunksize, len(paper_df)))
            
    paper_df = paper_df.set_index('Year')
    paper_df = paper_df.sort_index()
    paper_df.to_pickle('/media/sdb/p2v/pickles/paper_detailed.pkl')
    prinT("The file \'paper_detailed.pkl\' is ready")
    return paper_df