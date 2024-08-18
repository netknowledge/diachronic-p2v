import numpy as np
import pandas as pd
import pickle

from collections import Counter

from multiprocessing import cpu_count
import os

from gensim.models import word2vec, KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec

import seaborn as sns
from adjustText import adjust_text  

from preprocessing import get_paper_df, get_paper_set, get_ref_df, get_data_for_a_decade
from utils import prinT, pow10ceil
import random


class P2V:    
    def __init__(self, load_raw_MAG: bool=False):
        self.start_year_list = ['1800', '1810', '1820', '1830', '1840', '1850', '1860', '1870', '1880', '1890', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']
        self.end_year_list = ['1809', '1819', '1829', '1839', '1849', '1859', '1869', '1879', '1889', '1899', '1909', '1919', '1929', '1939', '1949', '1959', '1969', '1979', '1989', '1999', '2009', '2021']
        self.discipline2color = {'Chemical Engineering': '#6C0000',
                                 'Chemistry': '#9A0000',
                                 'Computer Science': '#FF5C29',
                                 'Earth and Planetary Sciences': '#FE0000',
                                 'Energy': '#FF7C80',
                                 'Engineering': '#D20000',
                                 'Environmental Science': '#D26B04',
                                 'Materials Science': '#FC9320',
                                 'Mathematics': '#FBFF57',
                                 'Physics and Astronomy': '#FFCC00',
                                 
                                 'Medicine' :'#7030A0',
                                 'Nursing': '#9900CC',
                                 'Veterinary': '#CC00FF',
                                 'Dentistry': '#A679FF',
                                 'Health Professions': '#CCB3FF',

                                 'Arts and Humanities': '#375623',
                                 'Business, Management and Accounting': '#187402',
                                 'Decision Sciences': '#16A90F',
                                 'Economics, Econometrics and Finance': '#8FA329',
                                 'Psychology': '#92D050',
                                 'Social Sciences': '#66FF66',

                                 'Agricultural and Biological Sciences': '#000099',
                                 'Biochemistry, Genetics and Molecular Biology': '#336699',
                                 'Immunology and Microbiology': '#0000F2',
                                 'Neuroscience': '#0099FF',
                                 'Pharmacology, Toxicology and Pharmaceutics': '#85D6FF',
                              
                                 'Multidisciplinary': '#000000'}
        
        self.disc2abbr = {"Biochemistry, Genetics and Molecular Biology": "Biochem",
                          "Earth and Planetary Sciences": "Earth",
                          "Medicine": "Med",
                          "Physics and Astronomy": "Phys",
                          "Agricultural and Biological Sciences": "Agri",
                          "Immunology and Microbiology": "Immuno",
                          "Chemistry": "Chem",
                          "Neuroscience": "Neuro",
                          "Materials Science": "Mat",
                          "Social Sciences": "Soc",
                          "Environmental Science": "Env",
                          "Engineering": "Eng",
                          "Pharmacology, Toxicology and Pharmaceutics": "Pharm",
                          "Psychology": "Psy",
                          "Arts and Humanities": "Arts",
                          "Mathematics": "Math",
                          "Veterinary": "Vet",
                          "Chemical Engineering": "ChemEng",
                          "Economics, Econometrics and Finance": "Econ",
                          "Nursing": "Nurs",
                          "Computer Science": "CS",
                          "Energy": "Energy",
                          "Dentistry": "Dent",
                          "Business, Management and Accounting": "Bus",
                          "Health Professions": "HealthPro",
                          "Decision Sciences": "Dec",

                          'Multidisciplinary': 'Multi'
                          }

        if load_raw_MAG:
            prinT('start loading \'paper_df\'...')
            with open('/media/sdb/p2v/pickles/paper.pkl', 'rb') as file:
                self.paper_df = pickle.load(file)
            prinT('finish.')
            
            prinT('start loading \'ref_df\'...')
            with open('/media/sdb/p2v/pickles/ref.pkl', 'rb') as file:
                self.ref_df = pickle.load(file)
            prinT('finish.')
        else:
            self.load_MAG_venue_info_df()
            self.load_labeled_journal_info_df()
        
    def create_paper_df(self, chunksize=4e+7):
        self.paper_df = get_paper_df(chunksize)
        
        
    def create_ref_df(self, chunksize=4e+8):
        self.ref_df = get_ref_df(self.paper_df, chunksize)
        
    
    def split_data_for_decades(self, shift=False):
        start_year_list = self.start_year_list
        end_year_list = self.end_year_list
        if shift:
            start_year_list = [str(int(year)+3) for year in start_year_list[-7:]]
            end_year_list = [str(int(year)+3) for year in end_year_list[-7:]]
            
        for i in range(len(start_year_list)):
            get_data_for_a_decade(start_year_list[i], end_year_list[i], self.paper_df, self.ref_df)
            prinT("{}/{} generated...".format(i+1, len(start_year_list)))
        
            
    def load_paper_df(self, full_load: bool=True, start_year: int=None, end_year: int=None):
        prinT('start loading \'paper_df\'...')
        if full_load:
            with open('/media/sdb/p2v/pickles/paper.pkl', 'rb') as file:
                self.paper_df = pickle.load(file)   
        else:
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/paper.pkl' %(start_year, end_year), 'rb') as file:
                self.target_paper_df = pickle.load(file)
        prinT('finish.')
    
    
    def load_ref_df(self, full_load: bool=True, start_year: int=None, end_year: int=None): 
        prinT('start loading \'ref_df\'...')
        if full_load:
            with open('/media/sdb/p2v/pickles/ref.pkl', 'rb') as file:
                self.ref_df = pd.read_pickle(file)
        else:
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/ref.pkl' %(start_year, end_year), 'rb') as file:
                self.target_ref_df = pd.read_pickle(file)
        prinT('finish.')


    def load_venue_name2NID_df(self, start_year: int=None, end_year: int=None): 
        # prinT('start loading \'venue_name2NID_df\'...')
        with open('/media/sdb/p2v/pickles/decades/%s_to_%s/venue_name2NID.pkl' %(start_year, end_year), 'rb') as file:
            self.venue_name2NID_df = pickle.load(file)
        # prinT('finish.')
    
    
    def load_walks(self, start_year: int, end_year: int, use_filtered_walks: False):
        
        if use_filtered_walks:
            prinT("start loading filtered walks...")
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/filtered_walks.pkl' %(start_year, end_year), 'rb') as file:
                self.walks = pickle.load(file)
        else:
            prinT("start loading filtered walks...")
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/walks.pkl' %(start_year, end_year), 'rb') as file:
                self.walks = pickle.load(file)
        prinT("finish.")
    
    
    def random_walk(self, num_walks):
        '''
        generate random walks unitl reaching expected number of walks. 
        As the raw reference file has triangle citation, we just terminate and discard a walk once a pid recurred during the walking. 
        We also discard walks with length one, as they provide no infromation about the journal citation flow.
        '''
        n = 0
        for pid_origin in self.target_paper_set:
            i = 0
            while i < num_walks:
                pwalk = []
                vwalk = []
                vid = self.target_paper_df.at[pid_origin,'VenueID']
                pwalk.append(pid_origin)
                vwalk.append(vid)
                pid = pid_origin
                while 1:
                    # ref_list: a Series containing PIDs of papers cited in a certain paper
                    try:
                        ref_list = self.target_ref_df.at[pid, 'PaperReferenceID']
                    except KeyError:
                        # the chosen paper has no reference
                        # print("No reference!")
                        break
                    if type(ref_list) == np.uint32: # there is only one paper in the ref_list
                        pid = ref_list
                    else:
                        pid = np.random.choice(ref_list)
                    if pid in pwalk: # pid reoccurrs
                        # print("PID reoccurrs!")
                        vwalk = []
                        break
                    vid = self.target_paper_df.at[pid,'VenueID']
                    pwalk.append(pid)
                    vwalk.append(vid)
                if len(vwalk) > 1:
                    self.walks.append(vwalk)
                i = i + 1                
            n = n + 1
            if n % int(len(self.target_paper_set)/5) == 0:
                    prinT("finish {}/{} papers...".format(n, int(len(self.target_paper_set))))
                            
                            
    def random_walk_for_decades(self, num_walks: int=5, shift=False):
        start_year_list = self.start_year_list
        end_year_list = self.end_year_list
        if shift:
            start_year_list = [str(int(year)+3) for year in start_year_list[-7:]]
            end_year_list = [str(int(year)+3) for year in end_year_list[-7:]]
        
        for i in range(len(start_year_list)):
            prinT("----------------------------------")
            prinT("start generating walks for {}-{} ({}/{})".format(start_year_list[i], end_year_list[i], i+1, len(start_year_list)))
            self.walks = []
            self.load_paper_df(full_load=False, start_year=start_year_list[i], end_year=end_year_list[i])
            self.load_ref_df(full_load=False, start_year=start_year_list[i], end_year=end_year_list[i])
            prinT('generating \'paper_set\', start counting unique PIDs...')
            self.target_paper_set = pd.unique(self.target_paper_df['PaperID'])
            prinT("finish. %d papers and %d reference records in this decade. %d papers cited their peers" %(len(self.target_paper_set), len(self.target_ref_df), len(self.target_ref_df.index.unique())))
            self.target_paper_df.set_index('PaperID', inplace=True)
            
            self.random_walk(num_walks)

            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/walks.pkl' %(start_year_list[i], end_year_list[i]), 'wb') as list_file:
                pickle.dump(self.walks, list_file)
            prinT("The file \'walks.pkl\' is ready, %d walks generated" %len(self.walks))
            prinT("{}/{} years generated...".format(i+1, len(start_year_list)))
            
        
    def get_overview_info(self):
        self.paper_num_list=[]
        self.ref_num_list=[]
        self.citing_paper_list=[]
        self.walk_num_list=[]
        self.venue_num_list=[]
        decades_list = []
        for i in range(len(self.start_year_list)):
            self.load_paper_df(full_load=False, start_year=self.start_year_list[i], end_year=self.end_year_list[i])
            self.load_ref_df(full_load=False, start_year=self.start_year_list[i], end_year=self.end_year_list[i]) 
            self.load_walks(start_year=self.start_year_list[i], end_year=self.end_year_list[i])

            self.paper_num_list.append(len(self.target_paper_df))
            self.ref_num_list.append(len(self.target_ref_df))
            self.citing_paper_list.append(len(self.target_ref_df.index.unique()))
            self.walk_num_list.append(len(self.walks))
            self.venue_num_list.append(len(pd.unique(self.target_paper_df['VenueID'])))
            decades_list.append(self.start_year_list[i]+', '+self.end_year_list[i])
            
            prinT("{}/{} recorded...".format(i+1, len(self.start_year_list)))
        return pd.DataFrame({'Num of paper': self.paper_num_list, 
                             'Num of citations': self.ref_num_list, 
                             'Num of papers citing peers': self.citing_paper_list, 
                             'Num of walks': self.walk_num_list, 
                             'Num of periodicals': self.venue_num_list}, 
                             index=decades_list)
        
        
    def plot_overview(self):
        x_axis_data = [start_year + 's'for start_year in self.start_year_list]
        y_axis_data1 = self.paper_num_list
        y_axis_data2 = self.ref_num_list
        y_axis_data3 = self.citing_paper_list
        y_axis_data4 = self.walk_num_list
        y_axis_data5 = self.venue_num_list

        fig = plt.figure(figsize = (12, 12), dpi=300)
        
        plt.plot(x_axis_data, y_axis_data1, 'gv--', alpha=0.4, linewidth=1, label='Num of papers')
        plt.plot(x_axis_data, y_axis_data2, 'yh--', alpha=0.4, linewidth=1, label='Num of citations')
        plt.plot(x_axis_data, y_axis_data5, 'co--', alpha=0.4, linewidth=1, label='Num of Periodicals')
        
        plt.plot(x_axis_data, y_axis_data3, 'bs-', alpha=0.65, linewidth=1, label='Num of papers citing peers')
        plt.plot(x_axis_data, y_axis_data4, 'r*-', alpha=0.65, linewidth=1, label='Num of trails')

        # for a, b1 in zip(x_axis_data, y_axis_data1):
        #     plt.text(a, b1, b1, ha='center', va='bottom', fontsize=6)
        # for a, b2 in zip(x_axis_data, y_axis_data2):
        #     plt.text(a, b2, b2, ha='center', va='bottom', fontsize=6)
        for a, b3 in zip(x_axis_data, y_axis_data3):
            plt.text(a, b3, b3, ha='center', va='bottom', fontsize=8)
        for a, b4 in zip(x_axis_data, y_axis_data4):
            plt.text(a, b4, b4, ha='center', va='bottom', fontsize=8)
        # for a, b5 in zip(x_axis_data, y_axis_data5):
        #     plt.text(a, b5, b5, ha='center', va='bottom', fontsize=6)
            
        plt.legend()
        plt.xlabel('decades')
        plt.xticks(rotation=45)
        plt.ylabel('counts')
        ax = plt.gca()
        plt.grid(b=True, which='major', axis='y', linestyle = '--')
        ax.set_yscale('log')
        
        fig.savefig('/media/sdb/p2v/figs/overview.pdf', dpi = 300, bbox_inches='tight')
        
        prinT("overview fig saved to file!")
        
        
    def walks_len_freq(self):
        for i in range(len(self.start_year_list)):
            self.load_walks(start_year=self.start_year_list[i], end_year=self.end_year_list[i])
            c = Counter([len(walk) for walk in self.walks])
            
            fig = plt.figure(figsize = (6, 4))
            fig.subplots_adjust(left = 0.15, bottom = 0.15)
            ax = plt.gca()
            x = [float(k) for k in c.keys()]
            y = [float(v) for v in c.values()]
            ax.scatter(x, y, c = 'blue', alpha = 0.5) # edgecolor = 'g'
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim([0.8, pow10ceil(max(x))])
            ax.set_ylim([1e-1, pow10ceil(max(y))])
            ax.set_xlabel('walk length')
            ax.set_ylabel('frequency')
            plt.title("walk length frequence (%s to %s)\nNumber of walks: %d"
                      %(self.start_year_list[i], self.end_year_list[i], len(self.walks)))
            plt.grid(True)
            
            dirs = '/media/sdb/p2v/figs/decades/%s_to_%s/' %(self.start_year_list[i], self.end_year_list[i])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            fig.savefig('/media/sdb/p2v/figs/decades/%s_to_%s/walk-length-dist.pdf' %(self.start_year_list[i], self.end_year_list[i]), 
                        dpi = 300)
            prinT("%d random walks' length freq fig for %s to %s saved to file!" %(len(self.walks), self.start_year_list[i], self.end_year_list[i]))
            prinT("{}/{} finish...".format(i+1, len(self.start_year_list)))
        
        
    def train_w2v_for_a_decade(self, start_year: int, end_year: int, 
                               num_features: int, context_win_size: int, min_word_count: int=50, negative: int=5,
                               use_filtered_walks=False):
        self.load_walks(start_year, end_year, use_filtered_walks)
        num_workers = cpu_count()
        downsampling = 1e-3
        prinT("Training model for papers from %d to %d..." %(int(start_year), int(end_year)))
        self.w2v_model = word2vec.Word2Vec(self.walks, 
                                      workers=num_workers, 
                                      vector_size=num_features,
                                      min_count=min_word_count, 
                                      window=context_win_size, 
                                      sample=downsampling, 
                                      sg=1, 
                                      negative=negative)
        vec_file_name = "/media/sdb/p2v/pickles/decades/%s_to_%s/%dfeat_%dcontext_win_size" %(start_year, end_year, num_features, context_win_size)
        self.w2v_model.wv.save(vec_file_name)
        prinT("done and saved model (%dfeat_%dcontext_win_size) to file!" %(num_features, context_win_size))
        
        # self.reduce_dimensions(start_year, end_year, num_features, context_win_size)
        
        
    def load_wv(self, start_year: int, end_year: int, d: int, w: int):
        prinT("start loading word vectors...")
        self.wv = KeyedVectors.load('/media/sdb/p2v/pickles/decades/%d_to_%d/%dfeat_%dcontext_win_size' 
                                    %(int(start_year), int(end_year), d, w))
        prinT('word vectors loaded, and its shape is: ' + str(self.wv.vectors.shape))
        return self.wv
    
    
    def reduce_dimensions(self, start_year: int, end_year: int, d: int, w:int):
        self.load_wv(start_year, end_year, d, w)
        prinT("start reducing dimension...")
        num_dimensions = 2

        # extract the words & their vectors, as numpy arrays
        vectors = self.wv.get_normed_vectors()
        VIDs = self.wv.index_to_key

        # reduce using t-SNE
        tsne = TSNE(n_components=num_dimensions, random_state=2023)
        vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        y_vals = [v[1] for v in vectors]
        prinT("finish.")
        
        wv_2d = {'x_val': x_vals,
                 'y_val': y_vals,
                 'VID': VIDs}
        with open('/media/sdb/p2v/pickles/decades/%s_to_%s/wv_2d_%dfeat_%dcontext_win_size.pkl' 
                  %(start_year, end_year, d, w), 'wb') as dict_file:
                pickle.dump(wv_2d, dict_file)
        prinT("The file \'wv_2d.pkl\' is ready.")
        return wv_2d
    
    
    def labelling(self, start_year: int, end_year: int, d: int, w: int):
        self.load_wv(start_year, end_year, d, w)
        full_vectors = self.wv.get_normed_vectors()
        full_VIDs = self.wv.index_to_key
        prinT("%d periodicals appear in this decade. Start labelling them..." %len(full_VIDs))
        labeled_venue_list = self.labeled_journal_info_df.index.to_list()

        labeled_VIDs = []
        labels = []
        founding_years = []
        for i in range(len(full_VIDs)):
            VID = int(full_VIDs[i])
            if VID in labeled_venue_list:
                label_list = self.labeled_journal_info_df.loc[VID]['ScopusCategory'].split(';;')
                if len(label_list) == 1:
                    label = label_list[0]
                    labels.append(label)
                    labeled_VIDs.append(VID)
                    founding_years.append(self.labeled_journal_info_df.loc[VID]['InceptionYear'])
                else:
                    neighbor_vid_list = [int(t[0]) for t in self.wv.similar_by_key(VID, topn=50)]
                    neighbor_label_list = []
                    label_count = []
                    for neighbor_vid in neighbor_vid_list:
                        if neighbor_vid in labeled_venue_list:
                            neighbor_label_list += self.labeled_journal_info_df.loc[neighbor_vid]['ScopusCategory'].split(';;')
                    if len(neighbor_label_list) != 0:
                        for label in label_list:
                            label_count.append(neighbor_label_list.count(label))
                        label = label_list[label_count.index(max(label_count))]
                        labels.append(label)
                        labeled_VIDs.append(VID)
                        founding_years.append(self.labeled_journal_info_df.loc[VID]['InceptionYear'])
        labels = ['Otorhinolaryngology' if label == 'Medicine: Otorhinolaryngology' else label.strip() for label in labels]
        subarea_labels = labels
        
        file_path = '/media/sdb/p2v/Scopus_discipline_mapping.xlsx'
        scopus_mapping_df = pd.read_excel(file_path, sheet_name = "Sheet1")
        mapping_dict = scopus_mapping_df[['ASJC category', 'Subject Area Classifications']].set_index('ASJC category').to_dict()['Subject Area Classifications']
        labels = [mapping_dict[label] for label in labels]

        prinT("%d of them are labeled by Scopus (got discipline categories)." %len(labeled_VIDs))
        VID_labeled = {'VID': labeled_VIDs,
                       'label': labels,
                       'subarea_label': subarea_labels,
                       'year_founded': founding_years}
        with open('/media/sdb/p2v/pickles/decades/%s_to_%s/VID_labeled_%dfeat_%dcontext_win_size.pkl' %(start_year, end_year, d, w), 'wb') as dict_file:
                pickle.dump(VID_labeled, dict_file)
        prinT("The file \'VID_labeled.pkl\' is ready.")
        
        return VID_labeled
    

    def load_MAG_venue_info_df(self):
        prinT("start loading Mag_venue_info_df")
        with open('/media/sdb/p2v/pickles/MAG_venue_info_df.pkl', 'rb') as df_file:
            self.MAG_venue_info_df = pd.compat.pickle_compat.load(df_file)
        prinT("finish.")

        
    def load_labeled_journal_info_df(self):
        prinT("start loading labeled_journal_info_df")
        with open('/media/sdb/p2v/pickles/labeled_journal_info_df.pkl', 'rb') as df_file:
            self.labeled_journal_info_df = pd.compat.pickle_compat.load(df_file)
        prinT("finish.")
        
        
    def load_wv_2d(self, start_year: int, end_year: int, d: int, w: int):
        try:
            prinT("start loading wv_2d...")
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/wv_2d_%dfeat_%dcontext_win_size.pkl' %(start_year, end_year, d, w), 'rb') as file:
                wv_2d = pickle.load(file)
            prinT("finish.")
            return wv_2d
        except:
            prinT("wv_2d file not exist, start it generating now by doing dimension reducing...")
            return self.reduce_dimensions(start_year, end_year, d, w)
        
    
    def load_VID_labeled(self, start_year: int, end_year: int, d: int, w: int):
        try:
            prinT("start loading VID_labeled...")
            with open('/media/sdb/p2v/pickles/decades/%s_to_%s/VID_labeled_%dfeat_%dcontext_win_size.pkl' %(start_year, end_year, d, w), 'rb') as dict_file:
                VID_labeled = pickle.load(dict_file)
            prinT("finish.")
            return VID_labeled
        except:
            prinT("VID_labeled file not exist, start labelling now...")
            return self.labelling(start_year, end_year, d, w)
        
        
    def plot_map_of_sci(self, start_year: int, end_year: int, d: int, w: int, 
                        rotate_180=False, y_flip=False, rotate_90=False, rotate_90_clockwise=False, 
                        annotate=False, center_VID=None, neighbor_VID_list=None, 
                        save_fig=False):
        VID_labeled = self.load_VID_labeled(start_year, end_year, d, w)
        wv_2d = self.load_wv_2d(start_year, end_year, d, w)
        
        plot_df = pd.DataFrame(wv_2d)
        plot_df = plot_df.loc[plot_df.VID.isin(VID_labeled['VID'])]
        plot_df['label'] = VID_labeled['label']
        plot_df['year_founded'] = VID_labeled['year_founded']
        plot_df['is_new'] = plot_df['year_founded'].apply(lambda x: 
                                                          0 if x==None
                                                          else 1 if int(x)>start_year
                                                          else 0)

        fig = plt.figure(figsize=(8, 9.6))
        gs = GridSpec(6, 1, figure=fig, hspace=0)
        map_ax = fig.add_subplot(gs[0:5, :])

        map_ax.set_title("Map of Science (%d to %d)\nNumber of journals: %d" %(start_year, end_year, len(plot_df)),)
        map_ax.set_aspect('equal')
        map_ax.axis('off')
        
        if annotate:
            point_alpha = 0.3
        else:
            point_alpha = 1.0

        if y_flip:
            plot_df['x_val'] = -1 * plot_df['x_val']
        original_x = plot_df['x_val']
        original_y = plot_df['y_val']
        if rotate_180:
            plot_df['x_val'] = -1 * original_x
            plot_df['y_val'] = -1 * original_y
        if rotate_90:
            plot_df['x_val'] = -1 * original_y
            plot_df['y_val'] = original_x
        if rotate_90_clockwise:
            plot_df['x_val'] = original_y
            plot_df['y_val'] = -1 * original_x
        
        scatter= sns.scatterplot(data=plot_df, x='x_val', y='y_val', hue='label', hue_order=self.discipline2color.keys(), 
                                 palette=self.discipline2color, alpha=point_alpha, s=5, ax=map_ax)
        handles, labels = scatter.get_legend_handles_labels()
        # 在下方子图的位置生成图例
        legend_ax = fig.add_subplot(gs[-1])
        legend_ax.axis('off')  # 隐藏坐标轴
        handles, labels = map_ax.get_legend_handles_labels()
        legend_ax.legend(handles,
                         [self.disc2abbr[label] for label in labels],
                         frameon=False,
                         fontsize=10, 
                         markerscale=3,
                         ncols=6, 
                         # mode='expand',
                         loc='upper center')
        
        # 移除第一个子图中的图例
        map_ax.get_legend().remove()

        if annotate:
            texts = []
            text = plt.text(plot_df[plot_df.VID==center_VID].x_val.values[0], 
                                    plot_df[plot_df.VID==center_VID].y_val.values[0], 
                                    self.MAG_venue_info_df.at[center_VID, 'OriginalVenue'],
                                    fontsize=10,
                                    color='white',
                                    fontweight='bold',
                                    ha='center')
            text.set_path_effects([path_effects.Stroke(linewidth=1.25, foreground='black'),
                                   path_effects.Normal()])
            texts.append(text)
            for VID in neighbor_VID_list:
                text = plt.text(plot_df[plot_df.VID==VID].x_val.values[0], 
                                plot_df[plot_df.VID==VID].y_val.values[0], 
                                self.MAG_venue_info_df.at[VID, 'OriginalVenue'],
                                fontsize=8,
                                color='white',
                                fontweight='bold',
                                ha='center')
                text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
                texts.append(text)
            
            neighbor_VID_list.append(center_VID)
            neighbor_df = plot_df.loc[plot_df.VID.isin(neighbor_VID_list)]
            sns.scatterplot(data=neighbor_df, 
                            x='x_val', 
                            y='y_val', 
                            hue='label',
                            hue_order = self.discipline2color.keys(),
                            palette=self.discipline2color,
                            s=5,
                            ax=map_ax,
                            legend=False)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', lw=0.4, color='red'))
    
        if save_fig:
            fig.savefig('map_of_sci_%s_to_%s.png' %(start_year, end_year), 
                        facecolor='white', 
                        transparent=False, 
                        bbox_inches='tight')
            prinT("%d journals' map of sci fig for %s to %s saved to file!" %(len(plot_df), start_year, end_year))
        # return fig
    
    
    def find_k(self, start_year: int, end_year: int, d: int, w: int):
        self.load_wv(start_year, end_year, d, w)
        VIDs = self.wv.index_to_key
        vectors = self.wv.get_normed_vectors()

        knn_df = pd.DataFrame(list(zip(VIDs, vectors)), columns =['VID', 'vector'])
        VID_labeled = self.load_VID_labeled(start_year, end_year, d, w)
        knn_df = knn_df.loc[knn_df.VID.isin(VID_labeled['VID'])]
        knn_df['label'] = VID_labeled['label']

        x = knn_df.vector.to_list()
        y = knn_df.label

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2023)
        score_list = []
        for k in range(1, 50):
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn.fit(x_train, y_train)
            score_list.append(knn.score(x_test, y_test))
        plt.plot(range(1, 50), score_list)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Classification accuracy')
        plt.show()

        best_k = score_list.index(max(score_list))
        prinT("the best k is: "+str(best_k))
        return best_k
    
    
    def cal_entropy_single(self, x):
        neighbor_index = self.knn.kneighbors(X=x.reshape(1, -1), return_distance=False)
        neighbor_df = self.knn_df.iloc[neighbor_index.reshape(-1,)]
        pe_value_array = neighbor_df['label'].unique()

        ent = 0.0
        for x_value in pe_value_array:
            p = float(neighbor_df[neighbor_df['label'] == x_value].shape[0]) / neighbor_df.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent
    
    
    def cal_entropy(self, start_year: int, end_year: int, d: int, w: int, k: int,):
        self.load_wv(start_year, end_year, d, w)
        VIDs = self.wv.index_to_key
        vectors = self.wv.get_normed_vectors()

        self.knn_df = pd.DataFrame(list(zip(VIDs, vectors)), columns =['VID', 'vector'])
        VID_labeled = self.load_VID_labeled(start_year, end_year, d, w)
        self.knn_df = self.knn_df.loc[self.knn_df.VID.isin(VID_labeled['VID'])]
        self.knn_df['label'] = VID_labeled['label']

        self.knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        self.knn.fit(self.knn_df.vector.to_list(), self.knn_df.label)
        
        self.knn_df['local_entropy'] = self.knn_df['vector'].apply(self.cal_entropy_single)
        
        self.knn_df['VID'] = self.knn_df['VID'].astype(int)
        self.knn_df.loc[:,'venue_name'] = self.knn_df['VID'].apply(lambda x: self.MAG_venue_info_df.at[x, 'OriginalVenue'])
        self.knn_df['year_founded'] = VID_labeled['year_founded']
        self.knn_df['is_new'] = self.knn_df['year_founded'].apply(lambda x: 
                                                                  0 if x==None
                                                                  else 1 if int(x)>=start_year
                                                                  else 0)
    
    def plot_local_entropy_map(self, start_year: int, end_year: int, d: int, w: int, k: int,
                                rotate_180=False, y_flip=False, rotate_90=False, rotate_90_clockwise=False):
        wv_2d = self.load_wv_2d(start_year, end_year, d, w) 
        plot_df = pd.DataFrame(wv_2d)
        plot_df = pd.merge(plot_df, self.knn_df, left_on='VID', right_on='VID')

        fig = plt.figure(figsize=(15, 15), dpi=300)
        ax = fig.gca()
        ax.set_aspect('equal')

        if y_flip:
            plot_df['x_val'] = -1 * plot_df['x_val']
        original_x = plot_df['x_val']
        original_y = plot_df['y_val']
        if rotate_180:
            plot_df['x_val'] = -1 * original_x
            plot_df['y_val'] = -1 * original_y
        if rotate_90:
            plot_df['x_val'] = -1 * original_y
            plot_df['y_val'] = original_x
        if rotate_90_clockwise:
            plot_df['x_val'] = original_y
            plot_df['y_val'] = -1 * original_x

        points = sns.scatterplot(data=plot_df, 
                                 x='x_val', 
                                 y='y_val', 
                                 hue='local_entropy',
                                 style='is_new',
                                 size='is_new',
                                 sizes=(5,50),
                                 size_order=[1, 0],
                                 palette=sns.color_palette("light:k", as_cmap=True),
                                 ax=ax)
        plt.title("The distribution of local entropy for journals (%d to %d)\nNumber of journals: %d\nk: %d" 
                  %(start_year, end_year, len(plot_df), k), 
                  loc='left')
        plt.axis('off')
        
