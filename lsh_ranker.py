import os, pickle
import numpy as np
# from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from params import get_params
import random
import time

class Ranker():

    def __init__(self,params):

        # Read image lists

        self.dataset= params['dataset']
        self.image_path = params['database_images']
        self.dimension = params['dimension']
        self.pooling = params['pooling']
        self.N_QE = params['N_QE']
        self.N_display = params['N_display']
        self.stage = params['stage']
        with open(params['frame_list'],'r') as f:
            self.database_list = f.read().splitlines()

        with open(params['query_list'],'r') as f:
            self.query_names = f.read().splitlines()

        # Distance type
        self.dist_type = params['distance']

        # Database features ---

        # PCA MODEL - use paris for oxford data and vice versa
        if self.dataset is 'paris':

            self.pca = pickle.load(open(params['pca_model'] + '_oxford.pkl', 'rb'))

        elif self.dataset is 'oxford':

            self.pca = pickle.load(open(params['pca_model'] + '_paris.pkl', 'rb'))

        #lsh
        lshash=RandomBinaryProjections('rankHash', self.N_display)
        self.engine=Engine(self.dimension, lshashes=[lshash])

        # Load features
        self.db_feats = pickle.load(open(params['database_feats'],'rb'))

        print "Applying PCA"

        self.db_feats = normalize(self.db_feats)
        if self.pooling is 'sum':
            self.db_feats = self.pca.transform(self.db_feats)
            self.db_feats = normalize(self.db_feats)

        ct=0
        for db_feat in self.db_feats:
            self.engine.store_vector(db_feat, self.database_list[ct])
            ct+=1

        print ct, " features loaded."

        # Where to store the rankings
        self.rankings_dir = params['rankings_dir']

    def get_query_vectors(self):


        self.query_feats = np.zeros((len(self.query_names),self.dimension))

        i = 0
        for query in self.query_names:

            query_file, box = self.query_info(query)
            self.query_feats[i,:] = self.db_feats[np.where(np.array(self.database_list) == query_file)]

            # add top elements of the ranking to the query
            if self.stage is 'QE':

                with open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'r') as f:
                    ranking = f.read().splitlines()

                for i_q in range(self.N_QE):

                    imfile = ranking[i_q]

                    # construct image path
                    if self.dataset is 'paris':
                        imname = os.path.join(self.image_path,imfile.split('_')[1],imfile + '.jpg')
                    elif self.dataset is 'oxford':
                        imname = os.path.join(self.image_path,imfile + '.jpg')
                    # find feature and add to query
                    feat = self.db_feats[np.where(np.array(self.database_list) == imname)].squeeze()

                    self.query_feats[i,:] += feat
                # find feature and add to query


            i+=1

        self.query_feats = normalize(self.query_feats)

    def query_info(self,filename):

        '''
        For oxford and paris, get query frame and box
        '''

        data = np.loadtxt(filename, dtype="str")

        if self.dataset is 'paris':

            query = data[0]

        elif self.dataset is 'oxford':

            query = data[0].split('oxc1_')[1]

        bbx = data[1:].astype(float).astype(int)

        if self.dataset is 'paris':
            query = os.path.join(self.image_path,query.split('_')[1],query + '.jpg')
        elif self.dataset is 'oxford':
            query = os.path.join(self.image_path,query + '.jpg')

        return query, bbx

    def write_rankings(self):
        i=0
        for query_feat in self.query_feats:
            neighbours = self.engine.neighbours(query_feat)
            savefile = open(os.path.join(self.rankings_dir,os.path.basename(self.query_names[i].split('_query')[0]) +'.txt'),'w')

            for neighbour in neighbours:
                savefile.write(os.path.basename(neighbour[1]).split('.jpg')[0] + '\n')
            savefile.close()

            i+=1

    def rank(self):

        self.get_query_vectors()

        print "Writing rankings to disk..."
        t0 = time.time()
        self.write_rankings()
        print "Done. Time elapsed", time.time() - t0


if __name__ == "__main__":
    params = get_params()

    R = Ranker(params)
    R.rank()

    R.engine.clean_all_buckets()
