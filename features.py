import sys, os
import cv2
import time
import numpy as np
from params import get_params
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

params = get_params()

# Add Faster R-CNN module to pythonpath
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import tensorflow as tf
from fast_rcnn.config import cfg
from networks.factory import get_network
import test as test_ops


def learn_transform(params,feats):

    feats = normalize(feats)
    pca = PCA(params['dimension'],whiten=True)

    pca.fit(feats)

    pickle.dump(pca,open(params['pca_model'] + '_' + params['dataset'] + '.pkl','wb'))


class Extractor():

    def __init__(self,params):

        self.dimension = params['dimension']
        self.dataset = params['dataset']
        self.pooling = params['pooling']
        self.network_name = params['network_name']
        # Read image lists
        with open(params['query_list'],'r') as f:
            self.query_names = f.read().splitlines()

        with open(params['frame_list'],'r') as f:
            self.database_list = f.read().splitlines()

        # Parameters needed
        self.layer = params['layer']
        self.save_db_feats = params['database_feats']

        # Init network
        weights_filename = os.path.splitext(os.path.basename(params['model']))

        # print "Extracting from:", params['net_proto']
        print "Using model:", self.network_name
        cfg.TEST.HAS_RPN = True
        # self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
        self.net = get_network(self.network_name)

    def extract_feat_image(self, sess, image):

        im = cv2.imread(image)

        scores, boxes, feat = test_ops.im_detect(sess, self.net, self.layer, im, True, boxes = None)
        # feat = self.net.get_output(self.layer)

        return feat


    def pool_feats(self, feat):

        if self.pooling is 'max':

            feat = np.max(np.max(feat,axis=2),axis=1)
        else:

            feat = np.sum(np.sum(feat,axis=2),axis=1)

        return feat

    def save_feats_to_disk(self, sess):

        print "Extracting database features..."
        t0 = time.time()
        counter = 0

        # Init empty np array to store all database features
        xfeats = np.zeros((len(self.database_list),self.dimension))

        for frame in self.database_list:
            counter +=1

            # Extract raw feature from cnn
            feat = self.extract_feat_image(sess, frame).squeeze()

            # Compose single feature vector
            feat = self.pool_feats(feat)

            # Add to the array of features
            xfeats[counter-1,:] = feat

            # Display every now and then
            if counter%50 == 0:
                print counter, '/', len(self.database_list), time.time() - t0
            # print frame


        # Dump to disk
        pickle.dump(xfeats,open(self.save_db_feats,'wb'))

        print " ============================ "


if __name__ == "__main__":

    params = get_params()

    if params['gpu'] == True:
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = params['device_id']
    else:
        cfg.USE_GPU_NMS = False

    # start a session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # load network (Extractor constructor)
    E = Extractor(params)

    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, params['model'])

    E.save_feats_to_disk(sess)

    print ('Loading model weights from {:s}').format(params['model'])

    feats = pickle.load(open(params['database_feats'],'rb'))
    learn_transform(params,feats)

    sess.close()
