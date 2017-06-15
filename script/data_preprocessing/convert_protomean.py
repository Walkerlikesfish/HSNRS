import sys
import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read("./setting_data.ini")
caffe_root = Config.get('Caffe', 'CaffeRoot')
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np


if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py proto.mean out.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
print arr.shape
out = arr[0]
np.save( sys.argv[2] , out )