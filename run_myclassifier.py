import subprocess as sp

data_folder = '/Users/xqu/datasets/lfw/lfw_mtcnnpy_160'
face_net_model = '/Users/xqu/datasets/pretrain/20180402-114759.pb'
output_model = '/Users/xqu/datasets/mymodels/lfw_classifier.pkl'
batch_size = 10
augment_times = 10

cmd = """python myclassifier.py TRAIN \
{} \
{} \
{} \
--batch_size {} \
--augment_times {}""".format(data_folder,face_net_model,output_model,batch_size,augment_times)

print("Now runing myclassifer........")
flag=sp.call(cmd,shell=True)
if flag!=0:
    raise Exception('Please check python myclassifier.py cmd ')
else:
    print('\nFinished')
