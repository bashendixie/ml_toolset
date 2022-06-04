from keras.models import load_model
import os 
import os.path as osp
from keras import backend as K
#路径参数
tf1.disable_v2_behavior()

weight_file = 'fcn32_af1.hdf5'
weight_file_path = weight_file
output_graph_name = 'test123.pb'

#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf1.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = tf1.compat.v1.keras.backend.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = True)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)
#输出路径
output_dir = osp.join(os.getcwd(),"trans_model")
#加载模型
h5_model = load_model(weight_file_path)
h5_to_pb(h5_model,output_dir = 'output_dir',model_name = output_graph_name)
print('model saved')