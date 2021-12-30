import onnx
from onnx import numpy_helper
import numpy as np

# model = onnx.load('mobilenet_v2.onnx')
model = onnx.load('mobilenet_v2_inplace.onnx')

def read_parameters(model=model):
    params = {}
    for t in model.graph.initializer:
        params[t.name] = numpy_helper.to_array(t)
    # import pdb;pdb.set_trace()
    return params
    
def params2txt(params):
    name_file = open('name.txt','w')
    params_file = open('params.txt','w')
    for name in params:
        if 'class' in name:
            continue
        data = params[name]
        name_file.write(str(name)+" ")
        for i in data.shape:
            name_file.write(str(i)+' ')
        name_file.write('\n')
        for d in data.flatten():
            params_file.write(str(d)+' ')
    
    for name in ['classifier.1.weight','classifier.1.bias']:
        data = params[name]
        name_file.write(str(name)+" ")
        for i in data.shape:
            name_file.write(str(i)+' ')
        name_file.write('\n')
        for d in data.flatten():
            params_file.write(str(d)+' ')
    name_file.close()
    params_file.close()

def nodes2txt(model=model):
    node_file = open('nodes_calc.txt','w')
    for n in model.graph.node:
        if 'Conv' not in n.name and 'Gemm' not in n.name:
            continue
        node_file.write(n.name+'\n')
        node_file.write(str(n.input[1:])[1:-1]+' ')
        # node_file.write(str(n.output)[1:-1]+'\n')
        for a in n.attribute:
            data = {1:str(a.f), 2:str(a.i), 7:a.ints} #str(a.ints)[1:-1]}
            try:
                value = data[a.type]
            except KeyError:
                print(n.name,a)
            node_file.write((value if a.type != 7 else str(value[0]))+' ')
        node_file.write('\n')
    node_file.close()
if __name__ == '__main__':
    params2txt(read_parameters())
    nodes2txt()