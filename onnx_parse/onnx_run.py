# import onnxruntime
# import numpy as np
# import torch
# ort_session = onnxruntime.InferenceSession("mobilenet_v2_inplace.onnx")

# with open('mobilenetInput.txt') as f:
#     x = f.readline().strip()
#     x = np.array(list(map(float,x.split(' ')))).reshape((1,3,244,244))
#     x = torch.tensor(x,dtype=float)
# with open('mobilenetOutput.txt') as f:
#     torch_out = f.readline().strip()
#     torch_out = np.array(list(map(float,torch_out.split(' ')))).reshape((1,1000))
#     torch_out = torch.tensor(torch_out,dtype=float)

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction

# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x).astype('float32')}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")

with open('nodes.txt') as f:
    nodename = [i for i in f.readlines() if 'Conv' in i]
with open('name.txt') as f:
    n = nodename.pop(0).strip()
    while data:=f.readline():
        data = data.strip().split(' ')
        name = data[0]
        params = data[1:]
        if len(params)>1:
            # print(f'float {n}_W[{params[0]}*{params[1]}*{params[2]}*{params[3]}];')
            print(f'for(int i=0;i<{params[0]}*{params[1]}*{params[2]}*{params[3]};i++)fscanf(fp,"%f",&{n}_W[i]);')
            print(f'cudaMalloc(temp, sizeof(float)*{params[0]}*{params[1]}*{params[2]}*{params[3]});')
            print(f'cudaMemcpy(temp, {n}_W, sizeof(float)*{params[0]}*{params[1]}*{params[2]}*{params[3]}, cudaMemcpyHostToDevice);')
            print(f'{n}_W = temp;')
        else:
            # print(f'float {n}_B[{params[0]}];')
            print(f'for(int i=0;i<{params[0]};i++)fscanf(fp,"%f",&{n}_B[i]);')
            print(f'cudaMalloc(temp, sizeof(float)*{params[0]});')
            print(f'cudaMemcpy(temp, {n}_B, sizeof(float)*{params[0]}, cudaMemcpyHostToDevice);')
            print(f'{n}_B = temp;')
            n = nodename.pop(0).strip()