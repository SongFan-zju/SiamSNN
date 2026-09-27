import os
from collections import OrderedDict
import shutil
from numpy import arange
from numpy.lib.arraysetops import isin

import torch
import torch.nn as nn
from spiking_utils.spike_layer import *
from spiking_utils.spike_tensor import SpikeTensor

def is_layer_weighted_spike(layer):
    return isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeConvTranspose2d) or isinstance(layer, SpikeLinear)

class DAGViewOp:
    """
    Wrapper of the x.view operation
    """

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, x):
        out = x.view(*self.sizes)
        return out


class DAGConcatOp:
    """
    Wrapper of torch.cat() operation
    """

    def __init__(self, dim):
        assert dim != 0
        self.dim = dim

    def __call__(self, *xs):
        if isinstance(xs[0], SpikeTensor):
            out = torch.cat([_.data for _ in xs], dim=self.dim)
            scale_factor = torch.cat([_.scale_factor for _ in xs], dim=self.dim - 1)
            out = SpikeTensor(out, xs[0].timesteps, scale_factor)
        else:
            out = torch.cat(xs, dim=self.dim)
        return out


class SpikeDAGModule(nn.Module):
    """
    This is a module contains `nodes` dict and `ops` dict.
    It can do inference under the topological ordering of the ops given the input nodes.
    - Attributes
        - `nodes` OrderedDict, in which the key is the name of the input (`<op_name>_out_<id>`)
        - `ops` OrderedDict, in which the key is the name of the op (`<op_type>_<id>`)
            and the value is a dict `{'op':op, 'in_nodes':tuple, 'out_nodes':tuple}`
        - `inputs_nodes` tuple, specifying the input nodes for forward
    - forward
        - copy the inputs to nodes in `inputs_nodes`
        - for `op` in `ops`:
            - get inputs from `nodes`
            - forward `op`
            - save outputs to `nodes`
    """

    def __init__(self):
        super().__init__()
        self.nodes = OrderedDict()
        self.ops = OrderedDict()
        self.inputs_nodes = []
        self.outputs_nodes = []

    def add_op(self, name, op, in_nodes, out_nodes):
        assert name not in self.ops, AssertionError(f"Error: op {name} already in dag.ops")
        assert len(in_nodes) and len(out_nodes)
        self.ops[name] = {'op': op, 'in_nodes': in_nodes, 'out_nodes': out_nodes}
        if isinstance(op, nn.Module):
            self.add_module(name, op)
        # print(f"add node {name}: {in_nodes}->{out_nodes}")

    def add_node(self, name, tensor=None):
        assert name not in self.nodes, AssertionError(f"Error: node {name} already in dag.nodes")
        self.nodes[name] = tensor

    def clear_nodes(self):
        for key in self.nodes:
            self.nodes[key] = None

    def find_end_nodes(self):
        in_nodes = []
        for _, op in self.ops.items():
            in_nodes.extend(op['in_nodes'])
        end_nodes = []
        for node in self.nodes:
            if node not in in_nodes:
                end_nodes.append(node)
        # print(end_nodes)
        return tuple(end_nodes)

    def forward(self, *inputs):
        assert len(self.inputs_nodes) == len(inputs)
        assert len(self.outputs_nodes)
        for inp, name in zip(inputs, self.inputs_nodes):
            self.nodes[name] = inp
        for op_name, op in self.ops.items():
            op_inputs = tuple(self.nodes[_] for _ in op['in_nodes'])
            op_outputs = op['op'](*op_inputs)
            if isinstance(op_outputs, torch.Tensor) or isinstance(op_outputs, SpikeTensor):
                op_outputs = [op_outputs]
            for node_name, op_output in zip(op['out_nodes'], op_outputs):
                self.nodes[node_name] = op_output
        outputs = tuple(self.nodes[_] for _ in self.outputs_nodes)
        return outputs[-1]
        # return outputs if len(outputs) > 1 else outputs[0]

    def save_conv_weights(self,save_dir,format="cnn"):
        SAVE_INPUT = True
        SAVE_OUTPUT = True
        start_neuron_id = [0]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir,exist_ok=True)
        if SAVE_INPUT:
            fp_input = open(os.path.join(save_dir,"snn.inputs"),"a",encoding="utf-8")
            save_spike(self.inputs_nodes,self.nodes,fp_input,start_neuron_id)
            fp_input.close()
        fp_wgt = open(os.path.join(save_dir,"snn.weights"),"a",encoding="utf-8")
        if SAVE_OUTPUT:
            fp_output = open(os.path.join(save_dir,"snn.outputs"),"a",encoding="utf-8")
        for op_name, op in self.ops.items():
            layer = op['op']
            outputs = self.nodes[op['out_nodes'][0]]
            if SAVE_OUTPUT and isinstance(layer,SpikeConv2d):
                save_spike(op['out_nodes'],self.nodes,fp_output,start_neuron_id)
            if isinstance(layer,SpikeConv2d):
                weight = layer.weight.data.cpu().numpy().reshape(1,-1)
                np.savetxt(fp_wgt,weight,fmt="%d",delimiter=',')
                if layer.bias is not None:
                    bias = layer.bias.data.cpu().numpy().reshape(1,-1)
                    np.savetxt(fp_wgt,bias,fmt="%d",delimiter=',')
                vTh = layer.Vthr.data.cpu().numpy().reshape(1,-1)
                np.savetxt(fp_wgt,vTh,fmt="%d",delimiter=',') 
            elif isinstance(layer,SpikeReLU):
                self.nodes[op['out_nodes'][0]] = self.nodes[op['in_nodes'][0]]

        fp_wgt.close()
        if SAVE_OUTPUT:
            fp_output.close()

    def save_weights(self,save_dir,format="darwin"):
        SAVE_INPUT = True
        SAVE_OUTPUT = True
        start_neuron_id = [0]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir,exist_ok=True)
        if SAVE_INPUT:
            fp_input = open(os.path.join(save_dir,"snn.inputs"),"a",encoding="utf-8")
            save_spike(self.inputs_nodes,self.nodes,fp_input,start_neuron_id)
            fp_input.close()
            # input_spike_trains = [] #spike format(neuron_id,spkie_t1,...,spike_ts)
            # for name in self.inputs_nodes:
            #     spikes = self.nodes[name]
            #     timesteps = spikes.timesteps
            #     chw = spikes.size()[1:]
            #     spikes = spikes.view(timesteps, -1, *chw).data.cpu().numpy()
            #     spikes = spikes[:,0,...].reshape(timesteps,-1).T
            #     for ind,spike in enumerate(spikes):
            #         input_spike_trains.append((neuron_id+ind,*spike))
            #     neuron_id += len(spikes)
            #     self.nodes[name] = np.arange(np.product(chw)).reshape(*chw)
            # np.savetxt(os.path.join(save_dir,"snn.inputs"),input_spike_trains,fmt="%d",delimiter=',')

        fp_wgt = open(os.path.join(save_dir,"snn.weights"),"a",encoding="utf-8")
        if SAVE_OUTPUT:
            fp_output = open(os.path.join(save_dir,"snn.outputs"),"a",encoding="utf-8")
        for op_name, op in self.ops.items():
            layer = op['op']
            outputs = self.nodes[op['out_nodes'][0]]
            if SAVE_OUTPUT and isinstance(layer,SpikeConv2d):
                save_spike(op['out_nodes'],self.nodes,fp_output,start_neuron_id)
            out_channel,out_height,out_width = outputs.size()[1:]
            inputs = tuple(self.nodes[_] for _ in op['in_nodes'])
            neurons = None #weights format(neuron_id1,neuron_id2,weight,bias)
            if isinstance(layer,SpikeConv2d):
                inputs = inputs[0]
                neuron_id = inputs.ravel()[-1] + 1
                ksize = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                chw= inputs.shape
                weight = layer.weight.data.cpu().numpy().reshape(out_channel,-1)
                bias = None
                if hasattr(layer,"bias"):
                    bias = layer.bias.data.cpu().numpy()
                inputs = img2col(inputs,out_height,out_width,ksize,stride,padding)
                outputs = np.arange(np.product([out_channel,out_height,out_width])) + neuron_id
                outputs = outputs.reshape(out_channel,out_height,out_width)
                self.nodes[op['out_nodes'][0]] = outputs
                outputs = outputs.reshape(out_channel,-1)
                neurons = get_neuron_weight(inputs,outputs,weight,bias,layer.Vthr.data.cpu().numpy())
            elif isinstance(layer,SpikeReLU):
                self.nodes[op['out_nodes'][0]] = self.nodes[op['in_nodes'][0]]

            if neurons is not None:
                np.savetxt(fp_wgt,neurons,fmt="%d,%d,%d,%d,%d")
        fp_wgt.close()
        if SAVE_OUTPUT:
            fp_output.close()

def save_spike(names,nodes,path,start_neuron_id):
    spike_trains = [] #spike format(neuron_id,spkie_t1,...,spike_ts)
    for name in names:
        spikes = nodes[name]
        timesteps = spikes.timesteps
        chw = spikes.size()[1:]
        scales = spikes.scale_factor.data.cpu().numpy()
        scales = scales.reshape(-1)
        spikes = spikes.view(timesteps, -1, *chw).data.cpu().numpy()
        spikes = spikes[:,0,...].reshape(timesteps,-1).T
        # spikes = spikes.reshape(timesteps,-1).T
        for ind,spike in enumerate(spikes):
            spike_trains.append((start_neuron_id[0]+ind,round(scales[ind//(chw[1]*chw[2])],6),*spike))
        start_neuron_id[0] += len(spikes)
        nodes[name] = np.arange(np.product(chw)).reshape(*chw)
    fmt = '%d,'+'%.6f,'+','.join(['%d']*timesteps)
    np.savetxt(path,spike_trains,fmt=fmt,delimiter=',')

def img2col(tensor,out_height,out_width,ksize,stride,padding):
    c,h,w = tensor.shape
    out = np.zeros((out_height*out_width,np.product(ksize)*c))
    count = 0
    step_y,step_x = ksize
    for idy,y in enumerate(range(0,h+padding[0],stride[0])):
        if y+step_y > h:
            continue
        for idx,x in enumerate(range(0,w+padding[1],stride[1])):
            if x+step_x > w:
                continue
            count += 1
            out[idy*out_width+idx,:] = tensor[:,y:y+step_y,x:x+step_x].flatten()

    return out

def get_neuron_weight(inputs,outputs,weight,biases,Vthr):
    neurons = []
    for ch_idx in range(outputs.shape[0]):
        if biases is None:
            bias = 0
        else:
            bias = biases[ch_idx]
        vthr = Vthr[ch_idx]
        for out_idx in range(outputs.shape[1]):
            dst_neuron_id = outputs[ch_idx,out_idx]
            for kernel_idx in range(inputs.shape[1]):
                src_neuron_id = inputs[out_idx,kernel_idx]
                neurons.append([src_neuron_id,dst_neuron_id,weight[ch_idx,kernel_idx],bias,vthr])
    return neurons