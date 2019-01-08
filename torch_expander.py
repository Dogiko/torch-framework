import torch
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)

def to_32(x):
    return x.type(torch.FloatTensor)

def to_64(x):
    return x.type(torch.DoubleTensor)

def pos_gram(gram, regularlizer = None):
    _type = gram.type()
    _size = len(gram)
    if regularlizer is None:
        if gram.abs().max() == 0:
            raise ValueError("gram error, expect matrix with none-zero element")
        
        # the fraction of float32 is 2**(-23)~10**(-7) we start with 10**(-7) times of maximun element
        regularlizer = gram.abs().max()*0.0000001
    
    if regularlizer <= 0:
        raise ValueError("regularlizer error, expect positive, got %s" %(regularlizer))
    
    while True:
        lambdas, vectors = torch.symeig(gram + regularlizer*torch.eye(_size).type(_type))
        if lambdas.min() > 0:
            break
        
        regularlizer *= 2.
    
    return gram + regularlizer*torch.eye(_size).type(_type)

class LinearExpander():
    def __init__(self, linear_model, activation_function, candidate_num=1, std = None):
        self.linear_model = linear_model
        self.activation_function = activation_function
        self.candidate = torch.nn.Linear(self.linear_model.in_features, candidate_num)
        if std is not None:
            self.candidate.weight.data *= torch.tensor(std*(3*self.candidate.in_features)**0.5)
        
        self.reset()
    
    def reset(self):
        # regressor_gram : store X^t*X
        # projector : store X^t*Y
        # responsor_ss : store component-wise square sum of y (=diag(Y^t*Y))
        self.regressor_gram = torch.zeros((self.linear_model.out_features+1, self.linear_model.out_features+1)).data
        self.projector = torch.zeros((self.linear_model.out_features+1, self.candidate.out_features)).data
        self.responsor_ss = torch.zeros((self.candidate.out_features)).data
        self.datums_acc = 0
    
    def data_input(self, data):
        datums = data.size()[0]
        regressor = self.linear_model(data).data
        regressor = self.activation_function(regressor)
        expand = torch.cat((regressor, torch.ones((datums, 1))), 1)
        self.regressor_gram += torch.mm(expand.t(), expand)
        responsor = self.candidate(data).data
        responsor = self.activation_function(responsor)
        self.projector += torch.mm(expand.t(), responsor)
        self.responsor_ss += (responsor**2).sum(0)
        self.datums_acc += datums
    
    def take(self, take_num=1, weighted=True):
        # return index of hitted candidate
        lots_num = self.candidate.out_features
        if take_num > lots_num:
            raise ValueError("take_num exceed candidate")
        
        if weighted:
            if self.datums_acc == 0:
                raise ZeroDivisionError("input data before take(with weighted)")
            mean_gram = pos_gram(to_64(self.regressor_gram / self.datums_acc)) # avoid singular gram
            lambdas, vectors = torch.symeig(mean_gram, eigenvectors=True) # eigen
            mean_projector = to_64(self.projector/self.datums_acc)
            lambdas_inv = 1/(lambdas+0.0000001) 
            VtXtY = vectors.t().mm(mean_projector)
            dependency = ((VtXtY.t()*lambdas_inv).t()*VtXtY).sum(0) # diag of Y^t*X*Gram^(-1)*X^t*Y
            independency = to_64(self.responsor_ss/self.datums_acc) - dependency
            prob = independency/independency.sum()
            output = np.random.choice(self.candidate.out_features, take_num, replace=False, p=prob)
        else:
            output = torch.randperm(lots_num)[:take_num]
        
        return output
    
    def expand(self, expand_size=1, weighted=True):
        # return new linear model by self.take
        ori_in = self.linear_model.in_features
        ori_out = self.linear_model.out_features
        output = torch.nn.Linear(ori_in, ori_out + expand_size)
        take_idx = self.take(expand_size, weighted)
        output.weight.data[:ori_out] = self.linear_model.weight.data
        output.weight.data[ori_out:] = self.candidate.weight.data[take_idx]
        output.bias.data[:ori_out] = self.linear_model.bias.data
        output.bias.data[ori_out:] = self.candidate.bias.data[take_idx]
        return output