import numpy as np
import torch
from torch.autograd import Variable

torch.set_default_tensor_type(torch.FloatTensor)

def to_32(x):
    return x.type(torch.FloatTensor)

def to_64(x):
    return x.type(torch.DoubleTensor)

def two_square_trace(A, B):
    # find tr(AB)
    # see https://github.com/Dogiko/Some-Tools
    return A.view(1,-1).mm(B.t().contiguous().view(-1,1)).view(())

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

class LinearRefiner():
    def __init__(self, linearModel, regularizer = None):
        self.set_model(linearModel)
        self.set_regularizer(regularizer)
    
    def set_model(self, linearModel):
        if type(linearModel) != torch.nn.Linear:
            raise TypeError("linearModel type error, got, expect torch.nn.Linear")
        
        self.model_weight = linearModel.weight.data
        self.model_bias = linearModel.bias.data
        self.in_features = linearModel.in_features
        self.reset()
    
    def set_regularizer(self, regularizer):
        self.regularizer = regularizer
    
    def reset(self):
        self.datums_acc = 0
        self.gram = torch.zeros((self.in_features+1, self.in_features+1))
    
    def data_input(self, data):
        datums = data.size()[0]
        expand = torch.cat((data, torch.ones((datums, 1))), 1)
        self.gram += torch.mm(expand.t(), expand)
        self.datums_acc += datums
    
    def drop_error(self, estimate = False):
        # return errors of drop each regressor
        # if estimate = True, using PCA get upper-bound of errors
        if self.datums_acc == 0:
            raise ValueError(".datums_acc == 0, must input data before refine.")
        
        expand_model = to_64(torch.cat((self.model_weight, self.model_bias.view(-1,1)), 1).t())
        mean_gram = pos_gram(to_64(self.gram / self.datums_acc), self.regularizer) # avoid singular gram
        res_ms = two_square_trace(expand_model.mm(expand_model.t()), mean_gram) # mean of square sum of y = xA
        if estimate:
            unbiased_gram = pos_gram(mean_gram[:-1, :-1] - torch.mm(mean_gram[:-1, -1:], mean_gram[-1:, :-1]))
            cori, comp = torch.symeig(unbiased_gram, eigenvectors=True) # PCA
            error_bound = 1/(0.0000001 + comp**2) # 0.0000001 for stability, avoid divide zero
            error_bound *= cori
            error_bound = error_bound.min(1)[0]*((expand_model[:-1]**2).sum(1)) # square-norm for weight without bias
            
            error_bound = to_32(error_bound)
            res_ms = to_32(res_ms)
            return error_bound, res_ms
        else:
            errors = to_64(torch.zeros((len(mean_gram)-1)))
            _idx = list(range(self.in_features + 1))
            for p in range(len(mean_gram)-1):
                drop_p_idx = _idx[:p] + _idx[p+1:]
                subgram = mean_gram[drop_p_idx][:, drop_p_idx]
                new_model = torch.gesv(mean_gram[drop_p_idx, : ].mm(expand_model), subgram)[0]
                errors[p] = res_ms - two_square_trace(new_model.mm(new_model.t()), subgram)
            
            errors = to_32(errors)
            res_ms = to_32(res_ms)
            return errors, res_ms
    
    def refine(self, drop = 1, quick = True):
        if self.datums_acc == 0:
            raise ValueError(".datums_acc == 0, must input data before refine.")
        
        leave_idx = list(range(self.in_features + 1)) # +1 for bias
        expand_model = to_64(torch.cat((self.model_weight, self.model_bias.view(-1,1)), 1).t())
        mean_gram = pos_gram(to_64(self.gram / self.datums_acc), self.regularizer)
        # avoid singular gram
        res_ms = two_square_trace(expand_model.mm(expand_model.t()), mean_gram) # mean of square sum of y = xA
        if quick:
            new_model = expand_model.clone()
            for d in range(drop):
                subgram = mean_gram[leave_idx][:, leave_idx]
                unbiased_subgram = pos_gram(subgram[:-1, :-1] - torch.mm(subgram[:-1, -1:], subgram[-1:, :-1]))
                cori, comp = torch.symeig(unbiased_subgram, eigenvectors=True) # PCA
                error_bound = 1/(0.000000000000001 + comp**2) # 0.000000000000001 for stability, avoid divide zero
                error_bound *= cori
                error_bound = error_bound.min(1)[0]*((new_model[:-1]**2).sum(1)) # square-norm for weight without bias
                pos = error_bound.min(0)[1] # argmin for bound, position for variable should be remove in leave index
                leave_pos = list(range(len(leave_idx))) # leave_idx for leave_idx
                leave_pos.remove(pos)
                drop_p_gram = subgram[leave_pos][:, leave_pos]
                # ridge regrassion : linear regression with regularlizer, guaranty existence of only one minimun
                new_model = torch.gesv(subgram[leave_pos, : ].mm(new_model), drop_p_gram)[0]
                leave_idx = leave_idx[:pos] + leave_idx[pos+1:] # update leave_idx
            
            mse = res_ms - two_square_trace(new_model.mm(new_model.t()), mean_gram[leave_idx][:, leave_idx])
        else:
            for d in range(drop):
                leave_err = torch.zeros((len(leave_idx)-1))
                for p in range(len(leave_idx)-1):
                    drop_p_idx = leave_idx[:p] + leave_idx[p+1:]
                    drop_p_gram = mean_gram[drop_p_idx][:, drop_p_idx]
                    new_model = torch.gesv(mean_gram[drop_p_idx, : ].mm(expand_model), drop_p_gram)[0]
                    leave_err[p] = res_ms - two_square_trace(new_model.mm(new_model.t()), drop_p_gram)
                
                mse, pos = leave_err.min(0)
                leave_idx = leave_idx[:pos] + leave_idx[pos+1:]
            
            subgram = mean_gram[leave_idx][:, leave_idx]
            new_model = torch.gesv(mean_gram[leave_idx, : ].mm(expand_model), subgram)[0]
        
        # mean square error
        
        new_model = to_32(new_model)
        mse = to_32(mse)
        res_ms = to_32(res_ms)
        return leave_idx[:-1], new_model[:-1].t(), new_model[-1].view((-1)), mse, res_ms
    
    def collinear_cut(self, method, threshold = 0.001, quick = True, drop_limit = None):
        # cheaking near-multicollinearity of regressor, and than use self.refine to drop dimension of nullity
        if method not in ["c", "v", "vr"]:
            raise ValueError("method must be 'c', 'v', 'vr', see https://github.com/Dogiko/Linear-Refine")
        # s_i is singular values of covariance matrix
        # c : condition number, drop = #{s_i/max(s_i) < threshold}
        # v : variance, drop = #{s_i < threshold}
        # vr : variance ratio, drop = #{s_i/sum(s_i) < threshold}
        
        if self.datums_acc == 0:
            raise ValueError(".datums_acc == 0, must input data before refine.")
        
        expand_model = to_64(torch.cat((self.model_weight, self.model_bias.view(-1,1)), 1).t())
        mean_gram = pos_gram(to_64(self.gram / self.datums_acc), self.regularizer) # avoid singular gram
        res_ms = two_square_trace(expand_model.mm(expand_model.t()), mean_gram) # mean of square sum of y = xA
        unbiased_gram = pos_gram(mean_gram[:-1, :-1] - torch.mm(mean_gram[:-1, -1:], mean_gram[-1:, :-1]))
        cori, comp = torch.symeig(unbiased_gram, eigenvectors=False) # PCA without eigenvalues
        if method == "c":
            drop = (cori/cori.max() < threshold).sum()
        elif method == "v":
            drop = (cori < threshold).sum()
        else:
            drop = (cori/cori.sum() < threshold).sum()
        
        if drop_limit is None:
            drop_limit = self.in_features-1
        
        drop = min(drop, drop_limit)
        result = self.refine(drop=drop, quick=quick)
        return result, drop