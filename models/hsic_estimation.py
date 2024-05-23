import sys
import math
import torch
import torch.nn.functional as F

from utils import device
from config import args
from models.cka import centering



class HSICEstimator:
    
    def __init__(self, scaleList:list=None):
        '''
        Args:
            scaleList: List. A list of scalar for further scaling the statistics.
        '''
        'Initialize the scaling coefficient list.'
        if scaleList is None:
            self.scaleList = [0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 5.0, 10.0]
        else:
            self.scaleList = scaleList
    
    
    def estimate_unbiased_hsic(self, X:torch.tensor, Y:torch.tensor, scale:torch.tensor, 
                               stat_mode:str='median', q_quantile:float=0.25, 
                               require_variance:bool=False, is_scaled:bool=False):
        '''
        Estimate HSIC in the unbiased way.
        Args:
            X: Tensor with shape [batch_size, dim].
            Y: Tensor with shape [batch_size, dim] for feature embeddings and [batch_size, 1] for labels.
        Return:
            hsic_value: A scalar tensor.
        '''
        m = X.size(0)
        
        'Generate tilde data'
        if args['kernel.type'] == 'rbf':
            KX = rbf(X, stat_mode=stat_mode, scale=scale, q_quantile=q_quantile)
            KY = rbf(Y, stat_mode=stat_mode, scale=scale, q_quantile=q_quantile)
            
        elif args['kernel.type'] == 'imq':
            KX = imq(X, scale=scale)
            KY = imq(Y, scale=scale)
        
        elif args['kernel.type'] == 'linear':
            KX = linear(X)
            KY = linear(Y)
            
        else:
            raise ValueError("Unrecognized kernel type! Please choose kernel from ['rbf', 'imq].")
        
        tilde_X = KX - torch.diag(torch.diag(KX))
        tilde_Y = KY - torch.diag(torch.diag(KY))
        
        'Calculate 3 terms of unbiased HSIC respectively.'
        trace_KXKY = torch.sum(tilde_X * tilde_Y)   # trace of $tildeX@tildeY
        vecTildeKX_vecTildeKY = (vectorize(tilde_X) * vectorize(tilde_Y)) / ((m-1)*(m-2))
        vecTildeKXKY = -(2*vectorize(torch.mm(tilde_X, tilde_Y))) / (m-2)
        
        hsic_value = (trace_KXKY + vecTildeKX_vecTildeKY + vecTildeKXKY) / (m**2 - 3*m)
        
        if require_variance:
            square_sigma = self._estimation_variance(tilde_X, tilde_Y, hsic_value)
            hsic_value = hsic_value / torch.sqrt(square_sigma + args['epsilon'])
            if torch.isnan(hsic_value).item():
                print(f"Unexpected NaN value: {square_sigma}!")
                sys.exit()
        
        if is_scaled:
            hsic_value = hsic_value*m
        
        return hsic_value
    
    
    def _estimation_variance(self, tildeX:torch.tensor, tildeY:torch.tensor, hsic_value:torch.tensor):
        '''
        Estimate the variance of the hsic. For more concrete details, please refer to Eq (5) of 
            Song et al. Feature selection via Dependence Maximization, JMLR 2012.
        Args:
            tildeX, tildeY: Tensor with shape [batch_size, batch_size].
            hsic_value: Scalar tensor. HSIC value.
        Return:
            variance: Scalar tensor.
        '''
        m = tildeX.size(0)  # batch size

        # Compute h
        term1 = (m-2)**2 * torch.mm(tildeX * tildeY, vector_ones(m, device))
        term2 = (m-2) * (torch.sum(tildeX * tildeY)*vector_ones(m, device) - torch.mm(torch.mm(tildeX, tildeY), vector_ones(m, device)) - torch.mm(torch.mm(tildeY, tildeX), vector_ones(m, device)))
        term3 = -m * (torch.mm(tildeX, vector_ones(m, device)) * torch.mm(tildeY, vector_ones(m, device)))
        term4 = vectorize(tildeY) * torch.mm(tildeX, vector_ones(m, device))
        term5 = vectorize(tildeX) * torch.mm(tildeY, vector_ones(m, device))
        term6 = -vectorize(torch.mm(tildeX, tildeY)) * vector_ones(m, device)
        
        h = term1 + term2 + term3 + term4 + term5 + term6
        
        # Compute R
        deter = (m-1)*(m-2)*(m-3)
        R = torch.mm(h.t(), h) / deter
        R = R / deter
        R = R / (4*m)
        
        # variance
        variance = 16 * (R - hsic_value**2) / m
        
        return variance
    
    
    def get_best_bandwidths(self, X, Y, stat_mode='median', q_quantile=0.25):
        '''
        Select best bandwidths.
        Args:
            X: Tensor with any shape. Usually feature embeddings with shape [batch_size, c, h, w].
            Y: Tensor with any shape. Can be either feature embeddings or one_hot labels with the shape [batch_size, 1].
        '''
        bestScale_hzy = 1.0
        bestScale_hzz = 1.0
        
        'Flatten and normalize the given data'
        normalized_X = F.normalize(X.flatten(1), dim=-1)
        normalized_Y = F.normalize(F.one_hot(Y).flatten(1).float(), dim=-1)

        
        # search for hzy
        max_hzy = 0.0
        max_hzz = 0.0
        for scale in self.scaleList:
            hsic_zy = self.estimate_unbiased_hsic(X=normalized_X, Y=normalized_Y, 
                                                  stat_mode=stat_mode, q_quantile=q_quantile,
                                                  scale=scale, require_variance=True)
            if hsic_zy > max_hzy:
                max_hzy = hsic_zy
                bestScale_hzy = scale
            
        bestScale_hzz = bestScale_hzy
                    
        return bestScale_hzy, bestScale_hzz


# *********************************** Functions ***********************************************
def rbf(X, stat_mode='median', scale=None, q_quantile=0.25):
    GX = torch.mm(X, X.transpose(0,1) )
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).transpose(0,1)
    
    if stat_mode == 'median':
        try:
            stat = scale*torch.median(KX[KX != 0])
        except:
            stat = torch.zeros(1).to(KX.device)
    
    elif stat_mode == 'quantile':
        try:
            stat = scale*torch.quantile(KX[KX != 0], q=q_quantile)
        except:
            stat = torch.zeros(1).to(KX.device)
            
    else:
        raise ValueError("Unrecognized statistical mode!")
    
    sigma = math.sqrt(stat.clamp(min=1e-12))
    KX = KX * (-0.5 / (sigma * sigma))
    KX = torch.exp(KX)
    return KX


def imq(X, scale=None):
    KX = torch.sum((X.unsqueeze(1) - X.unsqueeze(0)).pow(2), dim=-1)
    
    C = scale * 1.0

    KX = C / torch.sqrt(C*C + KX)
    return KX


def linear(X: torch.tensor) -> torch.tensor:
    X = X / torch.norm(X)
    return torch.mm(X, X.t())
    


def get_meidian(rbf_mat:torch.tensor):
    '''
    Calculate the median of the given rbf kernel.
    Args:
        rbf_mat: Tensor with shape [batch_size, batch_size]. RBF matrix.
    Return:
        median: scalar tensor.
    '''
    try:
        median = torch.median(rbf_mat[rbf_mat != 0])
    except:
        median = torch.zeros(1).to(device)
    
    return median.clamp(min=1e-12)


def get_quantile(rbf_mat:torch.tensor, q:float=0.5):
    '''
    Calculate the quantile of the given rbf kernel.
    Args:
        rbf_mat: Tensor with shape [batch_size, batch_size]. RBF matrix.
        q: Float scalar. Portion of the quantile selected from [0.25, 0.5, 0.75]. Default: 0.5
    Return:
        quantile: A scalar tensor.
    '''
    try:
        quantile = torch.quantile(rbf_mat[rbf_mat != 0], q=q)
    except:
        quantile = torch.zeros(1).to(device)
        
    return quantile.clamp(min=1e-12)


def vectorize(data):
    '''
    Used for calculating: 1^{T}A1, A can be an arbitrary matrix.
    Args:
        data: Tensor. Given data to be vectorize.
    Return
        A scalar Tensor.
    '''
    return torch.mm(torch.ones((1, data.size(0))).to(device), torch.mm(data, torch.ones((data.size(0), 1)).to(device)))


def vector_ones(bsz, device):
    '''
    Args:
        bsz: int. batch size.
        device: GPU info.
    Return:
        A all-ones tensor with the shape [batch_size, 1].
    '''
    return torch.ones((bsz, 1)).to(device)