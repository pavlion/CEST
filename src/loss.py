import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights



class LossHarmonizer:

    def __init__(self, bins=30, momentum=0, max_loss=2.0, device='cuda'):
        
        self.bins = bins # M
        self.momentum = momentum

        # Define the boundary of each unit region
        min_loss = 0.0
        self.M = (max_loss - min_loss) / bins
        self.edges = torch.arange(bins + 1).float().to(device) / bins * (max_loss - min_loss)
        if momentum > 0:
            self.history_bins = torch.zeros(bins).to(device)

    def reweight_loss(self, loss):
        
        loss_ = loss.data
        beta = torch.zeros_like(loss_)

        N = len(loss_)
        for i in range(self.bins):
            ids = (loss_ >= self.edges[i]) & (loss_ < self.edges[i+1])
            num_in_bin = ids.sum().item()
            
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.history_bins[i] = self.momentum * self.history_bins[i] + (1 - mmt) * num_in_bin
                    beta[ids] = N / self.history_bins[i] / self.bins
                else:
                    beta[ids] = N / num_in_bin / self.bins
        
        loss = loss * beta

        return loss

    def reset(self):
        if self.momentum > 0:
            self.history_bins = torch.zeros(bins).to(device)


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                                    target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


def PHuberCrossEntropy(input, target, tau: float = 10, reduction: str = 'mean'):

    prob_thresh = 1 / tau
    boundary_term = math.log(tau) + 1

    p = F.softmax(input, dim=-1)
    p = p[torch.arange(p.shape[0]), target]

    loss = torch.empty_like(p)
    clip = p <= prob_thresh
    loss[clip] = -tau * p[clip] + boundary_term
    loss[~clip] = -torch.log(p[~clip])

    if reduction == 'none':
        return loss
        
    return torch.mean(loss)



def taylor_softmax(self, x, dim=1, n=2):
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n+1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    return out

def TaylorCrossEntropyLoss(self, logits, labels, n=2, reduction='mean'):

    assert n % 2 == 0

    ### Taylor softmax
    log_probs = taylor_softmax(logits, dim=-1, n=2).log()
    loss = F.nll_loss(log_probs, labels, reduction=reduction)

    return loss




if __name__ == '__main__':

    a = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.3, 0.5])

    ghm = LossHarmonizer(bins=30, momentum=0, max_loss=2.0, device='cpu')
    b = ghm.reweight_loss(a)
    print(b)
    print(a.grad, b.grad)
