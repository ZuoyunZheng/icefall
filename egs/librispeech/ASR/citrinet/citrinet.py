"""
Code copied from https://github.com/erasedwalt/CTC-ASR/blob/main/src/models/citrinet.py 
"""

from typing import Optional, Tuple, Union, Iterable, Dict, Any, Callable

import torch
from torch import Tensor
from torch import nn
from transformer import Supervisions, Transformer, encoder_padding_mask
from torch_optimizer import Optimizer


KERNEL_SIZES = {
    'K1': [5, 3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, 9, 41],
    'K2': [5, 5, 7, 7, 9, 9, 11, 7, 7, 9, 9, 11, 11, 13, 13, 13, 15, 15, 17, 17, 19, 19, 41],
    'K3': [5, 9, 9, 11, 13, 15, 15, 9, 11, 13, 15, 15, 17, 19, 19, 21, 21, 23, 25, 27, 27, 29, 41],
    'K4': [5, 11, 13, 15, 17, 19, 21, 13, 15, 17, 19, 21, 23, 25, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}


########## SQUEEZE-EXCITE TAKEN FROM https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/jasper.py ##########


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation sub-module.
    Args:
        channels (int): Input number of channels.
        reduction_ratio (int): Reduction ratio for "squeeze" layer.
        context_window (int): Integer number of timesteps that the context
            should be computed over, using stride 1 average pooling.
            If value < 1, then global context is computed.
        interpolation_mode (str): Interpolation mode of timestep dimension.
            Used only if context window is > 1.
            The modes available for resizing are: `nearest`, `linear` (3D-only),
            `bilinear`, `area`
    """
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        #context_window: int = -1,
        interpolation_mode: str = 'nearest',
    ) -> None:

        super(SqueezeExcite, self).__init__()
        self.interpolation_mode = interpolation_mode
        #self.context_window = context_window
        #self.pool = nn.AdaptiveAvgPool1d(context_window)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps = x.size()[:3]

        #if timesteps < self.context_window:
        y = self.gap(x)
        #else:
        #    y = self.pool(x) 
        y = y.transpose(1, -1) 
        y = self.fc(y) 
        y = y.transpose(1, -1)

        #if self.context_window > 0:
        #    y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)
        y = torch.sigmoid(y)
        return x * y


class CitrinetBlock(nn.Module):
    """
    Citrinet block
    This class implements Citrinet block from the paper:
    https://arxiv.org/pdf/2104.01721.pdf
    Args:
        R (int): Number of repetition of each subblock in block
        C (int): Number of channels
        kernel_size (int): Kernel size
        stride (int, optional): Stride
        dropout (float, optional): Dropout probability
    """
    def __init__(
        self,
        R: int,
        C: int,
        kernel_size: int,
        stride: int = None,
        dropout: float = 0.
    ) -> None:

        super(CitrinetBlock, self).__init__()

        self.R = R

        self.block = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=kernel_size,
                    groups=C,
                    stride=stride if stride and i == 0 else 1,
                    padding=kernel_size // 2
                ),
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=1
                ),
                nn.BatchNorm1d(num_features=C),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            for i in range(R)
        ])

        self.block.append(
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=kernel_size,
                    groups=C,
                    padding=kernel_size // 2
                ),
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=1
                ),
                nn.BatchNorm1d(num_features=C),
                SqueezeExcite(
                    channels=C,
                    reduction_ratio=8
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
        )

        self.res_connection = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1,
                stride=stride if stride else 1
            ),
            nn.BatchNorm1d(num_features=C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_connection(x)
        for i, subblock in enumerate(self.block):
            for j, module in enumerate(subblock):
                if i == self.R and j == 4:
                    x += residual
                    del residual
                x = module(x)
        return x


class Citrinet(nn.Module):
    """
    Citrinet
    This class implements Citrinet from the paper:
    https://arxiv.org/pdf/2104.01721.pdf

    Args:
        C (int): Number of channels
        K (int): Type of kernel sizes
        R (int): Number of subblocks in each block
        num_features (int): Number of input channels
        num_classes (int): Vocab size
        dropout (float, optional): Dropout probability
    """
    def __init__(
        self,
        C: int,
        K: int,
        R: int,
        num_features: int,
        num_classes: int,
        dropout: float = 0.1
    ) -> None:

        super(Citrinet, self).__init__()

        self.K = KERNEL_SIZES['K' + str(K)]
        self.megablocks = [
            [1, 7],
            [7, 14],
            [14, 22]
        ]

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=C,
                groups=num_features,
                kernel_size=self.K[0],
                padding=self.K[0] // 2,
            ),
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=C),
            nn.ReLU(inplace=True)
        )

        self.megablock1 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 1],
                stride=2 if i == 0 else None,
                dropout=dropout
            )
            for i in range(6) # 6 block in first megablock from paper
        ])

        self.megablock2 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 7],
                stride=2 if i == 0 else None,
                dropout=dropout
            )
            for i in range(7) # 7 block in second
        ])

        self.megablock3 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 14],
                stride= 2 if i == 0 else None, #doesn't work with stride=2
                dropout=dropout
            )
            for i in range(8) # 8 block in third
        ])

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                groups=C,
                kernel_size=self.K[-1],
                padding=self.K[-1] // 2
            ),
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=C),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=num_classes,
                kernel_size=1
            )
        )

    def forward(
        self, 
        x: torch.Tensor,
        supervision: Optional[Supervisions] = None
        ) -> torch.Tensor:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C).
          supervisions: (Not used)
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute the encoder padding mask, which is used as memory key
            padding mask for the decoder.
        Returns:
          The output tensor has shape [N, T, C]
        """
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.megablock1(x)
        x = self.megablock2(x)
        x = self.megablock3(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(
            1, 2
        )  # (T, N, C) -> (N, T, C) -> linear expects "features" in the last dim
        return x.log_softmax(dim=-1)



class NovoGrad(Optimizer):
    r"""Implements Novograd optimization algorithm.

    It has been proposed in `Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging (default: False)
        amsgrad: whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`
            (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Yogi(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        >>> optimizer.step()
        >>> scheduler.step()

    __ https://arxiv.org/abs/1905.11286

    Note:
        Reference code: https://github.com/NVIDIA/DeepLearningExamples
    """

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0),
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_averaging: bool = False,
        amsgrad: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            amsgrad=amsgrad,
        )

        super(NovoGrad, self).__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super(NovoGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'NovoGrad does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(
                        state['exp_avg'].device
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq.
                        # grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(
                            state['exp_avg'].device
                        )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg.
                    # till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(exp_avg, alpha=-group['lr'])

        return loss
