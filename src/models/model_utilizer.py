import os
import torch
import torch.nn as nn

from pathlib import Path

class ModuleUtilizer(object):
    """Module utility class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.

    """
    def __init__(self, configer):
        """Class constructor for Module utility"""
        self.configer = configer
        self.device = self.configer.get("device")

        self.save_policy = self.configer.get("checkpoints", "save_policy")
        if self.save_policy in ["early_stop", "earlystop"]:
            self.save = self.early_stop
        elif self.save_policy == "all":
            self.save = self.save_all
        else:
            self.save = self.save_best

        self.best_accuracy = 0
        self.last_improvement = 0

    def update_optimizer(self, net, iters):
        """Load optimizer and adjust learning rate during training, if using SGD.

                Args:
                    net (torch.nn.Module): Module in use
                    iters (int): current iteration number

                Returns:
                    optimizer (torch.optim.optimizer): PyTorch Optimizer
                    lr (float): Learning rate for training procedure

        """
        optim = self.configer.get('solver', 'type')
        decay = self.configer.get('solver', 'weight_decay')

        if optim == "Adam":
            print("Using Adam.")
            lr = self.configer.get('solver', 'base_lr')
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                         weight_decay=decay)

        elif optim == "AdamW":
            lr = self.configer.get('solver', 'base_lr')
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                          weight_decay=decay)

        elif optim == "RMSProp":
            lr = self.configer.get('solver', 'base_lr')
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                            weight_decay=decay)

        elif optim == "SGD":
            print("Using SGD")
            policy = self.configer.get('solver', 'lr_policy')

            if policy == 'fixed':
                lr = self.configer.get('solver', 'base_lr')

            elif policy == 'step':
                gamma = self.configer.get('solver', 'gamma')
                ratio = gamma ** (iters // self.configer.get('solver', 'step_size'))
                lr = self.configer.get('solver', 'base_lr') * ratio

            elif policy == 'exp':
                lr = self.configer.get('solver', 'base_lr') * (self.configer.get('solver', 'gamma') ** iters)

            elif policy == 'inv':
                power = -self.configer.get('solver', 'power')
                ratio = (1 + self.configer.get('solver', 'gamma') * iters) ** power
                lr = self.configer.get('solver', 'base_lr') * ratio

            elif policy == 'multistep':
                lr = self.configer.get('solver', 'base_lr')
                for step_value in self.configer.get('solver', 'stepvalue'):
                    if iters >= step_value:
                        lr *= self.configer.get('solver', 'gamma')
                    else:
                        break
            else:
                raise NotImplementedError('Policy:{} is not valid.'.format(policy))

            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = lr,
                                        momentum=self.configer.get('solver', 'momentum'), weight_decay=decay)

        else:
            raise NotImplementedError('Optimizer: {} is not valid.'.format(optim))

        return optimizer, lr

    def load_net(self, net):
        """Loading net method. If resume is True load from provided checkpoint, if False load new DataParallel

                Args:
                    net (torch.nn.Module): Module in use

                Returns:
                    net (torch.nn.DataParallel): Loaded Network module
                    iters (int): Loaded current iteration number, 0 if Resume is False
                    epoch (int): Loaded current epoch number, 0 if Resume is False
                    optimizer (torch.nn.optimizer): Loaded optimizer state, None if Resume is False

        """
        iters = 0
        epoch = 0
        optimizer = None
        if self.configer.get('resume') is not None:
            print('Restoring checkpoint: ', self.configer.get('resume'))
            checkpoint_dict = torch.load(self.configer.get('resume'))
            # Remove "module." from DataParallel, if present
            checkpoint_dict['state_dict'] = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in
                                             checkpoint_dict['state_dict'].items()}
            net.load_state_dict(checkpoint_dict['state_dict'])
            iters = checkpoint_dict['iter'] if 'iter' in checkpoint_dict else 0
            optimizer = checkpoint_dict['optimizer'] if 'optimizer' in checkpoint_dict else None
            epoch = checkpoint_dict['epoch'] if 'epoch' in checkpoint_dict else None
        net = nn.DataParallel(net, device_ids=self.configer.get('gpu')).to(self.device)
        return net, iters, epoch, optimizer

    def _save_net(self, net, optimizer, iters, epoch, all=False):
        """Saving net state method.

                Args:
                    net (torch.nn.Module): Module in use
                    optimizer (torch.nn.optimizer): Optimizer state to save
                    iters (int): Current iteration number to save
                    epoch (int): Current epoch number to save

        """
        state = {
            'iter': iters,
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoints_dir = str(Path(self.configer.get('checkpoints', 'save_dir')) / self.configer.get("dataset"))
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if all:
            latest_name = '{}_{}.pth'.format(self.configer.get('checkpoints', 'save_name'), epoch)
        else:
            latest_name = 'best_{}.pth'.format(self.configer.get('checkpoints', 'save_name'))
        torch.save(state, os.path.join(checkpoints_dir, latest_name))

    def save_all(self, accuracy, net, optimizer, iters, epoch):
        self._save_net(net, optimizer, iters, epoch, all=True)
        return accuracy

    def save_best(self, accuracy, net, optimizer, iters, epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_net(net, optimizer, iters, epoch)
            return self.best_accuracy
        else:
            return 0

    def early_stop(self, accuracy, net, optimizer, iters, epoch):
        ret = self.save_best(accuracy, net, optimizer, iters, epoch)
        if ret > 0:
            self.last_improvement = 0
        else:
            self.last_improvement += 1
        if self.last_improvement >= self.configer.get("checkpoints", "early_stop"):
            return -1
        else:
            return ret