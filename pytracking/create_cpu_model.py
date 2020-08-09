import os
import torch
import ltr.models.bbreg.atom as atom_models
from ltr.admin import loading

def main():
    net = atom_models.atom_mobilenetsmall(backbone_pretrained=False, cpu=True)
    path = '/content/pytracking/pytracking/networks/atom_cfkd.pth.tar'
    net = loading.load_weights(net, path, strict=True)

    net_type = type(net).__name__
    state = {
        # 'epoch': self.epoch,
        # 'actor_type': actor_type,
        'net_type': net_type,
        'net': net.state_dict(),
        'net_info': getattr(net, 'info', None),
        'constructor': getattr(net, 'constructor', None)
        # 'optimizer': self.optimizer.state_dict(),
        # 'stats': self.stats,
        # 'settings': self.settings
    }

    torch.save(state, '/content/pytracking/pytracking/networks/atom_cfkd_cpu.tmp')
    os.rename('/content/pytracking/pytracking/networks/atom_cfkd_cpu.tmp', '/content/pytracking/pytracking/networks/atom_cfkd_cpu.pth.tar')


if __name__ == '__main__':
    main()