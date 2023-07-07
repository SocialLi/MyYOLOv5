import yaml
import torch.nn as nn
import contextlib

cfg = 'yolov5l.yaml'
with open(cfg, encoding='ascii', errors='ignore') as ff:
    d = yaml.safe_load(ff)

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                # print(type(a))
                print(isinstance(a, str))
                # args[j] = eval(a) if isinstance(a, str) else a  # eval strings
