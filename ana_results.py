"""
serach difference hyper-params of finetuen.py
"""

import os
from os.path import join, exists
import sys
import argparse
import numpy as np

class AnaResults:
    def __init__(self):
        self.dict_result1 = {}
        self.dict_result2 = {}
    def _add_to_dict(self, d, name, value):
        if not name in d:
            d[name] = []
        d[name].append(value)
    def _print_dict(self, d, prefix):
        names = sorted(d.keys())
        for name in names:
            outs = [prefix] + list(name) + [str(np.mean(d[name])), str(np.std(d[name])), str(len(d[name]))]
            print('\t'.join(outs))
    def add(self, settings, auc):
        name = tuple(settings)
        self._add_to_dict(self.dict_result1, name, auc)
        name = tuple([s for s in settings if not s.startswith('dataset')])
        self._add_to_dict(self.dict_result2, name, auc)
    def Print(self):
        self._print_dict(self.dict_result1, 'AnaResults1')
        self._print_dict(self.dict_result2, 'AnaResults2')

def main(args):
    ana_results = AnaResults()
    for line in sys.stdin:
        segs = line.strip().split('\t')
        ana_results.add(segs[1:-1], float(segs[-1]))
    ana_results.Print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
