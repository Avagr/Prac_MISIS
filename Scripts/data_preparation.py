import math

import numpy as np
import pandas as pd
import os
import pymatgen.core as mg
import scipy.constants as const


class DataExtractor:

    def __init__(self):
        self.data: dict[str, list] = {tag: [] for tag in
                                      ['el1', 'el2', 'el3', 'conc1', 'conc2', 'conc3', 'e', 'a']}
        self.valences: dict[mg.Element, float] = {}

    @property
    def dataframe(self):
        return pd.DataFrame(self.data)

    def read_file(self, data_dir: str, file: str, num_elements: int, base_element: str):
        with open(os.path.join(data_dir, file), 'r') as f:
            match num_elements:
                case 1:
                    values = f.readline().strip().split()
                    self.data['el1'].append(base_element)
                    self.data['el2'].append(None)
                    self.data['el3'].append(None)
                    self.data['conc1'].append(1.)
                    self.data['conc2'].append(0.)
                    self.data['conc3'].append(0.)
                    self.data['e'].append(values[7])
                    self.data['a'].append(values[10])
                case 2:
                    for line in f:
                        values = list(map(float, line.strip().split()))
                        self.data['el1'].append(base_element)
                        self.data['el2'].append(file.split('_')[0])
                        self.data['el3'].append(None)
                        self.data['conc2'].append(values[0] / 100)
                        self.data['conc1'].append(1 - self.data['conc2'][-1])
                        self.data['conc3'].append(0.)
                        self.data['e'].append(values[7])
                        self.data['a'].append(values[10])
                case 3:
                    for line in f:
                        values = list(map(float, line.strip().replace('-', ' -').split()))
                        self.data['el1'].append(base_element)
                        self.data['el2'].append(file.split('_')[0])
                        self.data['el3'].append(file.split('_')[1])
                        sum_conc = values[0] / 100
                        self.data['conc2'].append(values[1] / 100)
                        self.data['conc3'].append(sum_conc - values[1] / 100)
                        self.data['conc1'].append(1. - sum_conc)
                        self.data['e'].append(values[8])
                        self.data['a'].append(values[9])

    def extract_properties(self, row):
        elements = [mg.Element(row[el]) for el in ['el1', 'el2', 'el3'] if row[el] is not None]
        concentrations = (row['conc1'], row['conc2'], row['conc3'])
        mean_radius = 0.
        for el in elements:
            mean_radius += el.atomic_radius
        mean_radius /= len(elements)
        delta, v, xi, c_vec, s_mix, v_sound = 0, 0, 0, 0, 0, 0
        for el, conc in zip(elements, concentrations):
            delta += conc * (1 - el.atomic_radius / mean_radius) ** 2
            v += conc * 4 / 3 * const.pi * (el.atomic_radius ** 3)
            xi += conc * el.X
            if el not in self.valences:
                self.valences[el] = self._parse_valence(el.electronic_structure)
            c_vec += conc * self.valences[el]
            s_mix += conc * math.log(conc)
            # v_sound += conc * el.velocity_of_sound speed of sound doesn't work
        delta = math.sqrt(delta)
        s_mix = -const.R * s_mix
        return delta, v, xi, c_vec, s_mix

    @staticmethod
    def _parse_valence(electronic_structure: str) -> float:
        res = 0.
        for block in electronic_structure.split('.')[1:]:
            res += float(block[2:])
        return res
