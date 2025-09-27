import random
import copy
from typing import Dict, Any, Tuple, List

class CrossoverOps:
    def __init__(self, crossover_rate: float = 0.8):
        self.crossover_rate = crossover_rate
        self.scalar_params = {
            'total_span', 'root_chord', 'tip_chord', 'sweep_angle', 'dihedral_angle',
            'max_thickness_ratio', 'camber_ratio', 'camber_position',
            'endplate_height', 'endplate_max_width', 'endplate_min_width',
            'y250_step_height', 'y250_transition_length', 'central_slot_width'
        }
        self.array_params = {
            'flap_spans', 'flap_root_chords', 'flap_tip_chords', 'flap_cambers',
            'flap_slot_gaps', 'flap_vertical_offsets', 'flap_horizontal_offsets',
            'strake_heights'
        }

    def f1_aero_crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Wing structure crossover (keep main wing coherent)
        if random.random() < 0.4:
            wing_params = ['total_span', 'root_chord', 'tip_chord', 'chord_taper_ratio', 'sweep_angle']
            for param in wing_params:
                child1[param], child2[param] = parent2[param], parent1[param]
        
        # Airfoil crossover (keep airfoil profile coherent)
        if random.random() < 0.3:
            airfoil_params = ['max_thickness_ratio', 'camber_ratio', 'camber_position']
            for param in airfoil_params:
                child1[param], child2[param] = parent2[param], parent1[param]
        
        # Flap system crossover (entire system together)
        if random.random() < 0.5:
            for param in self.array_params:
                if param.startswith('flap_'):
                    child1[param], child2[param] = parent2[param].copy(), parent1[param].copy()
        
        # Endplate system crossover
        if random.random() < 0.3:
            endplate_params = ['endplate_height', 'endplate_max_width', 'endplate_min_width',
                             'endplate_forward_lean', 'endplate_rearward_sweep', 'endplate_outboard_wrap']
            for param in endplate_params:
                if param in child1:
                    child1[param], child2[param] = parent2[param], parent1[param]
        
        return child1, child2
        #recommended by sudharsan again

    def uniform_crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        for param in self.scalar_params:
            if param in parent1 and random.random()<0.5:
                child1[param], child2[param] = parent2[param], parent1[param]
        
        for param in self.array_params:
            if param in parent1 and param in parent2:
                for i in range(min(len(parent1[param]), len(parent2[param]))):
                    if random.random() < 0.5:
                        child1[param][i], child2[param][i] = parent2[param][i], parent1[param][i]
        
        return child1, child2