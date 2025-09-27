import random
import numpy as np
from typing import Dict, Any

#so for the gaussain mutation, this is the def from gfg, which is kinda nice, people are literally mad in combining diff domains always, fascinating!!

# Gaussian mutation is a technique used in genetic algorithms (GAs) to introduce small random changes in the individuals of a population. It involves adding a random value from a Gaussian (normal) distribution to each element of an individual's vector to create a new offspring.
#  This method is particularly useful for fine-tuning solutions and exploring the domain effectively.


# In Gaussian mutation, the variance of the distribution is determined by parameters such as scale and shrink. The scale controls the standard deviation of the mutation at the first generation, while the shrink controls the rate at which the average amount of mutation decreases over generations.
#  This approach helps in maintaining a balance between exploration and exploitation in the search process.


class F1MutationOperator:
    def __init__(self, mutation_rate: float = 0.6, mutation_strength: float = 0.5):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.param_bounds = self._get_f1_parameter_bounds()
    
    def f1_wing_mutation(self, individual):
        mutated = individual.copy()
        mutations_applied = 0
        
        wing_params = ['total_span', 'root_chord', 'tip_chord', 'sweep_angle', 'dihedral_angle']
        for param in wing_params:
            if random.random() < self.mutation_rate:
                if param == 'root_chord' or param == 'tip_chord':
                    chord_scale = 1 + np.random.normal(0, self.mutation_strength * 0.5)
                    mutated['root_chord'] *= chord_scale
                    mutated['tip_chord'] *= chord_scale
                    mutated['chord_taper_ratio'] = mutated['tip_chord'] / mutated['root_chord']
                    mutated['root_chord'] = np.clip(mutated['root_chord'], 200, 350)
                    mutated['tip_chord'] = np.clip(mutated['tip_chord'], 180, 300)
                    mutations_applied += 1
                else:
                    mutated[param] = self._gaussian_mutate(param, mutated[param])
                    mutations_applied += 1
        
        airfoil_params = ['max_thickness_ratio', 'camber_ratio', 'camber_position', 
                         'leading_edge_radius', 'trailing_edge_thickness', 'upper_surface_radius',
                         'lower_surface_radius']
        for param in airfoil_params:
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1
        
        endplate_params = ['endplate_height', 'endplate_max_width', 'endplate_min_width',
                          'endplate_thickness_base', 'endplate_forward_lean', 
                          'endplate_rearward_sweep', 'endplate_outboard_wrap']
        for param in endplate_params:
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1
        
        y250_params = ['y250_step_height', 'y250_transition_length', 'central_slot_width']
        for param in y250_params:
            if param in mutated and random.random() < self.mutation_rate * 0.8:  
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1
        
        footplate_params = ['footplate_extension', 'footplate_height', 'arch_radius', 'footplate_thickness']
        for param in footplate_params:
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1
        
        manufacturing_params = ['wall_thickness_structural', 'wall_thickness_aerodynamic', 
                               'wall_thickness_details', 'minimum_radius']
        for param in manufacturing_params:
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1

        flap_control_params = ['flap_gap_ratios', 'flap_mounting_angles', 'flap_chord_ratios']
        for param in flap_control_params:
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
                mutations_applied += 1
        
        if 'flap_cambers' in mutated:
            for i in range(len(mutated['flap_cambers'])):
                if random.random() < self.mutation_rate:
                    mutated['flap_cambers'][i] = self._gaussian_mutate('flap_cambers', mutated['flap_cambers'][i])
                    mutations_applied += 1
                if random.random() < self.mutation_rate:
                    mutated['flap_slot_gaps'][i] = self._gaussian_mutate('flap_slot_gaps', mutated['flap_slot_gaps'][i])
                    mutations_applied += 1
                if random.random() < self.mutation_rate and i < len(mutated.get('flap_vertical_offsets', [])):
                    mutated['flap_vertical_offsets'][i] = self._gaussian_mutate('flap_vertical_offsets', mutated['flap_vertical_offsets'][i])
                    mutations_applied += 1
                if random.random() < self.mutation_rate and i < len(mutated.get('flap_horizontal_offsets', [])):
                    mutated['flap_horizontal_offsets'][i] = self._gaussian_mutate('flap_horizontal_offsets', mutated['flap_horizontal_offsets'][i])
                    mutations_applied += 1
        
        if 'strake_heights' in mutated and isinstance(mutated['strake_heights'], list):
            for i in range(len(mutated['strake_heights'])):
                if random.random() < self.mutation_rate:
                    mutated['strake_heights'][i] = self._gaussian_mutate('strake_heights', mutated['strake_heights'][i])
                    mutations_applied += 1
        
        if mutations_applied < 3:
            remaining_params = ['sweep_angle', 'dihedral_angle', 'camber_ratio']
            for param in remaining_params[:3-mutations_applied]:
                if param in mutated:
                    mutated[param] = self._gaussian_mutate(param, mutated[param])
                    mutations_applied += 1
        
        return mutated
    
    def adaptive_f1_mutation(self, individual: Dict, generation: int, max_generations: int):
        adaptive_strength = self.mutation_strength * (1 - generation / max_generations)
        
        original_strength = self.mutation_strength
        self.mutation_strength = adaptive_strength
        
        mutated = self.f1_wing_mutation(individual)
        
        self.mutation_strength = original_strength
        
        return mutated
    
    def _apply_targeted_aggressive_mutations(self, individual: Dict):
        mutated = individual.copy()
        
        if random.random() < 0.6:  # 60% chance for flap system overhaul
            print("🔧 Applying aggressive flap system mutation...")
            
            for i in range(len(mutated.get('flap_slot_gaps', []))):
                if random.random() < 0.8:
                    optimal_gap = np.random.uniform(10, 16)  
                    mutated['flap_slot_gaps'][i] = optimal_gap

                    target_ratio = 0.25 # as per perplexity with its claude 4.0 extenive analysis
                    if i < len(mutated.get('flap_vertical_offsets', [])):
                        mutated['flap_vertical_offsets'][i] = optimal_gap / target_ratio

                        mutated['flap_vertical_offsets'][i] = np.clip(
                            mutated['flap_vertical_offsets'][i], 15, 120
                        )
            
            for i in range(len(mutated.get('flap_cambers', []))):
                if random.random() < 0.7:
                    base_camber = 0.08 + i * 0.02  
                    noise = np.random.normal(0, 0.03)
                    mutated['flap_cambers'][i] = np.clip(
                        base_camber + noise, 0.05, 0.20
                    )
        
        if random.random() < 0.5:
            print("⚖️ Applying aggressive weight optimization...")
            
            estimated_volume = (
                (mutated['total_span'] / 1000) * 
                (mutated['root_chord'] / 1000) * 
                mutated['max_thickness_ratio'] * 0.4
            )
            
            for i in range(len(mutated.get('flap_spans', []))):
                if i < len(mutated.get('flap_root_chords', [])):
                    flap_vol = (
                        (mutated['flap_spans'][i] / 1000) * 
                        (mutated['flap_root_chords'][i] / 1000) * 
                        0.08
                    )
                    estimated_volume += flap_vol
            
            material_density = mutated.get('density', 1600)
            corrected_weight = estimated_volume * material_density / 1000
            
            weight_variation = np.random.uniform(0.85, 1.15)
            mutated['weight_estimate'] = corrected_weight * weight_variation
            mutated['weight_estimate'] = np.clip(mutated['weight_estimate'], 3.0, 8.0)
        
        if random.random() < 0.4:
            print("🎯 Applying aggressive performance mutation...")
            
            extreme_mutations = {
                'camber_ratio': np.random.uniform(0.12, 0.18),  # High camber
                'max_thickness_ratio': np.random.uniform(0.08, 0.12),  # Thin for efficiency
                'sweep_angle': np.random.uniform(2, 6),  # Moderate sweep
                'dihedral_angle': np.random.uniform(1, 4),  # Ground effect optimization
            }
            
            for param, value in extreme_mutations.items():
                if param in mutated and random.random() < 0.6:
                    mutated[param] = value
        
        if random.random() < 0.3:
            print("🏗️ Applying aggressive structural mutation...")
            
            mutated['wall_thickness_structural'] = np.random.uniform(3.5, 5.0)
            mutated['wall_thickness_aerodynamic'] = np.random.uniform(2.0, 3.0)
            mutated['wall_thickness_details'] = np.random.uniform(1.8, 2.5)
            
            mutated['minimum_radius'] = np.random.uniform(0.3, 0.6)
        
        mutated = self._ensure_system_coherence(mutated)
        
        return mutated

    def _ensure_system_coherence(self, individual: Dict):
        if 'root_chord' in individual and 'tip_chord' in individual:
            individual['chord_taper_ratio'] = individual['tip_chord'] / individual['root_chord']
        
        if 'flap_spans' in individual and len(individual['flap_spans']) > 1:
            for i in range(1, len(individual['flap_spans'])):
                if individual['flap_spans'][i] > individual['flap_spans'][i-1]:
                    individual['flap_spans'][i] = individual['flap_spans'][i-1] * 0.95
        
        if 'flap_vertical_offsets' in individual and len(individual['flap_vertical_offsets']) > 1:
            for i in range(1, len(individual['flap_vertical_offsets'])):
                if individual['flap_vertical_offsets'][i] <= individual['flap_vertical_offsets'][i-1]:
                    individual['flap_vertical_offsets'][i] = individual['flap_vertical_offsets'][i-1] * 1.2
        
        if ('endplate_max_width' in individual and 'endplate_min_width' in individual):
            if individual['endplate_min_width'] >= individual['endplate_max_width']:
                individual['endplate_min_width'] = individual['endplate_max_width'] * 0.4
        
        return individual
    
    def aggressive_mutation(self, individual: Dict):
        mutated = individual.copy()
        
        aggressive_rate = min(0.6, self.mutation_rate * 2.0)
        aggressive_strength = min(0.5, self.mutation_strength * 1.5)
        
        original_rate = self.mutation_rate
        original_strength = self.mutation_strength
        
        self.mutation_rate = aggressive_rate
        self.mutation_strength = aggressive_strength
        
        mutated = self.f1_wing_mutation(mutated)

        mutated = self._apply_targeted_aggressive_mutations(mutated)
        
        self.mutation_rate = original_rate
        self.mutation_strength = original_strength
        
        return mutated
    
    def _gaussian_mutate(self, param_name: str, current_value: float):
        noise = np.random.normal(0, self.mutation_strength)
        new_value = current_value * (1 + noise)
        
        if param_name in self.param_bounds:
            min_val, max_val = self.param_bounds[param_name]
            new_value = np.clip(new_value, min_val, max_val)
        
        return new_value
    
    def _get_f1_parameter_bounds(self):
        return {
            'total_span': (1400, 1800),
            'root_chord': (200, 350),
            'tip_chord': (180, 300),
            'sweep_angle': (0, 8),
            'dihedral_angle': (0, 6),
            'max_thickness_ratio': (0.08, 0.25),
            'camber_ratio': (0.04, 0.18),
            'camber_position': (0.25, 0.55),
            'leading_edge_radius': (1.5, 4.0),
            'trailing_edge_thickness': (1.0, 4.0),
            'upper_surface_radius': (600, 1200),
            'lower_surface_radius': (800, 1400),
            'endplate_height': (200, 330),
            'endplate_max_width': (80, 180),
            'endplate_min_width': (25, 80),
            'endplate_thickness_base': (6, 15),
            'endplate_forward_lean': (2, 12),
            'endplate_rearward_sweep': (5, 18),
            'endplate_outboard_wrap': (10, 25),
            'y250_step_height': (10, 25),
            'y250_transition_length': (60, 150),
            'central_slot_width': (20, 50),
            'footplate_extension': (50, 120),
            'footplate_height': (20, 50),
            'arch_radius': (100, 200),
            'footplate_thickness': (3, 8),
            'wall_thickness_structural': (3, 6),
            'wall_thickness_aerodynamic': (2, 4),
            'wall_thickness_details': (1.5, 3.0),
            'minimum_radius': (0.2, 0.8),
            'flap_cambers': (0.06, 0.18),
            'flap_slot_gaps': (6, 12),
            'flap_vertical_offsets': (15, 120),
            'flap_horizontal_offsets': (20, 120),
            'strake_heights': (20, 80),
            'weight_estimate': (3.0, 8.0),
            'flap_count': (3, 6),    
        }
