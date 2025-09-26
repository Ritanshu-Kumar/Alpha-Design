import torch
import torch.nn as nn
import numpy as np

class ParamterTweaker:
    def __init__(self, param_bounds=None, mod_scale=0.1):
        self.param_bounds = param_bounds #so this defines the range of the parameters
        self.mod_scale = mod_scale #this is like the learning rate or low pass layer on how much to modify the parameters

    def apply_neural_tweaks(self, current_params, policy_output, exploration=True):
        if not isinstance(current_params, torch.Tensor):
            current_params = torch.tensor(current_params, dtype=torch.float32)
        if not isinstance(policy_output, torch.Tensor):
            policy_output = torch.tensor(policy_output, dtype=torch.float32)

        device = current_params.device
        policy_output = policy_output.to(device)

        mod_d = torch.tanh(policy_output) 

        param_changes = mod_d * self.mod_scale #this is the amount by which the parameters are modified
        modified_params = current_params + param_changes #this is the new parameters changed according to the policy output

        if exploration:
            noise = torch.randn_like(modified_params) * 0.01
            modified_params += noise


        if self.param_bounds is not None:
            modified_params = torch.clamp(modified_params, self.param_bounds['min'], self.param_bounds['max'])

        return modified_params
    
    def genetic_to_neural_params(self, genetic_population):
        try:
            if isinstance(genetic_population, list):
                flattened_params = []
                
                for individual in genetic_population:
                    individual_params = []
                    
                    if isinstance(individual, dict):
                        for key, value in individual.items():
                            if isinstance(value, (list, tuple)):
                                individual_params.extend([float(v) for v in value if isinstance(v, (int, float))])
                            elif isinstance(value, (int, float)):
                                individual_params.append(float(value))
                            elif isinstance(value, str):
                                continue
                            else:
                                try:
                                    individual_params.append(float(value))
                                except (ValueError, TypeError):
                                    continue
                    
                    elif isinstance(individual, (list, tuple)):
                        for param in individual:
                            if isinstance(param, (list, tuple)):
                                individual_params.extend([float(v) for v in param if isinstance(v, (int, float))])
                            elif isinstance(param, (int, float)):
                                individual_params.append(float(param))
                            elif isinstance(param, str):
                                continue
                            else:
                                try:
                                    individual_params.append(float(param))
                                except (ValueError, TypeError):
                                    continue
                    
                    else:
                        if isinstance(individual, (int, float)):
                            individual_params.append(float(individual))
                        elif not isinstance(individual, str):
                            try:
                                individual_params.append(float(individual))
                            except (ValueError, TypeError):
                                individual_params.append(0.0) 
                    
                    flattened_params.append(individual_params)
                
                return torch.tensor(flattened_params, dtype=torch.float32)
            
            else:
                if isinstance(genetic_population, dict):
                    flattened = []
                    for key, value in genetic_population.items():
                        if isinstance(value, (list, tuple)):
                            flattened.extend([float(v) for v in value if isinstance(v, (int, float))])
                        elif isinstance(value, (int, float)):
                            flattened.append(float(value))
                        elif isinstance(value, str):
                            continue
                        else:
                            try:
                                flattened.append(float(value))
                            except (ValueError, TypeError):
                                continue
                    return torch.tensor([flattened], dtype=torch.float32)
                else:
                    return torch.tensor(genetic_population, dtype=torch.float32)
        
        except Exception as e:
            print(f"❌ Error in genetic_to_neural_params: {e}")
            return torch.zeros(1, 50, dtype=torch.float32)
        
    def _apply_bounds(self, params):
        
        if isinstance(self.param_bounds, dict):
            if 'min' in self.param_bounds and 'max' in self.param_bounds:
                min_bounds = torch.tensor(self.param_bounds['min'], 
                                        dtype=params.dtype, device=params.device)
                max_bounds = torch.tensor(self.param_bounds['max'], 
                                        dtype=params.dtype, device=params.device)
                
                params = torch.clamp(params, min_bounds, max_bounds)
        
        return params
    
    def neural_to_cfd(self, neural_params):
        if isinstance(neural_params, torch.Tensor):
            return neural_params.detach().cpu().numpy()
        else:
            return np.array(neural_params)
    

    # def neural_to_cfd(self, neural_params):
    #     return neural_params.cpu().numpy()