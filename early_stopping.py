import numpy as np
from typing import Optional, List

class EarlyStoppingManager:
    def __init__(self, patience: int = 100, min_delta: float = 0.01, 
                 restore_best_weights: bool = True, monitor: str = 'fitness'):
        
        # patience: no. of generations to wait for improvement
        # min_delta: min change to qualify as improvement
        # restore_best_weights: whether to restore best weights on stop
        # monitor: metric to monitor ('fitness', 'efficiency', 'constraint_compliance')

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        
        self.best_score = -np.inf
        self.wait_counter = 0
        self.stopped_early = False
        self.best_weights = None
        self.score_history = []
        
        self.stagnation_threshold = 100 # consecutive generations with same score
        self.convergence_threshold = 0.001  # population diversity threshold
        
    def should_stop(self, current_score: float, additional_metrics: Optional[dict] = None):
        
        self.score_history.append(current_score)
        
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait_counter = 0
            if self.restore_best_weights:
                self.best_weights = "current"  
        else:
            self.wait_counter += 1
        
        if self.wait_counter >= self.patience:
            self.stopped_early = True
            return True
        
        if len(self.score_history) >= self.stagnation_threshold:
            recent_scores = self.score_history[-self.stagnation_threshold:]
            if all(abs(score - recent_scores[0]) < self.min_delta for score in recent_scores):
                print(f"🛑 Early stopping: Stagnation detected")
                self.stopped_early = True
                return True
        
        if additional_metrics and 'population_diversity' in additional_metrics:
            if additional_metrics['population_diversity'] < self.convergence_threshold:
                print(f"🛑 Early stopping: Population converged")
                self.stopped_early = True
                return True
        
        return False
    
    def get_best_score(self):
        return self.best_score
    
    def reset(self):
        self.best_score = -np.inf
        self.wait_counter = 0
        self.stopped_early = False
        self.score_history = []
