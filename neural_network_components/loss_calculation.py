import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaDesignLoss:
    def __init__(self, value_weight=1.0, policy_weight=1.0):
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss() #learned this new thing 
    
    def compute_pipeline_loss(self, policy_pred, value_pred, cfd_scores, param_improvements):
        # value_loss = self.mse_loss(value_pred.squeeze(), cfd_scores.float())
        value_loss = self.huber_loss(value_pred.squeeze(), cfd_scores.float())
        
        policy_loss = self._compute_policy_loss(policy_pred, param_improvements, cfd_scores)

        reg_loss = self.compute_regularization_loss(policy_pred)
        
        total_loss = (self.value_weight * value_loss +
                      self.policy_weight * policy_loss +
                      0.01 * reg_loss)
        
        return total_loss, {
            'total': total_loss.item(),
            'value': value_loss.item(),
            'policy': policy_loss.item(),
            'regularization': reg_loss.item()
        }
    
    # def _compute_policy_loss(self, policy_output, improvements):
    #     improvement_rewards = torch.where(improvements > 0, 1.0, -0.5)
    #     policy_loss = -torch.mean(policy_output * improvement_rewards.unsqueeze(-1))
    #     return policy_loss

    def _compute_policy_loss(self, policy_output, improvements, cfd_scores):
        improvements_normalized = torch.tanh(improvements / 10.0)
        advantages = torch.sigmoid(cfd_scores / 50.0)

        policy_loss = -torch.mean(policy_output * improvements_normalized.unsqueeze(-1) * advantages.unsqueeze(-1))

        return policy_loss
    
    def compute_regularization_loss(self, policy_output):
        l2_penalty = torch.mean(torch.square(policy_output))
        
        if policy_output.size(0) > 1:
            smoothness_penalty = torch.mean((policy_output[1:] - policy_output[:-1]) ** 2)
        else:
            smoothness_penalty = torch.tensor(0.0, device=policy_output.device)

        return l2_penalty + 0.1 * smoothness_penalty
    
    def cfd_reward_loss(self, value_predictions, actual_cfd_results):
        normalized_cfd = torch.sigmoid(actual_cfd_results / 100.0)
        return self.huber_loss(value_predictions.squeeze(), normalized_cfd)
    
    def compute_curriculum_loss(self, policy_pred, value_pred, cfd_scores, param_improvements, difficulty_factor=1.0):
        base_loss, loss_dict = self.compute_pipeline_loss(policy_pred, value_pred, cfd_scores, param_improvements)
        
        curriculum_loss = base_loss * difficulty_factor
        loss_dict['curriculum_factor'] = difficulty_factor
        
        return curriculum_loss, loss_dict
