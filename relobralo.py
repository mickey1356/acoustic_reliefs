import torch
from typing import List
from torch.nn.functional import softmax

EPS = 1e-8

class ReLoBRaLo:
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback) is a technique for balancing multiple loss components
    during training. This method adjusts the weighting of each loss dynamically based on their relative improvement 
    over time. It occasionally uses a "lookback" mechanism to reset the weights based on the initial loss values.
    
    Attributes:
        lookback_prob (float): The probability of using the lookback mechanism, which resets the weights using the 
                               initial loss values instead of the previous ones.
        decay_rate (float): The decay rate for smoothing the weights during updates. Higher values result in slower
                            changes to the weights.
        softmax_temp (float): The temperature parameter for the softmax function. Lower values make the weight 
                              distribution sharper.
        initial_losses (List[torch.Tensor]): The loss values at the start of training, used for lookback comparisons.
        previous_losses (List[torch.Tensor]): The loss values from the previous iteration, used for standard weight 
                                              updates.
        weights (torch.Tensor): The current weights applied to the losses, dynamically adjusted during training.
    """
    
    def __init__(self, lookback_prob=0.001, decay_rate=0.999, softmax_temp=.1):
        """
        Initializes the ReLoBRaLo instance.
        
        Args:
            lookback_prob (float): Probability of resetting weights using initial loss values.
            decay_rate (float): Decay factor to update weights smoothly.
            softmax_temp (float): Temperature for softmax to control sharpness of weight distribution.
        """
        self.lookback_prob = lookback_prob
        self.decay_rate = decay_rate
        self.softmax_temp = softmax_temp
        self.initial_losses = None
        self.previous_losses = None
        self.weights = None
    
    def compute_ratios(self, current_losses: List[torch.Tensor], use_initial: bool = False) -> torch.Tensor:
        """
        Computes the ratio of current losses to reference losses (either previous or initial losses).
        
        Args:
            current_losses (List[torch.Tensor]): Current loss values from the current iteration.
            use_initial (bool): If True, use initial losses as the reference; otherwise, use previous losses.
        
        Returns:
            torch.Tensor: A tensor of ratios representing how the current losses compare to the reference losses.
        """
        # Choose reference losses based on whether we use the initial or previous losses
        reference_losses = self.initial_losses if use_initial else self.previous_losses
        # Compute ratios as current loss / reference loss, adding EPS to prevent division by zero
        ratios = [current.detach() / (ref + EPS) for current, ref in zip(current_losses, reference_losses)]
        return torch.stack(ratios)

    def compute_weights(self, current_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the updated weights based on the current losses and either the previous or initial losses.
        The weights are updated using a softmax distribution, and decay is applied to smooth the updates over time.
        
        Args:
            current_losses (List[torch.Tensor]): Current loss values from the current iteration.
        
        Returns:
            torch.Tensor: The updated weights for the current losses.
        """
        # Randomly decide whether to use lookback (reset weights based on initial losses)
        look_back = torch.bernoulli(torch.tensor([self.lookback_prob])).item() == 1

        # On the first pass, initialize initial and previous losses and set equal weights
        if self.initial_losses is None:
            self.initial_losses = [loss.detach() for loss in current_losses]
            self.previous_losses = [loss.detach() for loss in current_losses]
            self.weights = torch.ones(len(current_losses))  # Start with equal weights

        if look_back:
            # Use ratios between current and initial losses, and set weights directly
            ratios = self.compute_ratios(current_losses, use_initial=True)
            self.weights = softmax(ratios / self.softmax_temp, dim=0)
        else:
            # Use ratios between current and previous losses, apply decay for smooth updates
            ratios = self.compute_ratios(current_losses, use_initial=False)
            new_weights = softmax(ratios / self.softmax_temp, dim=0)
            # Smoothly update the weights using exponential decay
            self.weights = self.decay_rate * self.weights + (1 - self.decay_rate) * new_weights
        
        # Update previous losses for the next iteration
        self.previous_losses = [loss.detach() for loss in current_losses]

        return self.weights

    def __call__(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the weighted sum of losses based on the dynamically adjusted weights.
        
        Args:
            losses (List[torch.Tensor]): A list of loss values from the current iteration.
        
        Returns:
            torch.Tensor: The weighted sum of losses.
        """
        # Compute updated weights for the losses
        weights = self.compute_weights(losses)
        
        # Apply the weights to each loss and return the sum of the weighted losses
        weighted_losses = [loss * weight for loss, weight in zip(losses, weights)]
        
        return sum(weighted_losses)
