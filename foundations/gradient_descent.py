class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        x = init
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        for i in range(iterations):
            x = x - learning_rate * 2 * x 
        # Round final answer to 5 decimal places
        return round(x, 5)
