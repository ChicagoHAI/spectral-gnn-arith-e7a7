# This experiment validates the approximation capabilities and numerical stability of a simplified neural network architecture for basic arithmetic operations. We'll test addition and multiplication on random inputs and verify error bounds and condition numbers.
# Verified: No (simulated)

import numpy as np

# Simple spectral filter implementation
def spectral_filter(x, k):
    return np.tanh(k * x) / (k + 1)

# Network approximation for addition
def network_add(x1, x2, K=5):
    result = 0
    for k in range(1, K+1):
        result += spectral_filter(x1, k) + spectral_filter(x2, k)
    return result

# Network approximation for multiplication
def network_multiply(x1, x2, K=5):
    result = 0
    for k in range(1, K+1):
        result += spectral_filter(x1, k) * spectral_filter(x2, k)
    return result

# Condition number estimation
def estimate_condition_number(func, x, delta=1e-6):
    f_x = func(*x)
    grads = []
    for i in range(len(x)):
        x_perturbed = list(x)
        x_perturbed[i] += delta
        f_x_perturbed = func(*x_perturbed)
        grad = (f_x_perturbed - f_x) / delta
        grads.append(abs(grad))
    return max(grads) * np.linalg.norm(x) / (abs(f_x) + 1e-10)

# Test cases
np.random.seed(42)
n_tests = 1000
K_values = [3, 5, 10]

print("VALIDATION EXPERIMENTS")
print("=====================")

for K in K_values:
    print(f"\nTesting with K={K} filters:")
    
    add_errors = []
    mult_errors = []
    add_conditions = []
    mult_conditions = []
    
    for _ in range(n_tests):
        x1, x2 = np.random.uniform(-10, 10, 2)
        
        # Test addition
        true_add = x1 + x2
        pred_add = network_add(x1, x2, K)
        add_errors.append(abs(true_add - pred_add))
        add_conditions.append(estimate_condition_number(network_add, (x1, x2), K))
        
        # Test multiplication
        true_mult = x1 * x2
        pred_mult = network_multiply(x1, x2, K)
        mult_errors.append(abs(true_mult - pred_mult))
        mult_conditions.append(estimate_condition_number(network_multiply, (x1, x2), K))
    
    print(f"Addition - Mean Error: {np.mean(add_errors):.6f}")
    print(f"Addition - Max Error: {np.max(add_errors):.6f}")
    print(f"Addition - Mean Condition Number: {np.mean(add_conditions):.2f}")
    print(f"Multiplication - Mean Error: {np.mean(mult_errors):.6f}")
    print(f"Multiplication - Max Error: {np.max(mult_errors):.6f}")
    print(f"Multiplication - Mean Condition Number: {np.mean(mult_conditions):.2f}")
