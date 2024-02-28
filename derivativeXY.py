import numpy as np
from numpy import diff
from sympy import symbols, diff, Matrix
import torch


def numerical_derivativeXY(x, y, values, delta=1e-5):
    derivatives = []
    for symbol in y:
        original_value = values[str(symbol)]
        
        values[symbol] = original_value + delta
        x_plus_delta = [expr.evalf(subs=values) for expr in x]
        
        values[symbol] = original_value - delta
        x_minus_delta = [expr.evalf(subs=values) for expr in x]
        
        derivative_for_symbol = [(fp - fm) / (2 * delta) for fp, fm in zip(x_plus_delta, x_minus_delta)]
        derivatives.append(derivative_for_symbol)
        
        values[symbol] = original_value
    
    return derivatives


def numerical_derivativeXY_optimized(x, y, values, delta=1e-5):
    values_dict = values.copy()
    derivatives = []
    
    for symbol in y:
        original_value = values_dict[str(symbol)]
        
        values_plus_delta = values_dict.copy()
        values_minus_delta = values_dict.copy()
        values_plus_delta[symbol] = original_value + delta
        values_minus_delta[symbol] = original_value - delta
        
       
        x_plus_delta = np.array([expr.evalf(subs=values_plus_delta) for expr in x])
        x_minus_delta = np.array([expr.evalf(subs=values_minus_delta) for expr in x])
        
        derivative_for_symbol = (x_plus_delta - x_minus_delta) / (2 * delta)
        derivatives.append(derivative_for_symbol.tolist())
        
        values_dict[symbol] = original_value
    
    return derivatives

def numerical_derivativeXY_optimized_torch(x, y, values, delta=1e-5):
        
    values_dict = values.copy()
    values_dict_filtered = values.copy()
    derivatives = []

    for symbol in y:
        symbol_str = str(symbol)  # Convert SymPy symbol to string
        if symbol_str in values_dict_filtered:
            values_dict_filtered.pop(symbol_str)

    x = [expr.subs(values_dict_filtered) for expr in x]
    
    for symbol in y:
        original_value = values_dict[str(symbol)]
        
        values_plus_delta = values_dict.copy()
        values_minus_delta = values_dict.copy()
        values_plus_delta[symbol] = original_value + delta
        values_minus_delta[symbol] = original_value - delta
       
        x_plus_delta = np.array([expr.evalf(subs=values_plus_delta) for expr in x])
        x_minus_delta = np.array([expr.evalf(subs=values_minus_delta) for expr in x])
        
        derivative_for_symbol = (x_plus_delta - x_minus_delta) / (2 * delta)
        derivatives.append(derivative_for_symbol.tolist())
        
        values_dict[symbol] = original_value
    
    return derivatives


def numerical_diferentiation(x, y, values):
    values_dict = values.copy()
    values_dict_filtered = values.copy()
    derivatives = []

    for symbol in y:
        symbol_str = str(symbol)  
        if symbol_str in values_dict_filtered:
            values_dict_filtered.pop(symbol_str)

    x = [expr.subs(values_dict_filtered) for expr in x]
    
    x = Matrix(x)

    matrix = x.jacobian(y)

    matrix = [expr.subs(values) for expr in matrix]

    return matrix
