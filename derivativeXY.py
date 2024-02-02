import numpy as np

def numerical_derivativeXY(x, y, values, delta=1e-5):
    derivatives = []
    print("values", values)
    print("x", x)
    print("y", y)
    for symbol in y:
        print(f"Computing derivative with respect to: {symbol}")  # Debugging line
        original_value = values[str(symbol)]
        
        values[symbol] = original_value + delta
        x_plus_delta = [expr.evalf(subs=values) for expr in x]
        
        values[symbol] = original_value - delta
        x_minus_delta = [expr.evalf(subs=values) for expr in x]
        
        derivative_for_symbol = [(fp - fm) / (2 * delta) for fp, fm in zip(x_plus_delta, x_minus_delta)]
        derivatives.append(derivative_for_symbol)
        
        values[symbol] = original_value
    
    return derivatives