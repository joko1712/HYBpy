def calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state):
    stateFinal = []
    
    for i, value in enumerate(state.values()):
        new_value = value + h * (k1_state[i] / 6 + k2_state[i] / 3 + k3_state[i] / 3 + k4_state[i] / 6)
        stateFinal.append(new_value)
    
    return stateFinal

state = {'species1': 10, 'species2': 20, 'species3': 30}
h = 0.1
k1_state = [1, 2, 3]
k2_state = [4, 5, 6]
k3_state = [7, 8, 9]
k4_state = [10, 11, 12]

stateFinal = calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state)
print(stateFinal)

