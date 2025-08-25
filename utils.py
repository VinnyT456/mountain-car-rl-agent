def custom_reward(state, action):
    velocity = state[1]

    if velocity > 0 and action == 2: 
        return 1
    elif velocity < 0 and action == 0: 
        return 1
    else:
        return -1
    
def custom_action(state, episode=0):
    velocity = state[1]
    return 0 if velocity < 0 else 2 

