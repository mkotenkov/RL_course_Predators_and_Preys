def calculate_state_value(processed_state_i, bonus_count,
                          w_step, w_kill_prey, w_kill_enemy, w_kill_bonus,
                          gamma_to_reduce_bonus_weight):
    stones_mask, preys_mask, teammates_mask, enemies_mask, bonuses_mask = processed_state[
        0, ...], processed_state[1, ...], processed_state[2, ...], 
        processed_state[3, ...], processed_state[ 4, ...]
        