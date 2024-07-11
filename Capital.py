
import random

class Capital:
    def __init__(self, sector, state, c_s_min, c_s_max, c_t_min, c_t_max, economy):
        self.sector=sector
        self.state=state
        self.r=economy.r[sector][state]
        self.move_sector_cost=random.uniform(c_s_min,c_s_max)
        self.move_state_cost=random.uniform(c_t_min,c_t_max)
        self.to_move=False
        self.desired_sector=sector
        self.desired_state=state

    def update(self, economy):
        self.to_move=False
        self.desired_sector=self.sector
        self.desired_state=self.state
        utility=self.r

        for i in range(economy.num_sectors):
            if i!=self.sector: #if sector is different from existing one, turn delta_sector on
                delta_sector=1
            else:
                delta_sector=0
            for j in range(economy.num_states):
                if j!=self.state: #if state is different from existing one, turn delta_state on
                    delta_state=1
                else:
                    delta_state=0

                alt_utility=economy.r[i][j] - self.move_sector_cost*delta_sector - self.move_state_cost*delta_state
                if alt_utility > utility:
                    if random.choices([True, False], weights=[economy.sensitivity, 1-economy.sensitivity], k=1)[0] == True:
                        self.r=economy.r[i][j]
                        self.desired_sector = i
                        self.desired_state = j
                        utility = alt_utility
                        self.to_move=True

