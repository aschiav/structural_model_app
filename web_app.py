

import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display
import numpy as np
import streamlit as st



class Worker:
    def __init__(self, sector, state, alpha, beta, c_s_min, c_s_max, c_t_min, c_t_max, economy):
        self.sector=sector
        self.state=state
        self.w=economy.w[sector][state]
        self.move_sector_cost=random.uniform(c_s_min,c_s_max)
        self.move_state_cost=random.uniform(c_t_min,c_t_max)
        self.to_move=False
        self.desired_sector=sector
        self.desired_state=state
        self.alpha=alpha
        self.beta=beta
        self.utility=(alpha * self.w)** alpha * (beta * self.w) ** beta #initial utility set without prices
        self.utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.p[self.sector][self.state]*economy.w[self.sector][self.state], state=self.state)

    def utility_function(self, economy, alpha, beta, w, state):
        if w < 0:
            w=0
        utility =  (alpha * w / economy.p[0][state]) ** alpha * (beta * w / economy.p[1][state]) ** beta
        return utility


    def update(self, economy):
        self.to_move=False
        self.desired_sector=self.sector
        self.desired_state=self.state
        self.utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.p[self.sector][self.state]*economy.w[self.sector][self.state], state=self.state)
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

                alt_utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.p[i][j]*economy.w[i][j] - self.move_sector_cost*delta_sector - self.move_state_cost*delta_state, state=j)
                print("utility: ", self.utility)
                print("alt utility: ", alt_utility)
                if alt_utility > self.utility:
                    self.w=economy.w[i][j]
                    self.desired_sector = i
                    self.desired_state = j
                    self.utility = alt_utility
                    self.to_move=True
        


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
        utility=economy.p[self.sector][self.state]*self.r

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

                alt_utility=economy.p[i][j]*economy.r[i][j] - self.move_sector_cost*delta_sector - self.move_state_cost*delta_state
                if alt_utility > utility:
                    self.r=economy.r[i][j]
                    self.desired_sector = i
                    self.desired_state = j
                    utility = alt_utility
                    self.to_move=True



class Economy:
    def __init__(self, num_sectors, num_states,
                 alpha, beta,
                 K, L,#initial values
                 A, B, gamma, sigma, #model parameters
                 c_ls_min, c_ls_max,
                 c_lt_min, c_lt_max,
                 c_ks_min, c_ks_max,
                 c_kt_min, c_kt_max):
        
        self.num_sectors = num_sectors
        self.num_states = num_states
        self.alpha=alpha
        self.beta=beta

        # Initialize L matrix 
        self.L = np.array([[L for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize K matrix
        self.K = np.array([[K for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize A matrix
        self.A = np.array([[A for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize B matrix
        self.B = np.array([[B for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize gamma matrix
        self.gamma = np.array([[gamma for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize sigma matrix
        self.sigma = np.array([[sigma for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        
        #Compute output matrix
        self.Y = self.A * np.power(self.gamma * np.power((self.B * self.K), (self.sigma-1)/self.sigma)+(1 - self.gamma) * np.power(self.L,(self.sigma-1)/self.sigma),(self.sigma/(self.sigma-1)))
        
        #Compiute profit matrix
        self.r = self.gamma * np.power(self.A * self.B, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.K, 1/self.sigma) 

        #Compute wage matrix
        self.w = (1-self.gamma) * np.power(self.A, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.L, 1/self.sigma) 

        # Initialize price matrices
        self.p = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        #initialize manufacturing prices
        self.p[0][0] = (alpha * np.sum(self.w * self.L)/(self.Y[0][0]+self.Y[0][1]))
        self.p[0][1] = self.p[0][0]

        #initialize service prices
        self.p[1][0] = beta*(self.w[0][0] * self.L[0][0] + self.w[1][0] * self.L[1][0])/(self.Y[1][0])
        self.p[1][1] = beta*(self.w[0][1] * self.L[0][1] + self.w[1][1] * self.L[1][1])/(self.Y[1][1])

        #Compute psi matrix
        self.psi=(self.w * self.L)/ (self.w * self.L + self.r * self.K) 
        
        # Initialize workers matrix with empty lists
        self.capital = [[[] for _ in range(num_states)] for _ in range(num_sectors)]

        # Initialize workers matrix with empty lists
        self.workers = [[[] for _ in range(num_states)] for _ in range(num_sectors)]

        # Populate the capital matrix with list of Capital type
        for i in range(num_sectors):
            for j in range(num_states):
                self.capital[i][j] = [Capital(sector=i, state=j, c_s_min=c_ks_min, c_s_max=c_ks_max, c_t_min=c_kt_min, c_t_max=c_kt_max, economy=self) for _ in range(int(self.K[i][j]))]
        
        # Populate the workers matrix with list of Worker type
        for i in range(num_sectors):
            for j in range(num_states):
                self.workers[i][j] = [Worker(sector=i, state=j, alpha=alpha, beta=beta, c_s_min=c_ls_min, c_s_max=c_ls_max, c_t_min=c_lt_min, c_t_max=c_lt_max, economy=self) for _ in range(int(self.L[i][j]))]
                
    def update_workers(self):
        #Step 1: iterate through workers to see if any desire to move.
        for i in range(self.num_sectors): #for each sector...
            for j in range(self.num_states): #... in each state...
                for k in range(int(self.L[i][j])): # ... for each worker...
                    self.workers[i][j][k].update(economy=self) # ... update worker.
        
        moving_workers=[]
        
        #Step 2: iterate through workers and add to list of moving_workers 
        for i in range(self.num_sectors): 
            for j in range(self.num_states): 
                for k in range(int(self.L[i][j])): 
                    if self.workers[i][j][k].to_move: #if worker desires to move...
                        moving_workers.append(self.workers[i][j][k]) #... assign to list of moving workers

        #Step 3: iterate through moving workers...
        for worker in moving_workers:
            self.workers[worker.sector][worker.state].remove(worker) #remove from old sector
            worker.sector=worker.desired_sector
            worker.state=worker.desired_state
            worker.to_move=False
            self.workers[worker.sector][worker.state].append(worker) #add to new sector

        
        #Step 3: update employment matrix
        for i in range(self.num_sectors):
            for j in range(self.num_states):
                self.L[i][j] = len(self.workers[i][j])
                
    def update_capital(self):
        
        #Step 1: iterate through capital to see if any desire to move.
        for i in range(self.num_sectors): #for each sector...
            for j in range(self.num_states): #... in each state...
                for k in range(int(self.K[i][j])): # ... for each capital unit...
                    self.capital[i][j][k].update(economy=self) # ... update capital.
        
        moving_capital=[]
        
        #Step 2: iterate through workers and add to list of moving_workers 
        for i in range(self.num_sectors): 
            for j in range(self.num_states): 
                for k in range(int(self.K[i][j])): 
                    if self.capital[i][j][k].to_move: #if worker desires to move...
                        moving_capital.append(self.capital[i][j][k]) #... assign to list of moving workers

        #Step 3: iterate through moving capital...
        for capital in moving_capital:
            self.capital[capital.sector][capital.state].remove(capital) #remove from old sector
            capital.sector=capital.desired_sector
            capital.state=capital.desired_state
            capital.to_move=False
            self.capital[capital.sector][capital.state].append(capital) #add to new sector

        
        #Step 3: update employment matrix
        for i in range(self.num_sectors):
            for j in range(self.num_states):
                self.K[i][j] = len(self.capital[i][j])
   
    def update(self):
                
        #update profits and wages first
        self.r = self.gamma * np.power(self.A * self.B, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.K, 1/self.sigma) 
        self.w = (1-self.gamma) * np.power(self.A, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.L, 1/self.sigma) 

        # Check for negative values and reassign them to 0.01
        self.r = np.where(self.r < 0, 0.01, self.r)
        self.w = np.where(self.w < 0, 0.01, self.w)

        #update inputs
        self.update_workers()   
        self.update_capital()

        #update output
        self.Y = self.A * np.power(self.gamma * np.power((self.B * self.K), (self.sigma-1)/self.sigma)+(1 - self.gamma) * np.power(self.L,(self.sigma-1)/self.sigma),(self.sigma/(self.sigma-1)))

        self.p[0][0] = self.alpha * np.sum(self.w * self.L)/(self.Y[0][0]+self.Y[0][1])
        self.p[0][1] = self.p[0][0]
        self.p[1][0] = self.beta*(self.w[0][0] * self.L[0][0] + self.w[1][0] * self.L[1][0])/(self.Y[1][0])
        self.p[1][1] = self.beta*(self.w[0][1] * self.L[0][1] + self.w[1][1] * self.L[1][1])/(self.Y[1][1])

        # Check for negative values and reassign them to 0.01
        self.p = np.where(self.p < 0, 0.01, self.p)

        #update distribution
        self.psi=(self.w * self.L)/ (self.w * self.L + self.r * self.K) 


    def shock(self, sector, state, parameter, delta):        
        if parameter=="A":
            self.A[sector][state]=self.A[sector][state]+delta
        elif parameter=="B":
            self.B[sector][state]=self.B[sector][state]+delta
        elif parameter=="gamma":
            self.gamma[sector][state]=self.gamma[sector][state]+delta
        elif parameter=="sigma":
            self.sigma[sector][state]=self.sigma[sector][state]+delta
        else:
            raise Exception("Valid shock parameters are A, B, gamma, and sigma.")
        
        
    def get_data(self):
        dataframe=[]
        
        #sector variables
        for i in range(self.num_sectors):
            for j in range(self.num_states):
                data = {
                    "sector":i,
                    "state":j,
                    "L":self.L[i][j],
                    "K":self.K[i][j],
                    "Y":self.Y[i][j],
                    "p":self.p[i][j],
                    "K/L":self.K[i][j]/self.L[i][j],
                    "w":self.w[i][j],
                    "r":self.r[i][j],
                    "MRPL":self.w[i][j]*self.p[i][j],
                    "MRPK":self.r[i][j]*self.p[i][j],
                    "A":self.A[i][j],
                    "B":self.B[i][j],
                    "gamma":self.gamma[i][j],
                    "sigma":self.sigma[i][j], 
                    "lambda_state":self.L[i][j]/(self.L[0][j]+self.L[1][j]), #this will only work with two sectors!
                    "psi_state_sector":self.psi[i][j],
                    "psi_state":(self.L[0][j]*self.w[0][j]+self.L[1][j]*self.w[1][j])/(self.Y[0][j]+self.Y[1][j]), #this will only work with two sectors!
                    "lambda_agg":(self.L[i][0]+self.L[i][1])/np.sum(self.L), #this will only work with two sectors!
                    "psi_agg":np.sum(self.L*self.w)/np.sum(self.Y) 
                }
                df=pd.DataFrame(data, index=[0])
                dataframe.append(df)
   
        
        # Concatenate all dataframes into a single dataframe
        final_df = pd.concat(dataframe, ignore_index=True)
        return final_df
        


    def display(self):
        result = []
        for i in range(self.num_sectors):
            for j in range(self.num_states):
                result.append(f"Sector {i+1}, State {j+1}:\nWage: {self.w[i][j]}\nEmployment: {self.L[i][j]}\n")
        return "\n".join(result)


class Simulation:
    def __init__(self, num_sectors, num_states, length, #simulation parameters
                alpha=0.5, beta=0.5,
                 K=10000, L=10000,#initial values
                 A=1, B=1, gamma=0.5, sigma=0.5, #model parameters
                 c_ls_min=0, c_ls_max=1, #model hyper-parameters
                 c_lt_min=0, c_lt_max=1,
                 c_ks_min=0, c_ks_max=1,
                 c_kt_min=0, c_kt_max=1,
                 shock_parameter="A", shock_delta=0, shock_time=0): #shock parameters
        
        self.num_sectors=num_sectors
        self.num_states=num_states
        self.length=length
        self.shock_parameter=shock_parameter
        self.shock_delta=shock_delta
        self.shock_time=shock_time
        self.economy=Economy(num_states=num_states, num_sectors=num_sectors,
                             alpha=alpha, beta=beta,
                             K=K, L=L,#initial values
                             A=A, B=B, gamma=gamma, sigma=sigma, #model parameters
                             c_ls_min=c_ls_min, c_ls_max=c_ls_max,
                             c_lt_min=c_lt_min, c_lt_max=c_lt_max,
                             c_ks_min=c_ks_min, c_ks_max=c_ks_max,
                             c_kt_min=c_kt_min, c_kt_max=c_kt_max,
                            )
        
                
    def run(self):
        simulation_data = []

        for t in range(0,self.length):
            if str(t) in self.shock_time:
                self.economy.shock(sector=0, state=0, parameter=self.shock_parameter, delta=self.shock_delta)
            self.economy.update()
            df=self.economy.get_data()
            df.insert(0, 't', t)  
            simulation_data.append(df)
        
        # Concatenate all the DataFrames in the state_data list into a single DataFrame
        combined_df = pd.concat(simulation_data, ignore_index=True)
        
        #transform state values from numeric to letters (A and B)
        combined_df["state"]=combined_df["state"].replace({0:'A',1:'B'})
        
        #transform sector values from numeric to letters (m and s)
        combined_df["sector"]=combined_df["sector"].replace({0:'m',1:'s'})

        return combined_df
    
    def visualize(self, sim_data):
        # List of y variables
        y_vars = ['r', 'MRPK', 'K','w', 'MRPL', 'L','Y', 'p', 'K/L', 'lambda_state', 'lambda_agg', 'psi_state_sector', 'psi_state','psi_agg']
        y_var_titles=['Profit rate', 'Marginal Revenue Product of Capital', 'Capital stock', 'Wage', 'Marginal Revenue Product of Labor', 'Employment', 'Output', 'Prices', 'Capital-labor ratio', 'State-sector employment share', 'Sector employment share (National)', 'Sectoral labor share',
                     'State labor share', 'National labor share']
        y_var_labels=[r'$r$', r'$MRP_{K}$', r'$K$', r'$w$', r'$MRP_{L}$', r'$L$', r'$Y$', r'$p$', r'$K/L$', r'$\lambda_{ij}$', r'$\lambda_{i}$', r'$\psi_{i,j}$', r'$\psi_{j}$', r'$\psi$']
        
        # Create a subplot grid
        fig = make_subplots(rows=len(y_vars), cols=1, subplot_titles=y_var_titles)

        # Add each plot to the grid
        for i, y_var in enumerate(y_vars):
            # Use Plotly Express to create the line plot
            trace = px.line(sim_data, 
                            x='t',
                            y=y_var,
                            color='state',
                            line_dash='sector',
                            markers=False, 
                            title=y_var_titles[i], 
                            labels={'t': 't', y_var: y_var, 'state': 'State'}
                           )
            

            # Add the trace to the subplot grid
            for data in trace.data:
                fig.add_trace(data, row=i+1, col=1)

            # Update axes labels and titles
            fig.update_xaxes(title_text='', row=i+1, col=1)
            fig.update_yaxes(title_text='', row=i+1, col=1)

        
       # Use combined raw and formatted string for the title
        main_title = ""

        # Update overall layout
        fig.update_layout(
            title_text=main_title,
            height=600*len(y_vars), width=1000
        )
        
        st.plotly_chart(fig)




# ## Application
# Function to run the simulation with given parameters
def run_simulation(num_sectors, num_states, K, L, A, B, gamma, sigma, 
        c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
        length, shock_parameter, shock_delta, shock_time):
    # Convert shock_time string to a list of integers
    shock_time_list = list(map(int, shock_time.split(',')))

    # Run the simulation
    sim=Simulation(num_sectors=num_sectors, 
                num_states=num_states,
                K=K,
                L=L,
                A=A,
                B=B,
                gamma=gamma,
                sigma=sigma,
                c_ls_min=c_ls_min,
                c_ls_max=c_ls_max,
                c_lt_min=c_lt_min,
                c_lt_max=c_lt_max,
                c_ks_min=c_ks_min,
                c_ks_max=c_ks_max,
                c_kt_min=c_kt_min,
                c_kt_max=c_kt_max,
                length=length, 
                shock_parameter=shock_parameter, 
                shock_delta=shock_delta, 
                shock_time=shock_time)
    
    sim_data=sim.run()
    sim.visualize(sim_data)

# Define the 'Documentation' page
def page_documentation():
    st.markdown(r"""
    # Two-Sector Two-State Model

    The model consists of two states. Each state has two sectors. State-sectors begin with an initial endowment of capital ($K$) and labor ($L$). Sectors produce according to a constant elasticity of substitution (CES) production function:

    $$
    Y_{ij}=A_{ij}\Big[\gamma_{ij}(\tilde{B}_{ij}K_{ij})^{\frac{\sigma_{ij}-1}{\sigma_{ij}}}+(1-\gamma_{ij})L_{ij}^{\frac{\sigma_{ij}-1}{\sigma_{ij}}}\Big]^{\frac{\sigma_{ij}}{\sigma_{ij} -1}}
    $$

    Assuming perfect competition, factor rewards for capital and labor are:

    $$
    R_{ij}=p_{ij}\cdot\gamma_{ij}(A_{ij}\tilde{B}_{ij})^{\frac{\sigma_{ij}-1}{\sigma_{ij}}}\Big(\frac{Y_{ij}}{K_{ij}}\Big)^{\frac{1}{\sigma_{ij}}}
    $$

    $$
    w_{ij}=p_{ij}\cdot(1-\gamma_{ij})A_{ij}^{{\frac{\sigma_{ij}-1}{\sigma_{ij}}}}\Big(\frac{Y_{ij}}{L_{ij}}\Big)^{\frac{1}{\sigma_{ij}}}
    $$

    where $p_{ij}$ is the price of good $i$ in state $j$. We assume that the price of the tradable good is equalized automatically across both states, such that $p_{mA}=p_{mB}$.
                
    ## Workers
    Consider two states, ($A$) and ($B$), and two sectors, manufacturing ($m$) and services ($s$). Let ($w_{mA}$) and ($w_{sA}$) be the wages in state ($A$) for the manufacturing and service sectors, respectively, and ($w_{mB}$) and ($w_{sB}$) be the corresponding wages in state ($B$). Transitioning between sectors involves a cost ($c_S$), and transitioning between states involves a cost ($c_T$).
    
    Workers earn their income solely from wages, and their utility is represented by a Cobb-Douglas function:

    $$U(C_m, C_s) = C_m^\alpha \cdot C_s^\beta$$
                
    where ($C_m$) and ($C_s$) are the consumption levels of the manufacturing good and the service good, respectively, and ($\alpha + \beta = 1$).
                
    ##### Consumption
                
    The budget constraint for a worker is:
                
    $$p_m \cdot C_m + p_s \cdot C_s \leq w_{ij}$$
                
    where $p_m$ and $p_s$ are the prices of the manufacturing and service goods, respectively, and $w_{ij}$ is the wage in sector $i$ (either $m$ or $s$) and state $j$ (either $A$ or $B$).

    Given the Cobb-Douglas utility function, the optimal consumption bundle is:
                
    $$C_m = \alpha \frac{w_{ij}}{p_m(\alpha+\beta)}, \quad C_s = \beta \frac{w_{ij}}{p_s(\alpha+\beta)}$$
                
    Under homothetic preferences, $\alpha+\beta=1$. The maximized utility for a worker in sector \(i\) and state \(j\) is:

    $$
    U_{ij} = \left( \alpha \frac{w_{ij}}{p_m} \right)^\alpha \cdot \left( \beta \frac{w_{ij}}{p_s} \right)^\beta
    $$
    
    ##### Switching
    A worker starts assigned to a state-sector. The worker can choose to switch sectors and/or states each iteration of the model. If she transitions sectors, she incurs a one-time cost of $c_{L,S}$. Likewise, a transition of states incurs a cost $c_{L,T}$.

    When transitioning states, the worker faces a new price for the non-tradable good. If the worker starts in sector $m$ in state $A$, her utility is:

    $$
        U_{mA}=\left( \alpha \frac{w_{mA}}{p_m} \right)^\alpha \cdot \left( \beta \frac{w_{mA}}{p_{sA}} \right)^\beta
    $$

    By switching sectors, her utility would be:

    $$
        U_{sA}=\left( \alpha \frac{w_{sA}-c_{L,S}}{p_m} \right)^\alpha \cdot \left( \beta \frac{w_{sA}-c_{L,S}}{p_{sA}} \right)^\beta
    $$

    By switching states, her utility would be:

    $$
        U_{mB}=\left( \alpha \frac{w_{mB}-c_{L,T}}{p_m} \right)^\alpha \cdot \left( \beta \frac{w_{mB}-c_{L,T}}{p_{sB}} \right)^\beta
    $$

    Finally, by switching *both* sectors and states, her utility would be:

    $$
        U_{sB}=\left( \alpha \frac{w_{sB}-c_{L,S}-c_{L,T}}{p_m} \right)^\alpha \cdot \left( \beta \frac{w_{sB}-c_{L,S}-c_{L,T}}{p_{sB}} \right)^\beta
    $$

    The worker will thus select the sector and state that maximizes temporal utility. A discounting mechanism could be added without loss of generality. 
                    
    ## Capital
    Each unit of capital is controlled by a single profit maximizing capitalist. There costs $c_{K,S}$ and $c_{K,T}$ associated with moving capital between states and sectors. These costs are drawn from a normal distribution to introduce heterogeneity into the model. The capitalist will switch from sector $m$ to $s$ if:

    $$
        R_{mA}-c_{K,S}>R_{sA}
    $$

    and will switch from state $A$ to $B$ if:

    $$
        R_{mB}-c_{K,T}>R_{mA}
    $$

    ## Output Prices
    For the tradable good, consumer demand is met by production in both states:
    $$
        Y_{mA}+Y_{mB}=L_{mA}\alpha\frac{\omega_{mA}}{p_m}+L_{sA}\alpha\frac{\omega_{sA}}{p_m}+L_{mB}\alpha\frac{\omega_{mB}}{p_m}+L_{sB}\alpha\frac{\omega_{sB}}{p_m}
    $$

    Solving for $p_{m}$ yields:


    $$
        p_{m}=\alpha\frac{w_{mA}L_{mA}+w_{sA}L_{sA}+w_{mB}L_{mB}+w_{sB}L_{sB}}{Y_{mA}+Y_{mB}}
    $$

    For the non-tradable good, consumer demand is met by production within each state:

    $$
        Y_{sA}=L_{mA}\alpha\frac{\omega_{mA}}{p_{sA}}+L_{sA}\alpha\frac{\omega_{sA}}{p_{sA}}
    $$

    $$
        Y_{sB}=L_{mB}\alpha\frac{\omega_{mB}}{p_{sB}}+L_{sB}\alpha\frac{\omega_{sB}}{p_{sB}}
    $$

    Such that:

    $$
        p_{sA}=\beta\frac{w_{mA}L_{mA}+w_{sA}L_{sA}}{Y_{sA}}
    $$

    $$
        p_{sB}=\beta\frac{w_{mB}L_{mB}+w_{sB}L_{sB}}{Y_{sB}}
    $$

    Factors are fully utilized, such that output and thus prices are determined.
                
    """)

# Define the 'Simulation' page
def page_simulation():
    st.markdown(r"""
    # Two-Sector Two-State Model

    All state-sectors begin in identical states. Adjust the sliders to set initial conditions and economic parameters. Choose the parameter and magnitude of the shock you would like to simulate. All shocks are applied to sector $m$ in state $A$.

    Click on the 'Run Simulation' button at the bottom to start the simulation.
    
    #### Model Parameters

    """)
    
    # Define the interactive widgets
    num_sectors = 2
    num_states = 2

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        K = st.slider(r"Initial capital stock ($K$)", min_value=10, max_value=20000, value=10000)
        L = st.slider(r"Initial employment ($L$)", min_value=10, max_value=20000, value=10000)

    with col2:
        A = st.slider(r"Hicks-neutral coefficient ($A$)", min_value=0.1, max_value=2.0, value=1.0)
        B = st.slider(r"Factor-bias coefficient ($B$)", min_value=0.1, max_value=2.0, value=1.0)
        gamma = st.slider(r"Factor-importance ($\gamma$)", min_value=0.0, max_value=1.0, value=0.5)
        sigma = st.slider(r"Elasticity of substitution ($\sigma$)", min_value=0.0, max_value=2.0, value=0.5)

    with col3:
        c_ls_min, c_ls_max = st.slider(
            r'$c_{L,S}\sim U(c_{L,S}^{\min}, c_{L,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 1.0)
        )
        c_lt_min, c_lt_max = st.slider(
            r'$c_{L,T}\sim U(c_{L,T}^{\min}, c_{L,T}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 1.0)
        )
        c_ks_min, c_ks_max = st.slider(
            r'$c_{K,S}\sim U(c_{K,S}^{\min}, c_{K,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 1.0)
        )
        c_kt_min, c_kt_max = st.slider(
            r'$c_{K,T}\sim U(c_{K,T}^{\min}, c_{K,T}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 1.0)
        )

    st.markdown(r"""
    
    #### Shock parameters

    """)

    #Additional parameters
    length = int(st.text_input('Length', value=10))
    shock_parameter = st.selectbox('Shock parameters:', ['B', 'A', 'gamma', 'sigma'])
    shock_delta = st.number_input('Shock Delta:', value=-0.5)
    shock_time = st.text_input('Shock Time (comma-separated):', value='2')


    # Button to trigger the simulation
    if st.button('Run Simulation'):
        st.markdown(r"""#### Results""") 
        run_simulation(num_sectors, num_states, K, L, A, B, gamma, sigma, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, shock_parameter, shock_delta, shock_time)   

       

page = "Simulation"

# Define page navigation using buttons
if st.sidebar.button("Simulation"):
    page = "Simulation"
if st.sidebar.button("Documentation"):
    page = "Documentation"



# Page routing
if page == "Documentation":
    page_documentation()
elif page == "Simulation":
    page_simulation()


