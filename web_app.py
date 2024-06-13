

import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display
import numpy as np
import streamlit as st


# In[10]:


class Worker:
    def __init__(self, sector, state, c_s_min, c_s_max, c_t_min, c_t_max, economy):
        self.sector=sector
        self.state=state
        self.w=economy.w[sector][state]
        self.move_sector_cost=random.uniform(c_s_min,c_s_max)
        self.move_state_cost=random.uniform(c_t_min,c_t_max)
        self.to_move=False
        self.desired_sector=sector
        self.desired_state=state

    def update(self, economy):
        self.to_move=False
        for i in range(economy.num_sectors):
            for j in range(economy.num_states):
                if economy.w[i][j] - self.move_sector_cost*(self.sector-i)**2 - self.move_state_cost*(self.state-j)**2 > self.w:
                #^this will only work with two sectors/states!
                    self.w=economy.w[i][j]
                    self.desired_sector = i
                    self.desired_state = j
                    self.to_move=True
        


# In[11]:


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
        for i in range(economy.num_sectors):
            for j in range(economy.num_states):
                if economy.r[i][j] - self.move_sector_cost*(self.sector-i)**2 - self.move_state_cost*(self.state-j)**2 > self.r:
                #^this will only work with two sectors/states!
                    self.r=economy.r[i][j]
                    self.desired_sector = i
                    self.desired_state = j
                    self.to_move=True


# In[12]:


class Economy:
    def __init__(self, num_sectors, num_states,
                 K, L,#initial values
                 A, B, gamma, sigma, #model parameters
                 c_ls_min, c_ls_max,
                 c_lt_min, c_lt_max,
                 c_ks_min, c_ks_max,
                 c_kt_min, c_kt_max):
        
        self.num_sectors = num_sectors
        self.num_states = num_states

        # Initialize L matrix 
        self.L = np.array([[L for _ in range(num_states)] for _ in range(num_sectors)])

        # Initialize K matrix
        self.K = np.array([[K for _ in range(num_states)] for _ in range(num_sectors)])

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
        
        #Compute psi matrix
        self.psi=(self.w * self.L)/ (self.w * self.L + self.r * self.K) 
        
        # Initialize workers matrix with empty lists
        self.capital = [[[] for _ in range(num_states)] for _ in range(num_sectors)]

        # Initialize workers matrix with empty lists
        self.workers = [[[] for _ in range(num_states)] for _ in range(num_sectors)]

        # Populate the capital matrix with list of Capital type
        for i in range(num_sectors):
            for j in range(num_states):
                self.capital[i][j] = [Capital(sector=i, state=j, c_s_min=c_ks_min, c_s_max=c_ks_max, c_t_min=c_kt_min, c_t_max=c_kt_max, economy=self) for _ in range(self.K[i][j])]
        
        # Populate the workers matrix with list of Worker type
        for i in range(num_sectors):
            for j in range(num_states):
                self.workers[i][j] = [Worker(sector=i, state=j, c_s_min=c_ls_min, c_s_max=c_ls_max, c_t_min=c_lt_min, c_t_max=c_lt_max, economy=self) for _ in range(self.L[i][j])]
                
    def update_workers(self):
        #Step 1: iterate through workers to see if any desire to move.
        for i in range(self.num_sectors): #for each sector...
            for j in range(self.num_states): #... in each state...
                for k in range(self.L[i][j]): # ... for each worker...
                    self.workers[i][j][k].update(economy=self) # ... update worker.
        
        moving_workers=[]
        
        #Step 2: iterate through workers and add to list of moving_workers 
        for i in range(self.num_sectors): 
            for j in range(self.num_states): 
                for k in range(self.L[i][j]): 
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
                for k in range(self.K[i][j]): # ... for each capital unit...
                    self.capital[i][j][k].update(economy=self) # ... update capital.
        
        moving_capital=[]
        
        #Step 2: iterate through workers and add to list of moving_workers 
        for i in range(self.num_sectors): 
            for j in range(self.num_states): 
                for k in range(self.K[i][j]): 
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

        #update inputs
        self.update_workers()   
        self.update_capital()

        #update output
        self.Y = self.A * np.power(self.gamma * np.power((self.B * self.K), (self.sigma-1)/self.sigma)+(1 - self.gamma) * np.power(self.L,(self.sigma-1)/self.sigma),(self.sigma/(self.sigma-1)))
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
                    "w":self.w[i][j],
                    "r":self.r[i][j],
                    "A":self.A[i][j],
                    "B":self.B[i][j],
                    "gamma":self.gamma[i][j],
                    "sigma":self.sigma[i][j], 
                    "psi_state_sector":self.psi[i][j],
                    "psi_state":(self.L[0][j]*self.w[0][j]+self.L[1][j]*self.w[1][j])/(self.Y[0][j]+self.Y[1][j]), #this will only work with two sectors!
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
        
        return combined_df
    
    def visualize(self, sim_data):
        # List of y variables
        y_vars = ['r', 'K','w','L','Y', 'psi_state_sector', 'psi_state', 'psi_agg']
        y_var_titles=['Profit rate', 'Capital stock', 'Wage', 'Employment', 'Output', 'Sectoral labor share',
                     'State labor share', 'National labor share']
        y_var_labels=[r'$r$', r'$K$', r'$w$', r'$L$', r'$Y$', r'$\psi_{i,j}$', r'$\psi_{j}$', r'$\psi$']
        
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
        main_title = "Simulation Results"

        # Update overall layout
        fig.update_layout(
            title_text=main_title,
            height=600*len(y_vars), width=1000
        )
        
        st.plotly_chart(fig)




# ## Application

# Define the interactive widgets
num_sectors = st.slider('Num Sectors:', min_value=1, max_value=10, value=2)
num_states = st.slider('Num States:', min_value=1, max_value=10, value=2)
length = st.slider('Length:', min_value=1, max_value=100, value=30)

K = st.slider("Enter initial capital stock (K) (same for all states):", min_value=100, max_value=100000, value=10000)
L = st.slider("Enter initial employment (L) (same for all states):", min_value=100, max_value=100000, value=10000)
# Gather inputs for economic parameters using Streamlit sliders
A = st.slider("Enter Hicks-neutral coefficient (A) (same for all states):", min_value=0.1, max_value=10.0, value=1.0)
B = st.slider("Enter factor-bias coefficient (B) (same for all states):", min_value=0.1, max_value=10.0, value=1.0)
gamma = st.slider("Enter factor-importance (gamma) (same for all states):", min_value=0.0, max_value=1.0, value=0.5)
sigma = st.slider("Enter elasticity of substitution (sigma) (same for all states):", min_value=0.0, max_value=10.0, value=0.5)

c_ls_min = st.slider("Enter minimum cost for worker to switch sectors:", min_value=0.0, max_value=10.0, value=0.0)
c_ls_max = st.slider("Enter maximum cost for worker to switch sectors:", min_value=0.0, max_value=10.0, value=1.0)
c_lt_min = st.slider("Enter minimum cost for worker to switch states:", min_value=0.0, max_value=10.0, value=0.0)
c_lt_max = st.slider("Enter maximum cost for worker to switch states:", min_value=0.0, max_value=10.0, value=1.0)
c_ks_min = st.slider("Enter minimum cost for capital to switch sectors:", min_value=0.0, max_value=10.0, value=0.0)
c_ks_max = st.slider("Enter maximum cost for capital to switch sectors:", min_value=0.0, max_value=10.0, value=1.0)
c_kt_min = st.slider("Enter minimum cost for capital to switch states:", min_value=0.0, max_value=10.0, value=0.0)
c_kt_max = st.slider("Enter maximum cost for capital to switch states:", min_value=0.0, max_value=10.0, value=1.0)


shock_parameter = st.selectbox('Shock Param:', ['A', 'B', 'gamma', 'sigma'])
shock_delta = st.number_input('Shock Delta:', value=-0.5)
shock_time = st.text_input('Shock Time (comma-separated):', value='2')

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

# Button to trigger the simulation
if st.button('Run Simulation'):
    run_simulation(num_sectors, num_states, K, L, A, B, gamma, sigma, 
     c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
     length, shock_parameter, shock_delta, shock_time)    

# Visualization of the results could be added here

