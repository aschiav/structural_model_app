
import random
import pandas as pd
import matplotlib.pyplot as plte
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
        self.utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.w[self.sector][self.state], state=self.state)

    def utility_function(self, economy, alpha, beta, w, state):
        if w < 0:
            w=0
        utility =  (alpha * w / economy.p[0][state]) ** alpha * (beta * w / economy.p[1][state]) ** beta
        return utility


    def update(self, economy):
        self.to_move=False
        self.desired_sector=self.sector
        self.desired_state=self.state
        self.utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.w[self.sector][self.state], state=self.state)
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

                alt_utility=self.utility_function(economy=economy, alpha=self.alpha, beta=self.beta, w=economy.w[i][j] - self.move_sector_cost*delta_sector - self.move_state_cost*delta_state, state=j)
                if alt_utility > self.utility:
                    if random.choices([True, False], weights=[economy.sensitivity, 1-economy.sensitivity], k=1)[0] == True:
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

                alt_utility=economy.r[i][j] - self.move_sector_cost*delta_sector - self.move_state_cost*delta_state
                if alt_utility > utility:
                    if random.choices([True, False], weights=[economy.sensitivity, 1-economy.sensitivity], k=1)[0] == True:
                        self.r=economy.r[i][j]
                        self.desired_sector = i
                        self.desired_state = j
                        utility = alt_utility
                        self.to_move=True


class Economy:
    def __init__(self, num_sectors, num_states,
                 alpha, beta,
                 K_L_m, K_L_s,#initial values
                 A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, #model parameters
                 c_ls_min, c_ls_max,
                 c_lt_min, c_lt_max,
                 c_ks_min, c_ks_max,
                 c_kt_min, c_kt_max, sensitivity):
        
        self.sensitivity=sensitivity
        self.num_sectors = num_sectors
        self.num_states = num_states
        self.alpha=alpha
        self.beta=beta


        self.K_L = np.array([[K_L_m for _ in range(num_states)] if i == 0 else [K_L_s for _ in range(num_states)] for i in range(num_sectors)], dtype=float)

        # Initialize L and K matrices 
        self.L = np.array([[10000 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.K = self.L * self.K_L

        # Initialize A matrix
        self.A = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.A[0][0]=A_m
        self.A[0][1]=A_m
        self.A[1][0]=A_s
        self.A[1][1]=A_s


        # Initialize B=1 matrix
        self.B = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize gamma matrix
        self.gamma = np.array([[0.5 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.gamma[0][0]=gamma_m
        self.gamma[0][1]=gamma_m
        self.gamma[1][0]=gamma_s
        self.gamma[1][1]=gamma_s

        # Initialize sigma matrix
        self.sigma = np.array([[sigma_m for _ in range(num_states)] if i == 0 else [sigma_s for _ in range(num_states)] for i in range(num_sectors)], dtype=float)
        
        #Compute output matrix
        self.Y = self.A * np.power(self.gamma * np.power((self.B * self.K), (self.sigma-1)/self.sigma)+(1 - self.gamma) * np.power(self.L,(self.sigma-1)/self.sigma),(self.sigma/(self.sigma-1)))

        #compute marginal product matrixes
        self.MPL=(1-self.gamma) * np.power(self.A, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.L, 1/self.sigma) 
        self.MPK = self.gamma * np.power(self.A * self.B, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.K, 1/self.sigma) 

        # Initialize price matrices
        self.p = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        
        #ACetal prices
        term1=(self.gamma[0][0]*self.A[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0])))/(self.gamma[1][0]*self.A[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0])))
        term2_numerator=(self.gamma[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0]))+(1-self.gamma[0][0])*self.K_L[0][0]**((1-self.sigma[0][0])/self.sigma[0][0]))**(1/(self.sigma[0][0]-1))
        term2_denominator=(self.gamma[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0]))+(1-self.gamma[1][0])*self.K_L[1][0]**((1-self.sigma[1][0])/self.sigma[1][0]))**(1/(self.sigma[1][0]-1))
        self.p[1][0]=term1*(term2_numerator/term2_denominator)

        term1 = (self.gamma[0][1] * self.A[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1]))) / (self.gamma[1][1] * self.A[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])))
        term2_numerator = (self.gamma[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1])) + (1 - self.gamma[0][1]) * self.K_L[0][1] ** ((1 - self.sigma[0][1]) / self.sigma[0][1])) ** (1 / (self.sigma[0][1] - 1))
        term2_denominator = (self.gamma[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])) + (1 - self.gamma[1][1]) * self.K_L[1][1] ** ((1 - self.sigma[1][1]) / self.sigma[1][1])) ** (1 / (self.sigma[1][1] - 1))
        self.p[1][1] = term1 * (term2_numerator / term2_denominator)

        #Compute profit matrix
        self.r = self.p * self.MPK

        #Compute wage matrix
        self.w = self.p * self.MPL

        #Compute psi matrix
        self.psi=(self.w * self.L)/ (self.p*self.Y) 
        
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
        

        #import warnings
        #warnings.filterwarnings("ignore")

        #print("Y:")
        #print(self.Y)
        #print("MPL:")
        #print(self.MPL)
        #print("MPK:")
        #print(self.MPK)
        #print("r:")
        #print(self.r)
        #print("w:")
        #print(self.w)
        #print("L:")
        #print(self.L)
        #print("K:")
        #print(self.K)
        #print("p:")
        #print(self.p)
    
        #shock happens here 

        #update output
        self.Y = self.A * np.power(self.gamma * np.power((self.B * self.K), ((self.sigma-1)/self.sigma))+(1 - self.gamma) * np.power(self.L,((self.sigma-1)/self.sigma)),(self.sigma/(self.sigma-1)))
    
        #update marginal product matrixes
        self.MPL=(1-self.gamma) * np.power(self.A, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.L, 1/self.sigma) 
        self.MPK = self.gamma * np.power(self.A * self.B, (self.sigma - 1)/self.sigma) * np.power(self.Y/self.K, 1/self.sigma) 

        #update profits and wages
        self.r = self.p * self.MPK
        self.w = self.p * self.MPL
    
        #update prices
        #self.p[1][0]= (self.beta*(self.w[0][0]*self.L[0][0]+self.w[1][0]*self.L[1][0])/(self.Y[1][0]))/self.alpha*np.sum(self.w*self.L)/(self.Y[0][0]+self.Y[0][1])
        #self.p[1][1]= (self.beta*(self.w[0][1]*self.L[0][1]+self.w[1][1]*self.L[1][1])/(self.Y[1][1]))/self.alpha*np.sum(self.w*self.L)/(self.Y[0][0]+self.Y[0][1])

        #ACetal prices
        term1=(self.gamma[0][0]*self.A[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0])))/(self.gamma[1][0]*self.A[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0])))
        term2_numerator=(self.gamma[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0]))+(1-self.gamma[0][0])*self.K_L[0][0]**((1-self.sigma[0][0])/self.sigma[0][0]))**(1/(self.sigma[0][0]-1))
        term2_denominator=(self.gamma[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0]))+(1-self.gamma[1][0])*self.K_L[1][0]**((1-self.sigma[1][0])/self.sigma[1][0]))**(1/(self.sigma[1][0]-1))
        self.p[1][0]=term1*(term2_numerator/term2_denominator)

        term1 = (self.gamma[0][1] * self.A[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1]))) / (self.gamma[1][1] * self.A[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])))
        term2_numerator = (self.gamma[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1])) + (1 - self.gamma[0][1]) * self.K_L[0][1] ** ((1 - self.sigma[0][1]) / self.sigma[0][1])) ** (1 / (self.sigma[0][1] - 1))
        term2_denominator = (self.gamma[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])) + (1 - self.gamma[1][1]) * self.K_L[1][1] ** ((1 - self.sigma[1][1]) / self.sigma[1][1])) ** (1 / (self.sigma[1][1] - 1))
        self.p[1][1] = term1 * (term2_numerator / term2_denominator)


        
        #update inputs
        self.update_workers()   
        self.update_capital()

        #update capital labor ratios
        self.K_L=self.K/self.L

        #update distribution
        self.psi=(self.w * self.L)/ (self.p*self.Y) 

 


    def shock(self, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb):

        self.A[0][0]=self.A[0][0]*(1+g_A_ma/100)
        self.A[0][1]=self.A[0][1]*(1+g_A_mb/100)        
        self.A[1][0]=self.A[1][0]*(1+g_A_sa/100)
        self.A[1][1]=self.A[1][1]*(1+g_A_sb/100)

        self.B[0][0]=self.B[0][0]*(1+g_Bt_ma/100)
        self.B[0][1]=self.B[0][1]*(1+g_Bt_mb/100)        
        self.B[1][0]=self.B[1][0]*(1+g_Bt_sa/100)
        self.B[1][1]=self.B[1][1]*(1+g_Bt_sb/100)
        
        
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
                    "MPL":self.MPL[i][j],
                    "MPK":self.MPK[i][j],
                    "p":self.p[i][j],
                    "K/L":self.K[i][j]/self.L[i][j],
                    "w":self.w[i][j],
                    "r":self.r[i][j],
                    "lambda_state":self.L[i][j]/(self.L[0][j]+self.L[1][j]), #this will only work with two sectors!
                    "psi_state_sector":self.psi[i][j],
                    "psi_state":(self.L[0][j]*self.w[0][j]+self.L[1][j]*self.w[1][j])/(self.p[0][j]*self.Y[0][j]+self.p[1][j]*self.Y[1][j]), #this will only work with two sectors!
                    "lambda_agg":(self.L[i][0]+self.L[i][1])/np.sum(self.L), #this will only work with two sectors!
                    "psi_agg":np.sum(self.L*self.w)/(np.sum(self.p*self.Y) )
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
                 K_L_m=1, K_L_s=1,#initial values
                 A_m=1, A_s=1, B=1, gamma_m=0.5, gamma_s=0.5, sigma_m=0.5, sigma_s=0.5, #model parameters
                 c_ls_min=0, c_ls_max=1, #model hyper-parameters
                 c_lt_min=0, c_lt_max=1,
                 c_ks_min=0, c_ks_max=1,
                 c_kt_min=0, c_kt_max=1,
                 g_A_ma=0, g_A_mb=0, g_A_sa=0, g_A_sb=0, g_Bt_ma=0, g_Bt_mb=0, g_Bt_sa=0, g_Bt_sb=0,
                 sensitivity=0): 
        
        self.num_sectors=num_sectors
        self.num_states=num_states
        self.length=length
        self.g_A_ma=g_A_ma 
        self.g_A_mb=g_A_mb
        self.g_A_sa=g_A_sa
        self.g_A_sb=g_A_sb 
        self.g_Bt_ma=g_Bt_ma 
        self.g_Bt_mb=g_Bt_mb 
        self.g_Bt_sa=g_Bt_sa 
        self.g_Bt_sb=g_Bt_sb 
        self.economy=Economy(num_states=num_states, num_sectors=num_sectors,
                             alpha=alpha, beta=beta,
                             K_L_m=K_L_m, K_L_s=K_L_s,#initial values
                             A_m=A_m, A_s=A_s, gamma_m=gamma_m, gamma_s=gamma_s, sigma_m=sigma_m, sigma_s=sigma_s, #model parameters
                             c_ls_min=c_ls_min, c_ls_max=c_ls_max,
                             c_lt_min=c_lt_min, c_lt_max=c_lt_max,
                             c_ks_min=c_ks_min, c_ks_max=c_ks_max,
                             c_kt_min=c_kt_min, c_kt_max=c_kt_max,
                             sensitivity=sensitivity
                            )
        
                
    def run(self):
        simulation_data = []

        for t in range(0,self.length):
            self.economy.shock(g_A_ma=self.g_A_ma, g_A_mb=self.g_A_mb, g_A_sa=self.g_A_sa, g_A_sb=self.g_A_sb, g_Bt_ma=self.g_Bt_ma, g_Bt_mb=self.g_Bt_mb, g_Bt_sa=self.g_Bt_sa, g_Bt_sb=self.g_Bt_sb)
            self.economy.update()
            df=self.economy.get_data()
            df.insert(0, 't', t)  
            simulation_data.append(df)
        
        # Concatenate all the DataFrames in the state_data list into a single DataFrame
        combined_df = pd.concat(simulation_data, ignore_index=True)
        
        # Remove the first row where t == 0
        combined_df = combined_df[combined_df['t'] != 0]
    
        #transform state values from numeric to letters (A and B)
        combined_df["state"]=combined_df["state"].replace({0:'A',1:'B'})
        
        #transform sector values from numeric to letters (m and s)
        combined_df["sector"]=combined_df["sector"].replace({0:'m',1:'s'})

        return combined_df
    
    def visualize(self, sim_data):
        # List of y variables
        y_vars = ['r','w', 'MPL', 'MPK','Y', 'p', 'K/L', 'lambda_state', 'lambda_agg', 'psi_state_sector', 'psi_state','psi_agg']
        y_var_titles=['Profit rate (MRPK)', 'Wage (MRPL)', 'Marginal Prod. (L)', 'Marginal Prod. (K)', 'Output', 'Prices', 'Capital-labor ratio', 'State-sector employment share', 'Sector employment share (National)', 'Sectoral labor share',
                     'State labor share', 'National labor share']
        
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




### Application ####

# Function to run the simulation with given parameters
def run_simulation(num_sectors, num_states, K_L_m, K_L_s, A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb, sensitivity):

    # Run the simulation
    sim=Simulation(num_sectors=num_sectors, 
                num_states=num_states,
                K_L_m=K_L_m,
                K_L_s=K_L_s,
                A_m=A_m,
                A_s=A_s,
                gamma_m=gamma_m,
                gamma_s=gamma_s,
                sigma_m=sigma_m,
                sigma_s=sigma_s,
                c_ls_min=c_ls_min,
                c_ls_max=c_ls_max,
                c_lt_min=c_lt_min,
                c_lt_max=c_lt_max,
                c_ks_min=c_ks_min,
                c_ks_max=c_ks_max,
                c_kt_min=c_kt_min,
                c_kt_max=c_kt_max,
                length=length, 
                g_A_ma=g_A_ma, 
                g_A_mb=g_A_mb,
                g_A_sa=g_A_sa,
                g_A_sb=g_A_sb, 
                g_Bt_ma=g_Bt_ma, 
                g_Bt_mb=g_Bt_mb, 
                g_Bt_sa=g_Bt_sa, 
                g_Bt_sb=g_Bt_sb, 
                sensitivity=sensitivity
                )
    
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
    We let the tradable good be the numeraire, such that $p_{sA}=P_{sA}/P_{m}$ and $p_{sB}=P_{sB}/P_{m}$. We follow Alvarez-Quadrado et al. (2018) for determining the relative price of services:


    $$
    p_{sA}=\frac{\gamma_{mA}}{\gamma_{sA}}\frac{A_{mA}\tilde{B}_{mA}^{\frac{\sigma_{m}-1}{\sigma_{m}}}}{A_{sA}\tilde{B}_{sA}^{\frac{\sigma_{s}-1}{\sigma_{s}}}}\frac{\Big[\gamma_{mA}\tilde{B}^{\frac{\sigma_{m}-1}{\sigma_{m}}}+(1-\gamma_{mA})\Big(\frac{K_{mA}}{L_{mA}}\Big)^{\frac{1-\sigma_{m}}{\sigma_{m}}}\Big]^{\frac{1}{\sigma_{m}-1}}}{\Big[\gamma_{sA}\tilde{B}^{\frac{\sigma_{s}-1}{\sigma_{s}}}+(1-\gamma_{sA})\Big(\frac{K_{sA}}{L_{sA}}\Big)^{\frac{1-\sigma_{s}}{\sigma_{s}}}\Big]^{\frac{1}{\sigma_{s}-1}}}
    $$

    $$
    p_{sB}=\frac{\gamma_{mB}}{\gamma_{sB}}\frac{A_{mB}\tilde{B}_{mB}^{\frac{\sigma_{m}-1}{\sigma_{m}}}}{A_{sB}\tilde{B}_{sB}^{\frac{\sigma_{s}-1}{\sigma_{s}}}}\frac{\Big[\gamma_{mB}\tilde{B}^{\frac{\sigma_{m}-1}{\sigma_{m}}}+(1-\gamma_{mB})\Big(\frac{K_{mB}}{L_{mB}}\Big)^{\frac{1-\sigma_{m}}{\sigma_{m}}}\Big]^{\frac{1}{\sigma_{m}-1}}}{\Big[\gamma_{sB}\tilde{B}^{\frac{\sigma_{s}-1}{\sigma_{s}}}+(1-\gamma_{sB})\Big(\frac{K_{sB}}{L_{sB}}\Big)^{\frac{1-\sigma_{s}}{\sigma_{s}}}\Big]^{\frac{1}{\sigma_{s}-1}}}
    $$
        

    """)

# Define the 'Simulation' page
def page_simulation():
    st.markdown(r"""
    # Two-Sector Two-State Model

    Click on the 'Run Simulation' button at the bottom to start the simulation.
    
    #### Simulation Parameters
    Tip: If simulation is unstable, try reducing the 'Sensitivity' parameter.

    """)
    
    # Define the interactive widgets
    num_sectors = 2
    num_states = 2

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        length = int(st.text_input('Simulation Length', value=30))
    with col2:
        allow_labor_state_switch = st.checkbox(r'$L$ flows between states', value=True)
        allow_capital_state_switch = st.checkbox(r'$K$ flows between states', value=True)
    with col3:
        sensitivity = st.slider('Sensitivity', min_value=0.0, max_value=1.0, value=0.25)
    
    st.markdown("""#### Model Parameters""")

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        A_m = st.slider(r"Hicks neutral, initial ($A_{m0}$)", min_value=0.1, max_value=2.0, value=0.604)
        A_s = st.slider(r"Hicks-neutral coefficient ($A_{s0}$)", min_value=0.1, max_value=2.0, value=0.709)
        gamma_m = st.slider(r"Capital importance ($\gamma_m$)", min_value=0.0, max_value=1.0, value=0.333)
        gamma_s = st.slider(r"Capital importance ($\gamma_s$)", min_value=0.0, max_value=1.0, value=0.362)

    with col2:
        K_L_m = st.slider(r"Capital-labor ratio ($k_m$)", min_value=0.1, max_value=2.0, value=1.0)
        K_L_s = st.slider(r"Capital-labor ratio ($k_s$)", min_value=0.1, max_value=2.0, value=1.0)
        sigma_m = st.slider(r"Elasticity of substitution ($\sigma_{m}$)", min_value=0.0, max_value=2.0, value=0.776)
        sigma_s = st.slider(r"Elasticity of substitution ($\sigma_{s}$)", min_value=0.0, max_value=2.0, value=0.571)

    with col3:
        c_ls_min, c_ls_max = st.slider(
            r'$c_{L,S}\sim U(c_{L,S}^{\min}, c_{L,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 2.0)
        )
        c_ks_min, c_ks_max = st.slider(
            r'$c_{K,S}\sim U(c_{K,S}^{\min}, c_{K,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 1.0)
        )
        if allow_labor_state_switch:
            c_lt_min, c_lt_max = st.slider(
                r'$c_{L,T}\sim U(c_{L,T}^{\min}, c_{L,T}^{\max})$',
                min_value=0.0, 
                max_value=2.0, 
                value=(0.0, 2.0)
            )
        else:
            c_lt_min=99999999 #used large number instead of infinity so that 0*c_lt=0
            c_lt_max=99999999
        if allow_capital_state_switch:
            c_kt_min, c_kt_max = st.slider(
                r'$c_{K,T}\sim U(c_{K,T}^{\min}, c_{K,T}^{\max})$',
                min_value=0.0, 
                max_value=2.0, 
                value=(0.0, 1.0)
            )
        else:
            c_kt_min=99999999
            c_kt_max=99999999



    st.markdown(r"""
    
    #### Technology parameters

    """)

    # Divide the layout into columns
    col1, col2 = st.columns(2)

    with col1:
        g_A_ma = st.number_input(r"Growth rate, Hick's neutral $g(A_{mA})$", value=7.8)
        g_A_mb = st.number_input(r"Growth rate, Hick's neutral $g(A_{mB})$", value=7.8)
        g_A_sa = st.number_input(r"Growth rate, Hick's neutral $g(A_{sA})$", value=1.1)
        g_A_sb = st.number_input(r"Growth rate, Hick's neutral $g(A_{sB})$", value=1.1)
    
    with col2:
        g_Bt_ma = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{mA})$", value=-9.4)
        g_Bt_mb = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{mB})$", value=-9.4)
        g_Bt_sa = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{sA})$", value=-7.0)
        g_Bt_sb = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{sB})$", value=-7.0)

    # Button to trigger the simulation
    if st.button('Run Simulation'):
        st.markdown(r"""#### Results""") 
        run_simulation(num_sectors, num_states, K_L_m, K_L_s, A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb, sensitivity)   

       

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



