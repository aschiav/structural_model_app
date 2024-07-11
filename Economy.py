import Worker as worker
import Capital as capital
import pandas as pd
import numpy as np


class Economy:
    def __init__(self, num_sectors, num_states,
                 alpha, beta,
                 K_L_m, K_L_s,
                 A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
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
        self.L = np.array([[1000 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.K = self.L * self.K_L

        # Initialize A matrix
        self.A = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.A[0][0]=A_m
        self.A[1][0]=A_s
        if num_states==2:
            self.A[0][1]=A_m
            self.A[1][1]=A_s


        # Initialize B=1 matrix
        self.B = np.array([[1 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)

        # Initialize gamma matrix
        self.gamma = np.array([[0.5 for _ in range(num_states)] for _ in range(num_sectors)], dtype=float)
        self.gamma[0][0]=gamma_m
        self.gamma[1][0]=gamma_s
        if num_states==2:
            self.gamma[0][1]=gamma_m
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
        if self.num_states==2:
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
                self.capital[i][j] = [capital.Capital(sector=i, state=j, c_s_min=c_ks_min, c_s_max=c_ks_max, c_t_min=c_kt_min, c_t_max=c_kt_max, economy=self) for _ in range(int(self.K[i][j]))]
        
        # Populate the workers matrix with list of Worker type
        for i in range(num_sectors):
            for j in range(num_states):
                self.workers[i][j] = [worker.Worker(sector=i, state=j, alpha=alpha, beta=beta, c_s_min=c_ls_min, c_s_max=c_ls_max, c_t_min=c_lt_min, c_t_max=c_lt_max, economy=self) for _ in range(int(self.L[i][j]))]
                
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
        

        import warnings
        warnings.filterwarnings("ignore")

        print("Y:")
        print(self.Y)
        print("MPL:")
        print(self.MPL)
        print("MPK:")
        print(self.MPK)
        print("r:")
        print(self.r)
        print("w:")
        print(self.w)
        print("L:")
        print(self.L)
        print("K:")
        print(self.K)
        print("p:")
        print(self.p)
    
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
        #ACetal prices
        term1=(self.gamma[0][0]*self.A[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0])))/(self.gamma[1][0]*self.A[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0])))
        term2_numerator=(self.gamma[0][0]*(self.B[0][0]**((self.sigma[0][0]-1)/self.sigma[0][0]))+(1-self.gamma[0][0])*self.K_L[0][0]**((1-self.sigma[0][0])/self.sigma[0][0]))**(1/(self.sigma[0][0]-1))
        term2_denominator=(self.gamma[1][0]*(self.B[1][0]**((self.sigma[1][0]-1)/self.sigma[1][0]))+(1-self.gamma[1][0])*self.K_L[1][0]**((1-self.sigma[1][0])/self.sigma[1][0]))**(1/(self.sigma[1][0]-1))
        self.p[1][0]=term1*(term2_numerator/term2_denominator)

        if self.num_states==2:
            term1 = (self.gamma[0][1] * self.A[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1]))) / (self.gamma[1][1] * self.A[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])))
            term2_numerator = (self.gamma[0][1] * (self.B[0][1] ** ((self.sigma[0][1] - 1) / self.sigma[0][1])) + (1 - self.gamma[0][1]) * self.K_L[0][1] ** ((1 - self.sigma[0][1]) / self.sigma[0][1])) ** (1 / (self.sigma[0][1] - 1))
            term2_denominator = (self.gamma[1][1] * (self.B[1][1] ** ((self.sigma[1][1] - 1) / self.sigma[1][1])) + (1 - self.gamma[1][1]) * self.K_L[1][1] ** ((1 - self.sigma[1][1]) / self.sigma[1][1])) ** (1 / (self.sigma[1][1] - 1))
            self.p[1][1] = term1 * (term2_numerator / term2_denominator)

        #update distribution
        self.psi=(self.w * self.L)/ (self.w*self.L+self.r*self.K) 


        #update inputs
        self.update_workers()   
        self.update_capital()

        #update capital labor ratios
        self.K_L=self.K/self.L


 


    def shock(self, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb):
        
        self.A[0][0]=self.A[0][0]*(1+g_A_ma/100)
        self.A[1][0]=self.A[1][0]*(1+g_A_sa/100)
        if self.num_states==2:
            self.A[0][1]=self.A[0][1]*(1+g_A_mb/100)        
            self.A[1][1]=self.A[1][1]*(1+g_A_sb/100)

        self.B[0][0]=self.B[0][0]*(1+g_Bt_ma/100)
        self.B[1][0]=self.B[1][0]*(1+g_Bt_sa/100)
        if self.num_states==2:
            self.B[0][1]=self.B[0][1]*(1+g_Bt_mb/100)        
            self.B[1][1]=self.B[1][1]*(1+g_Bt_sb/100)
        
        
    def get_data(self):
        dataframe=[]
        
        #sector variables
        for i in range(self.num_sectors):
            for j in range(self.num_states):
                if self.num_states==1:
                   data = {
                        "sector":i,
                        "state":j,
                        "L":self.L[i][j],
                        "K":self.K[i][j],
                        "Y":self.Y[i][j],
                        "pY":self.p[i][j]*self.Y[i][j],
                        "MPL":self.MPL[i][j],
                        "MPK":self.MPK[i][j],
                        "p":self.p[i][j],
                        "K/L":self.K[i][j]/self.L[i][j],
                        "w":self.w[i][j],
                        "r":self.r[i][j],
                        "utility":(self.alpha * self.w[i][j] / self.p[0][j]) ** self.alpha * (self.beta * self.w[i][j] / self.p[1][j]) ** self.beta,
                        "psi_state_sector":self.psi[i][j],
                        "lambda_agg":self.L[i][j]/(self.L[0][j]+self.L[1][j]), #this will only work with two sectors!
                        "psi_agg":np.sum(self.L*self.w)/(np.sum(self.p*self.Y) )
                    } 
                if self.num_states==2:
                    data = {
                        "sector":i,
                        "state":j,
                        "L":self.L[i][j],
                        "K":self.K[i][j],
                        "Y":self.Y[i][j],
                        "pY":self.p[i][j]*self.Y[i][j],
                        "MPL":self.MPL[i][j],
                        "MPK":self.MPK[i][j],
                        "p":self.p[i][j],
                        "K/L":self.K[i][j]/self.L[i][j],
                        "w":self.w[i][j],
                        "r":self.r[i][j],
                        "utility":(self.alpha * self.w[i][j] / self.p[0][j]) ** self.alpha * (self.beta * self.w[i][j] / self.p[1][j]) ** self.beta,
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

