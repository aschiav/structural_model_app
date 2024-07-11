import Economy as economy

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

class Simulation:
    def __init__(self, num_sectors, num_states, length, #simulation parameters
                alpha=0.5, beta=0.5,
                 K_L_m=1, K_L_s=1,#initial values
                 A_m=1, A_s=1, gamma_m=0.5, gamma_s=0.5, sigma_m=0.5, sigma_s=0.5, #model parameters
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
        self.economy=economy.Economy(num_states=num_states, num_sectors=num_sectors,
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
            print("period: ",t)
            self.economy.shock(g_A_ma=self.g_A_ma, g_A_mb=self.g_A_mb, g_A_sa=self.g_A_sa, g_A_sb=self.g_A_sb, g_Bt_ma=self.g_Bt_ma, g_Bt_mb=self.g_Bt_mb, g_Bt_sa=self.g_Bt_sa, g_Bt_sb=self.g_Bt_sb)
            self.economy.update()
            df=self.economy.get_data()
            df.insert(0, 't', t)  
            simulation_data.append(df)
        
        # Concatenate all the DataFrames in the state_data list into a single DataFrame
        combined_df = pd.concat(simulation_data, ignore_index=True)
        
        # Remove the first row where t == 0
        #combined_df = combined_df[combined_df['t'] != 0]
    
        #transform state values from numeric to letters (A and B)
        combined_df["state"]=combined_df["state"].replace({0:'A',1:'B'})
        
        #transform sector values from numeric to letters (m and s)
        combined_df["sector"]=combined_df["sector"].replace({0:'m',1:'s'})

        return combined_df
    
    def visualize(self, sim_data):

        if self.num_states==1:
        # List of y variables
            y_vars = ['r','w', 'utility','MPL', 'MPK','Y', 'pY', 'p', 'K/L', 'K', 'L', 'lambda_agg', 'psi_state_sector','psi_agg']
            y_var_titles=['Profit rate (MRPK)', 'Wage (MRPL)', 'Utility', 'Marginal Prod. (L)', 'Marginal Prod. (K)', 'Output', 'Nominal Output', 'Prices', 'Capital-labor ratio', 'Capital', 'Labor', 'Sector employment share', 'Sectoral labor share','State labor share']
        if self.num_states==2:
        # List of y variables
            y_vars = ['r','w', 'utility','MPL', 'MPK','Y', 'pY', 'p', 'K/L', 'K', 'L', 'lambda_state', 'lambda_agg', 'psi_state_sector', 'psi_state','psi_agg']
            y_var_titles=['Profit rate (MRPK)', 'Wage (MRPL)', 'Utility', 'Marginal Prod. (L)', 'Marginal Prod. (K)', 'Output', 'Nominal Output', 'Prices', 'Capital-labor ratio', 'Capital', 'Labor', 'State-sector employment share', 'Sector employment share (National)', 'Sectoral labor share',
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

