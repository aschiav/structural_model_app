import streamlit as st
import Simulation as simulation



### Application ####

# Function to run the simulation with given parameters
def run_simulation(num_sectors, num_states, K_L_m, K_L_s, A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb, sensitivity):

    # Run the simulation
    sim=simulation.Simulation(num_sectors=num_sectors, 
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

# Define the 'Single-State Simulation' page
def page_simulation_singlestate():
    st.markdown(r"""
    # Two-Sector One-State Model

    Click on the 'Run Simulation' button at the bottom to start the simulation.
    
    #### Simulation Parameters
    Tip: If simulation is unstable, try reducing the 'Sensitivity' parameter.

    """)
    
    # Define the interactive widgets
    num_sectors = 2
    num_states = 1

    # Divide the layout into columns
    col1, col2 = st.columns(2)

    with col1:
        length = int(st.text_input('Simulation Length', value=30))
    with col2:
        sensitivity = st.slider('Sensitivity', min_value=0.0, max_value=1.0, value=0.25)
    
    st.markdown("""#### Model Parameters""")

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        A_m = st.slider(r"Hicks neutral, initial ($A_{m}$)", min_value=0.1, max_value=2.0, value=0.604)
        A_s = st.slider(r"Hicks-neutral coefficient ($A_{s}$)", min_value=0.1, max_value=2.0, value=0.709)
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
            value=(0.0, 0.0)
        )
        c_ks_min, c_ks_max = st.slider(
            r'$c_{K,S}\sim U(c_{K,S}^{\min}, c_{K,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 0.0)
        )

    st.markdown(r"""
    
    #### Technology parameters

    """)

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        g_Bt_ma = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{m})$", value=-9.4)
        g_Bt_sa = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{s})$", value=-7.0)
    
    with col2:
        g_A_ma = st.number_input(r"Growth rate, Hick's neutral $g(A_{m})$", value=7.8)
        g_A_sa = st.number_input(r"Growth rate, Hick's neutral $g(A_{s})$", value=1.1)


    with col3:
        st.number_input(r"Implied capital-augmenting $g(B_{m})$", value=g_Bt_ma+g_A_ma, min_value=g_Bt_ma+g_A_ma, max_value=g_Bt_ma+g_A_ma)
        st.number_input(r"Implied capital-augmenting $g(B_{s})$", value=g_Bt_sa+g_A_sa, min_value=g_Bt_sa+g_A_sa, max_value=g_Bt_sa+g_A_sa)
    
    #unused parameters
    c_lt_min=0
    c_lt_max=0
    c_kt_min=0
    c_kt_max=0
    g_A_mb=0
    g_A_sb=0
    g_Bt_mb=0
    g_Bt_sb=0

    # Button to trigger the simulation
    if st.button('Run Simulation'):
        st.markdown(r"""#### Results""") 
        run_simulation(num_sectors, num_states, K_L_m, K_L_s, A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb, sensitivity)   

# Define the 'Multi-State Simulation' page
def page_simulation_multistate():
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
        A_m = st.slider(r"Hicks neutral, initial ($A_{m}$)", min_value=0.1, max_value=2.0, value=0.604)
        A_s = st.slider(r"Hicks-neutral coefficient ($A_{s}$)", min_value=0.1, max_value=2.0, value=0.709)
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
            value=(0.0, 0.0)
        )
        c_ks_min, c_ks_max = st.slider(
            r'$c_{K,S}\sim U(c_{K,S}^{\min}, c_{K,S}^{\max})$',
            min_value=0.0, 
            max_value=2.0, 
            value=(0.0, 0.0)
        )
        if allow_labor_state_switch:
            c_lt_min, c_lt_max = st.slider(
                r'$c_{L,T}\sim U(c_{L,T}^{\min}, c_{L,T}^{\max})$',
                min_value=0.0, 
                max_value=2.0, 
                value=(0.0, 0.0)
            )
        else:
            c_lt_min=99999999 #used large number instead of infinity so that 0*c_lt=0
            c_lt_max=99999999
        if allow_capital_state_switch:
            c_kt_min, c_kt_max = st.slider(
                r'$c_{K,T}\sim U(c_{K,T}^{\min}, c_{K,T}^{\max})$',
                min_value=0.0, 
                max_value=2.0, 
                value=(0.0, 0.0)
            )
        else:
            c_kt_min=99999999
            c_kt_max=99999999



    st.markdown(r"""
    
    #### Technology parameters

    """)

    # Divide the layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        g_Bt_ma = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{mA})$", value=-9.4)
        g_Bt_mb = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{mB})$", value=-9.4)
        g_Bt_sa = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{sA})$", value=-7.0)
        g_Bt_sb = st.number_input(r"Growth of factor imbalance $g(\tilde{B}_{sB})$", value=-7.0)
    
    with col2:
        g_A_ma = st.number_input(r"Growth rate, Hick's neutral $g(A_{mA})$", value=7.8)
        g_A_mb = st.number_input(r"Growth rate, Hick's neutral $g(A_{mB})$", value=7.8)
        g_A_sa = st.number_input(r"Growth rate, Hick's neutral $g(A_{sA})$", value=1.1)
        g_A_sb = st.number_input(r"Growth rate, Hick's neutral $g(A_{sB})$", value=1.1)


    with col3:
        st.number_input(r"Implied capital-augmenting $g(B_{mA})$", value=g_Bt_ma+g_A_ma, min_value=g_Bt_ma+g_A_ma, max_value=g_Bt_ma+g_A_ma)
        st.number_input(r"Implied capital-augmenting $g(B_{mB})$", value=g_Bt_mb+g_A_mb, min_value=g_Bt_mb+g_A_mb, max_value=g_Bt_mb+g_A_mb)
        st.number_input(r"Implied capital-augmenting $g(B_{sA})$", value=g_Bt_sa+g_A_sa, min_value=g_Bt_sa+g_A_sa, max_value=g_Bt_sa+g_A_sa)
        st.number_input(r"Implied capital-augmenting $g(B_{sB})$", value=g_Bt_sb+g_A_sb, min_value=g_Bt_sb+g_A_sb, max_value=g_Bt_sb+g_A_sb)
    # Button to trigger the simulation
    if st.button('Run Simulation'):
        st.markdown(r"""#### Results""") 
        run_simulation(num_sectors, num_states, K_L_m, K_L_s, A_m, A_s, gamma_m, gamma_s, sigma_m, sigma_s, 
         c_ls_min, c_ls_max, c_lt_min, c_lt_max, c_ks_min, c_ks_max, c_kt_min, c_kt_max, 
         length, g_A_ma, g_A_mb, g_A_sa, g_A_sb, g_Bt_ma, g_Bt_mb, g_Bt_sa, g_Bt_sb, sensitivity)


if 'page' not in st.session_state:
    st.session_state['page']="Documentation"

# Define page navigation using buttons
if st.sidebar.button("Documentation"):
    st.session_state['page'] = "Documentation"
if st.sidebar.button("Single-State Simulation"):
    st.session_state['page'] = "Single-State Simulation"
if st.sidebar.button("Multi-State Simulation"):
    st.session_state['page'] = "Multi-State Simulation"




# Page routing
if st.session_state['page'] == "Documentation":
    page_documentation()
elif st.session_state['page'] == "Multi-State Simulation":
    page_simulation_multistate()
elif st.session_state['page'] == "Single-State Simulation":
    page_simulation_singlestate()
