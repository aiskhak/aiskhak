import datetime as dtime
import numpy as np
#from dwave_qbsolv import QBSolv
#from dimod import ExactSolver
from dimod import BinaryQuadraticModel
from dimod import Vartype
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# timer begins
now1 = dtime.datetime.now()

# ---------------------------------------
# INPUT VARIABLES -----------------------
n = 9           # number of grid points
p = 9           # precision
j0 = 1          # position of the fixed point
nu = 1.e-3      # fluid viscosity
rho = 1.e3      # fluid density
dpdx = -2.e0    # pressure gradient
u0 = 1.e-8      # initial velocity
ly = 1.e0       # domain size
dy = ly/(n - 1) # spatial step
tol = 1.e-6     # tolerance for SS solution
nsteps = 9      # number of time steps
t = 0.e0        # initial time
#alpha = 0.4e0   # for time step

# ---------------------------------------
# NUMPY ARRAYS --------------------------
a = np.zeros((n,n))
b = np.zeros((n))
y = np.zeros((n))
u = np.zeros((n))
u_old = np.zeros((n))
ad = np.zeros((n,n*p))
v = np.zeros((n*p))
w = np.zeros((n*p,n*p))

# ---------------------------------------
# INITIAL CONDITIONS --------------------
for i in range(0,n):
    u[i] = u0

# ---------------------------------------
# GRID ----------------------------------
for i in range(0,n):
    y[i] = i*dy

    # ---------------------------------------
# TIME STEP -----------------------------
dt = 250.e0 #alpha*dy*dy/nu

# ============================================
# SOLUTION VIA QUANTUM COMPUTER ==============
# ---------------------------------------
# CONSTRUCT LINEAR SYSTEM ---------------
def construct_linear_system(n, u, dt):

    a = np.zeros((n,n))
    a[0,0] = 1.0
    for i in range(1,n-1): 
        a[i,i] = (-1.0 - 2.0*nu*dt/dy**2)
        a[i,i-1] = nu*dt/dy**2
        a[i,i+1] = a[i,i-1]
    a[n-1,n-1] = 1.0

    return a

# ---------------------------------------
# CONVERT MATRIX TO FIXED POINT FORMAT --
def convert_to_fixed_point(a, p, j0, n):
  
    for i_n in range(0,n):
        k_it = 0
        for i in range(0,n):
            k = 0
            start = k_it*p
            end = (k_it+1)*p
            for j in range(start,end):
                ad[i_n,j] = 2**(j0 - k - 1)*a[i_n,i]
                k = k + 1
            k_it = k_it + 1

    return ad

# ---------------------------------------
# CONSTRUCT RIGHT HAND SIDE -------------
def construct_rhs(n, u_old, dt):

    b = np.zeros((n))
    b[0] = 0.e0
    for i in range(1,n-1):
        b[i] = dt*dpdx/rho - u_old[i]
    b[n-1] = 0.e0

    return b

# ---------------------------------------
# CONSTRUCT QUBO MATRIX -----------------
def construct_qubo_matrix(ad, b, n, p):

    for j in range (0,n*p):
        sum1 = 0.0
        for i in range (0,n):
            sum1 = sum1 + ad[i,j]*(ad[i,j] - 2.0*b[i])
        v[j] = sum1
        for k in range (0,n*p):
            sum2 = 0.0
            for i in range (0,n):
                sum2 = sum2 + 2.0*ad[i,j]*ad[i,k]
            w[j,k] = sum2

    return v, w

# ---------------------------------------
# CONVERT FROM BINARY TO DECIMAL FORMAT -
def convert_binary_to_real(binary, length): 

	point = binary.find('.') 
	if (point == -1) : 
		point = length 
	intDecimal = 0
	fracDecimal = 0
	twos = 1
	for i in range(point-1, -1, -1): 
		intDecimal += ((ord(binary[i]) - ord('0')) * twos) 
		twos *= 2
	twos = 2	
	for i in range(point + 1, length):	
		fracDecimal += ((ord(binary[i]) - ord('0')) / twos); 
		twos *= 2.0
	ans = intDecimal + fracDecimal 
	
	return ans 

# ---------------------------------------
# SOLVE SYSTEM VIA CLASSICAL COMPUTER ---
def solve_classic(n, a, b, u_old):

    u[0] = b[0] / a[0,0]
    for i in range(1,n-1):
        u[i] = (b[i] - a[i,i-1]*u_old[i-1] - a[i,i+1]*u_old[i+1]) / a[i,i]
    u[n-1] = b[n-1] / a[n-1,n-1]

    return u

# ---------------------------------------
# SOLVE SYSTEM VIA QUANTUM COMPUTER -----
def solve_quantum(a, b, n, p, j0):
    
    ad = convert_to_fixed_point(a, p, j0, n)
    v, w = construct_qubo_matrix(ad, b, n, p)
    #response = QBSolv().sample_ising(v,w)
    #response = ExactSolver().sample_qubo(v, w)
    #dwave_sampler = DWaveSampler(solver={'qpu': True})
    #emb_sampler = EmbeddingComposite(dwave_sampler)
    #response = sampler.sample_qubo(v, w)
    #print(response)
    #response = ExactSolver().sample(bqm)
    #response = QBSolv().sample_qubo(bqm)
    #samples = list(response.sample())
    #energies = list(response.data_vectors['energy'])
    #minpos = energies.index(min(energies))
    offset = 0.0
    bqm = BinaryQuadraticModel(v, w, offset, Vartype.BINARY)
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm)
    sample = list(response.record)
    min_sample = sample[[0][0]]
    minen_sample = min_sample[0]
    k_it = 1
    u_bin = np.empty((n),dtype="<U40")
    for i in range(0,n):
        start = (k_it - 1)*p
        end = k_it*p
        dec = 0
        for j in range(start,end):
            if dec == j0:
                u_bin[i] = u_bin[i] + "."
            if minen_sample[j] == -1:
                u_bin[i] = u_bin[i] + str(0)
            else:
                u_bin[i] = u_bin[i] + str(minen_sample[j])
            dec = dec + 1
        k_it = k_it + 1
        start = (k_it - 1)*p
        u[i] = convert_binary_to_real(u_bin[i], len(u_bin[i]))
    
    return u

# ---------------------------------------
# WRITE PROFILE -------------------------
def write_prof(u, y, m, time_it):

    name = "u_prof_" + str(time_it) + ".dat"
    u_prof = open(name, "w", newline = "\n")
    for i in range(0,m):
        u_prof.write("%.8f" % y[i] + " ")
        u_prof.write("%.8f" % u[i] + " ")
        u_prof.write("\n")
    u_prof.close()

# ---------------------------------------
# MAIN PROGRAM --------------------------
def lam_flow_dwave(n, p, j0):
      
    global u
    global u_old
    global t

    # solution
    time_it = 0
    while (time_it <= nsteps):

        a = construct_linear_system(n, u_old, dt)
        b = construct_rhs(n, u_old, dt)

        # solve system via classical computer
        #u = solve_classic(n, a, b, u_old)

        # solve system via quantum computer
        u = solve_quantum(a, b, n, p, j0)

        # new time
        t = t + dt
        time_it = time_it + 1

        # writing profiles
        write_prof(u, y, n, time_it)

        # old value
        u_old = np.copy(u)
        
    return u

u = lam_flow_dwave(n, p, j0)
# SOLUTION VIA QUANTUM COMPUTER ==============
# ============================================

# ============================================
# DIRECT SOLUTION VIA CLASSICAL COMPUTER =====
def lam_flow_jacobi(n):

    global u
    global u_old
    global t

    # Jacobi method
    time_it = 0
    while (time_it <= nsteps):

        # old value
        u_old = np.copy(u)
        
        # new value
        u[0] = 0.0
        u[n-1] = 0.0
        for i in range(1,n-1):
                
            # matrix a and free vector b
            a[i,i] = (-1.0 - 2.0*nu*dt/dy**2)
            a[i,i-1] = nu*dt/dy**2
            a[i,i+1] = a[i,i-1]
            b[i] = dt*dpdx/rho - u_old[i]

            # new velocity value
            u[i] = (b[i] - a[i,i-1]*u_old[i-1] - a[i,i+1]*u_old[i+1]) / a[i,i]

        # new time
        t = t + dt
        time_it = time_it + 1

        # writing profiles
        write_prof(u, y, n, time_it)

    return u

#u = lam_flow_jacobi(n)
# DIRECT SOLUTION VIA CLASSICAL COMPUTER =====
# ============================================

# timer ends
now2 = dtime.datetime.now()

# print execution time
print ('The program execution is completed')
print ('The execution time is ', (now2 - now1))