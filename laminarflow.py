import datetime as dt
import numpy as np

# timer
now1 = dt.datetime.now()

# ---------------------------------------
# INPUT VARIABLES -----------------------
n = 20          # number of nodes along y
nbits = 20      # number of bits
j0 = 20         # precision
nu = 1.e-6      # fluid viscosity
rho = 1.e3      # fluid density
dpdx = -2.4e0   # pressure gradient
u0 = 2.e-2      # initial velocity
ly = 1.e-2      # domain size
dy = ly/(n - 1) # spatial step
t = 0.e0        # initial time
tol = 1.e-6     # tolerance

# ---------------------------------------
# NUMPY ARRAYS --------------------------
a = np.zeros((n,n))
b = np.zeros((n))
y = np.zeros((n))
u = np.zeros((n))
u_old = np.zeros((n))
ad = np.zeros((n,n*nbits))
v = np.zeros((n*nbits))
w = np.zeros((n*nbits,n*nbits))

# ---------------------------------------
# INITIAL CONDITIONS --------------------
for i in range(0,n):
    u[i] = u0

# ---------------------------------------
# GRID ----------------------------------
for i in range(0,n):
    y[i] = i*dy

# ---------------------------------------
# CONSTRUCT LINEAR SYSTEM ---------------
def construct_linear_system(n,u,dt):

    a = np.zeros((n,n))
    a[1,1] = 1.0
    for i in range(1,n-1): 
        a[i,i] = (-1.0 - 2.0*nu*dt/dy**2)
        a[i,i-1] = nu*dt/dy**2
        a[i,i+1] = a[i,i-1]
    a[n-1,n-1] = 1.0

    return a

# ---------------------------------------
# CONVERT MATRIX TO FIXED POINT FORMAT --
def convert_to_fixed_point(a,nbits,j0,n):
  
    for i_n in range(0,n):
        k_it = 0
        for i in range(0,n):
            k = 0
            start = k_it*nbits
            end = (k_it+1)*nbits
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
def construct_qubo_matrix(ad, b, n, nbits):

    for j in range (0,n*nbits):
        sum1 = 0.0
        for i in range (0,n):
            sum1 = sum1 + ad[i,j]*(ad[i,j] - 2.0*b[i])
        v[j] = sum1
        for k in range (0,n*nbits):
            sum2 = 0.0
            for i in range (0,n):
                sum2 = sum2 + 2.0*ad[i,j]*ad[i,k]
            w[j,k] = sum2

    return v, w # ???

# ---------------------------------------
# SOLVE QUBO MATRIX ---------------------
def solve_qubo(virtualQ, num_reads):

    # convert a problem specified as a qubo to an ising form (-1,1)
    virtualh, virtualJ, ising_offset = util.qubo_to_ising(virtualQ)

    # metaheuristic provided by sapi to embed any logical problem graph to the chimera architecture
    embeddings = embedding.find_embedding(virtualQ.keys(), adjacency)
    h, J, Jc, newembeddings = embedding.embed_problem(virtualh, virtualJ, embeddings, adjacency)

    # the solver interface to DWave QPU which solves the embedded problem
    answer = core.solve_ising(solver, h, J, num_reads = numreads)

    # post-processing algorithm to obtain a solution in the logical space
    unembedded_answer = embedding.unembed_answer(answer["solutions"], newembeddings, broken_chains = "weighted_random")
    qsol = map(lambda x: map(lambda y: 0.5*(y + 1), x), unembedded_answer)
    
    return qsol

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
# FIND WEIGHTED AVERAGE -----------------
def weighted_average(u_all):

    global u

    u = u_all   # ???

    return u

# ---------------------------------------
# SOLVE SYSTEM VIA CLASSICAL COMPUTER ---
def solve_classic(n, a, b, u_old):

    u[0] = b[1] / a[1,1]
    for i in range(1,n-1):
        u[i] = (b[i] - a[i,i-1]*u_old[i-1] - a[i,i+1]*u_old[i+1]) / a[i,i]
    u[n-1] = b[n-1] / a[n-1,n-1]

    return u

# ---------------------------------------
# MAIN PROGRAM --------------------------
def lam_flow_dwave(n, nbits, j0):
      
    import numpy as np
    global u
    global u_old
    global t

    # solution
    k_it = 1
    norm = 1.e8
    while (norm > tol):

        u_max = np.ndarray.max(u)
        dt = 0.1*dy / u_max
        a = construct_linear_system(n, u_old, dt)
        ad = convert_to_fixed_point(a, nbits, j0, n)
        b = construct_rhs(n, u_old, dt)

        # solve system via classical computer
        u = solve_classic(n, a, b, u_old)

        # solve system via quantum computer
        # v, w
        # virtualQ = construct_qubo_matrix(ad, b, n, nbits)
        # QSol = solve_qubo(virtualQ, num_reads)
        # loop over all solution states:
        #     u_all[i] = convert_binary_to_real(srt(QSol[i]), len(str(QSol[i])))
        # u = weighted_average(u_all)
        
        t = t + dt
        norm = np.linalg.norm(u - u_old)
        u_old = np.copy(u)
        
    return u

# ---------------------------------------
# DIRECT SOLUTION VIA CLASSICAL COMPUTER
def lam_flow_jacobi(n):

    import numpy as np

    # Jacobi method
    k_it = 1
    norm = 1.e8
    while (norm > tol):

        # old value
        u_old = np.copy(u)
        
        # time step
        u_max = np.ndarray.max(u)
        dt = 0.1*dy / u_max

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

        # matrix 2nd norm
        norm = np.linalg.norm(u - u_old)

    return u

# direct solution via classical computer
#u = lam_flow_jacobi(n)

# solution via quantum computer
u = lam_flow_dwave(n, nbits, j0)

# writing profile
def write_prof(u, y, m):

    u_prof = open("../work/u_prof.dat", "w", newline = "\n")
    for i in range(0,m):
        u_prof.write("%.8f" % y[i] + " ")
        u_prof.write("%.8f" % u[i] + " ")
        u_prof.write("\n")
    u_prof.close()

# writing profiles
write_prof(u, y, n)

# timer
now2 = dt.datetime.now()
print ('execution time is ', (now2 - now1))