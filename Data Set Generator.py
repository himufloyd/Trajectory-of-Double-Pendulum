class Pendulum():
    def __init__(self, M1, M2, L1, L2, Tmax, time_step, G = 9.81):
        self.g = G
        self.m1 = M1
        self.m2 = M2
        self.l1 = L1
        self.l2 = L2
        self.T = Tmax
        self.dt = time_step
    
    def system(self, theta1, theta2, omega1, omega2, t):
        # Define the equations of motion
        f1 = omega1
        f2 = omega2
        f3 = (-self.g*(2*self.m1+self.m2)*np.sin(theta1) - self.m2*self.g*np.sin(theta1-2*theta2) -\
            2*np.sin(theta1-theta2)*self.m2*(omega2**2*self.l2 + omega1**2*self.l1*np.cos(theta1-theta2))) / \
            (self.l1*(2*self.m1+self.m2-self.m2*np.cos(2*theta1-2*theta2)))
        f4 = (2*np.sin(theta1-theta2)*(omega1**2*self.l1*(self.m1+self.m2) + self.g*(self.m1+self.m2)*np.cos(theta1) + \
            omega2**2*self.l2*self.m2*np.cos(theta1-theta2))) / \
            (self.l2*(2*self.m1+self.m2-self.m2*np.cos(2*theta1-2*theta2)))
        
        return np.array([f1, f2, f3, f4])
    
    def implicit_solv(self, theta1, theta2, omega1, omega2):
        # Define the implicit midpoint rule
        def F(y):
            return y - np.array([theta1, theta2, omega1, omega2]) - self.dt*self.system((y[0]+theta1)/2, (y[1]+theta2)/2, (y[2]+omega1)/2, (y[3]+omega2)/2, 0)
        
        # Solve for the next state using the implicit midpoint rule
        sol = root(F, [theta1, theta2, omega1, omega2])
        theta1_new, theta2_new, omega1_new, omega2_new = sol.x
        theta1_new, theta2_new = (theta1_new + np.pi) % (2*np.pi) - np.pi, (theta2_new + np.pi) % (2*np.pi) - np.pi
        return theta1_new, theta2_new, omega1_new, omega2_new
    
    def evolve(self, theta1, theta2, omega1, omega2, t, sampling_rate = 100):
        # Create arrays to store the values of theta1 and theta2 over time
        time_list = [t]
        theta1_list = [theta1]
        theta2_list = [theta2]
        omega1_list = [omega1]
        omega2_list  = [omega2]

        # Solve the system using the implicit midpoint rule
        while (t<self.T):
            theta1, theta2, omega1, omega2 = self.implicit_solv(theta1, theta2, omega1, omega2)
            t+=self.dt

            # Append the current state of the system to the arrays
            time_list.append(t)
            theta1_list.append(theta1)
            theta2_list.append(theta2)
            omega1_list.append(omega1)
            omega2_list.append(omega2)
        
        time_list, theta1_list, theta2_list, omega1_list, omega2_list = time_list[::sampling_rate], theta1_list[::sampling_rate], theta2_list[::sampling_rate], omega1_list[::sampling_rate], omega2_list[::sampling_rate]
        return time_list, theta1_list, theta2_list, omega1_list, omega2_list


def solve(theta1_0, theta2_0, omega1_0, omega2_0, L1, L2, M1, M2, file_path, Tmax = 100, time_step = 0.01, intial_time = 0.0, sampling_rate = 1):
    pen = Pendulum(M1, M2, L1, L2, Tmax, time_step)
    time, theta1_final, theta2_final, omega1_final, omega2_final = pen.evolve(theta1_0, theta2_0, omega1_0, omega2_0, intial_time, sampling_rate)
    arr = np.array([time, theta1_final, theta2_final, omega1_final, omega2_final]).T
    system_info = {"Masses": [M1, M2], "Lengths": [L1, L2], "Intial Angles": [theta1_0, theta2_0], "Initial Angular Velocity": [omega1_0, omega2_0]}
    np.savez(file_path, data=arr, metadata = system_info)

def generate_data(N, Tmax, Lmax, Mmax, Wmax, path):

    print("Generating Trainning Data.....", flush = True)

    st = timeit.default_timer()

    # Generate initial conditions
    th1_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(0.8*N)).tolist()
    th2_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(0.8*N)).tolist()
    w1_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(0.8*N)).tolist()
    w2_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(0.8*N)).tolist()
    L1 = np.random.uniform(low=1.0, high=Lmax, size=(int)(0.8*N)).tolist()
    L2 = np.random.uniform(low=1.0, high=Lmax, size=(int)(0.8*N)).tolist()
    M1 = np.random.uniform(low=1.0, high=Mmax, size=(int)(0.8*N)).tolist()
    M2 = np.random.uniform(low=1.0, high=Mmax, size=(int)(0.8*N)).tolist()
    train_path = np.array([path+'Train/'+str(i)+".npz" for i in range((int)(0.8*N))]).tolist()

    
    # Generate trajectories
    X = list(zip(th1_0, th2_0, w1_0, w2_0, L1, L2, M1, M2, train_path))
    with Pool(11) as pool:
        for _ in pool.starmap(solve, X):
            pass
    
    en = timeit.default_timer()
    print("It took {} seconds to generate {} datasets....".format(en-st, 0.8*N), flush = True)

    print("Generating Validation Data.....", flush = True)

    st = timeit.default_timer()

    # Generate initial conditions
    th1_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(0.1*N)).tolist()
    th2_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(0.1*N)).tolist()
    w1_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(0.1*N)).tolist()
    w2_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(0.1*N)).tolist()
    L1 = np.random.uniform(low=1.0, high=Lmax, size=(int)(0.1*N)).tolist()
    L2 = np.random.uniform(low=1.0, high=Lmax, size=(int)(0.1*N)).tolist()
    M1 = np.random.uniform(low=1.0, high=Mmax, size=(int)(0.1*N)).tolist()
    M2 = np.random.uniform(low=1.0, high=Mmax, size=(int)(0.1*N)).tolist()
    val_path = np.array([path+'Val/'+str(i)+".npz" for i in range((int)(0.1*N))]).tolist()

    
    # Generate trajectories
    X = list(zip(th1_0, th2_0, w1_0, w2_0, L1, L2, M1, M2, val_path))
    with Pool(7) as pool:
        for _ in pool.starmap(solve, X):
            pass
    
    en = timeit.default_timer()
    print("It took {} seconds to generate {} datasets....".format(en-st, 0.1*N), flush = True)

    print("Generating Testing Data.....", flush = True)

    st = timeit.default_timer()

    # Generate initial conditions
    th1_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(1)).tolist()
    th2_0 = np.random.uniform(low=-np.pi, high=np.pi, size=(int)(1)).tolist()
    w1_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(1)).tolist()
    w2_0 = np.random.uniform(low=-Wmax, high=Wmax, size=(int)(1)).tolist()
    L1 = np.random.uniform(low=1.0, high=Lmax, size=(int)(1)).tolist()
    L2 = np.random.uniform(low=1.0, high=Lmax, size=(int)(1)).tolist()
    M1 = np.random.uniform(low=1.0, high=Mmax, size=(int)(1)).tolist()
    M2 = np.random.uniform(low=1.0, high=Mmax, size=(int)(1)).tolist()
    test_path = np.array([path+'Test/'+str(i)+".npz" for i in range((int)(1))]).tolist()
    
    # Generate trajectories
    X = list(zip(th1_0, th2_0, w1_0, w2_0, L1, L2, M1, M2, test_path))
    with Pool(7) as pool:
        for _ in pool.starmap(solve, X):
            pass
    
    en = timeit.default_timer()
    print("It took {} seconds to generate {} datasets....".format(en-st, 1), flush = True)
