test_path = '/content/drive/MyDrive/Pendulum/Data/Test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_param = model_params['RNN']
test_dataset = DoublePendulumDataset(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=model_param['batch_size'], shuffle=True)
model = torch.load(model_param['model_path'])
model.eval()
output_list = []
with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader, 0), desc = 'Test', leave=False, total = len(test_dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

class simulate_plot():
    def __init__(self, labels, outputs):
        labels = labels.detach().cpu()
        outputs = outputs.detach().cpu()
        theta1, theta2, omega1, omega2 = labels[:,:,0][0].tolist(), labels[:,:,1][0].tolist(), labels[:,:,2][0].tolist(), labels[:,:,3][0].tolist()
        theta1_pre, theta2_pre, omega1_pre, omega2_pre = outputs[:,:,0][0].tolist(), outputs[:,:,1][0].tolist(), outputs[:,:,2][0].tolist(), outputs[:,:,3][0].tolist()
        self.data = {}
        self.data['theta1'] = theta1
        self.data['theta2'] = theta2
        self.data['theta1_pre'] = theta1_pre
        self.data['theta2_pre'] = theta2_pre
        self.data['omega1'] = omega1
        self.data['omega1_pre'] = omega1_pre
        self.data['omega2'] = omega2
        self.data['omega2_pre'] = omega2_pre
        self.rescale()
    
    def rescale(self):
        self.data['theta1'] = np.array([(i + np.pi) % (2*np.pi) - np.pi for i in self.data['theta1']])
        self.data['theta2'] = np.array([(i + np.pi) % (2*np.pi) - np.pi for i in self.data['theta2']])
        self.data['theta1_pre'] = np.array([(i + np.pi) % (2*np.pi) - np.pi for i in self.data['theta1_pre']])
        self.data['theta2_pre'] = np.array([(i + np.pi) % (2*np.pi) - np.pi for i in self.data['theta2_pre']])
    
    def plot_trajectory(self, st = 0, len = 1000):
        # Plot the trajectory in the theta1-theta2 plane
        plt.plot(self.data['theta1'][st:st+len], self.data['theta2'][st:st+len], label = 'RK4')
        plt.plot(self.data['theta1_pre'][st:st+len], self.data['theta2_pre'][st:st+len], '--', label = 'Neural Network')

        # Set the axis labels and title
        plt.xlabel('Theta 1 (radians)')
        plt.ylabel('Theta 2 (radians)')
        plt.title('Trajectory Plot')

        # Show the plot
        plt.legend()
        plt.show()

    def plot_phase_space(self, st = 0, len = 1000):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Phase Space Plot', fontsize=16)
        
        ax1.plot(self.data['theta1'][st:st+len], self.data['omega1'][st:st+len], label = 'RK4')
        ax1.plot(self.data['theta1_pre'][st:st+len], self.data['omega1_pre'][st:st+len], label = 'Neural Network')
        ax1.set_xlabel('theta1')
        ax1.set_ylabel('omega1')
        ax1.set_title('Phase Space Plot for Bob 1')
        ax1.legend()

        ax2.plot(self.data['theta2'][st:st+len], self.data['omega2'][st:st+len], label = 'RK4')
        ax2.plot(self.data['theta2_pre'][st:st+len], self.data['omega2_pre'][st:st+len], label = 'Neural Network')
        ax2.set_xlabel('theta2')
        ax2.set_ylabel('omega2')
        ax2.set_title('Phase Space Plot for Bob 2')
        ax2.legend()
        
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def plot_time_series(self, st = 0, len = 1000):
        time_steps = range(self.data['theta1'].shape[0])

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Time Series Plot", fontsize=16)

        axs[0, 0].plot(time_steps[st:st+len], self.data['theta1'][st:st+len], label='RK4')
        axs[0, 0].plot(time_steps[st:st+len], self.data['theta1_pre'][st:st+len], '--', label='Neural Network')
        axs[0, 0].set_xlabel('Time step')
        axs[0, 0].set_ylabel('Theta')
        axs[0, 0].set_title('Theta 1')
        axs[0, 0].legend()

        axs[0, 1].plot(time_steps[st:st+len], self.data['theta2'][st:st+len], label='RK4')
        axs[0, 1].plot(time_steps[st:st+len], self.data['theta2_pre'][st:st+len], '--', label='Neural Network')
        axs[0, 1].set_xlabel('Time step')
        axs[0, 1].set_ylabel('Theta')
        axs[0, 1].set_title('Theta 2')
        axs[0, 1].legend()

        axs[1, 0].plot(time_steps[st:st+len], self.data['omega1'][st:st+len], label='RK4')
        axs[1, 0].plot(time_steps[st:st+len], self.data['omega1_pre'][st:st+len], '--', label='Neural Network')
        axs[1, 0].set_xlabel('Time step')
        axs[1, 0].set_ylabel('Omega')
        axs[1, 0].set_title('Omega 1')
        axs[1, 0].legend()

        axs[1, 1].plot(time_steps[st:st+len], self.data['omega2'][st:st+len], label='RK4')
        axs[1, 1].plot(time_steps[st:st+len], self.data['omega2_pre'][st:st+len], '--', label='Neural Network')
        axs[1, 1].set_xlabel('Time step')
        axs[1, 1].set_ylabel('Omega')
        axs[1, 1].set_title('Omega 2')
        axs[1, 1].legend()

        fig.text(0.5, 0.04, 'Time', ha='center')
        fig.text(0.08, 0.5, 'Value', va='center', rotation='vertical')

        plt.tight_layout()
        plt.show()

    def plot_energy(self, l1 = 1, l2 = 1, m1 = 1, m2 = 1, g = 9.81, st = 0, len = 1000):
        time_steps = range(len(self.data['theta1']))

        # Calculate energy for each time step
        KE = 0.5 * (m1 * l1**2 * self.data['omega1']**2 + m2 * (l1**2 * self.data['omega1']**2 + l2**2 * self.data['omega2']**2 + 2 * l1 * l2 * self.data['omega1'] * self.data['omega2'] * np.cos(self.data['theta1'] - self.data['theta2'])))
        PE = -g * (m1 * l1 * np.cos(self.data['theta1']) + m2 * (l1 * np.cos(self.data['theta1']) + l2 * np.cos(self.data['theta2'])))
        energy = KE + PE

        KE_pre = 0.5 * (m1 * l1**2 * self.data['omega1_pre']**2 + m2 * (l1**2 * self.data['omega1']**2 + l2**2 * self.data['omega2']**2 + 2 * l1 * l2 * self.data['omega1'] * self.data['omega2'] * np.cos(self.data['theta1'] - self.data['theta2'])))
        PE_pre = -g * (m1 * l1 * np.cos(self.data['theta1_pre']) + m2 * (l1 * np.cos(self.data['theta1_pre']) + l2 * np.cos(self.data['theta2_pre'])))
        energy_pre = KE_pre + PE_pre
        
        # Plot energy as a function of time
        fig, ax = plt.subplots(1, 3, figsize=(10, 8))
        fig.suptitle("Energy Plot", fontsize=16)

        ax[0,0].plot(time_steps[st:st+len], KE[st:st+len], label = 'RK4')
        ax[0,0].plot(time_steps[st:st+len], KE_pre[st:st+len], label = 'Neural Network')
        ax[0,0].set_title("Kinetic Energy")
        ax[0,0].legend()

        ax[0,1].plot(time_steps[st:st+len], PE[st:st+len], label = 'RK4')
        ax[0,1].plot(time_steps[st:st+len], PE_pre[st:st+len], label = 'Neural Network')
        ax[0,1].set_title("Potential Energy")
        ax[0,1].legend()

        ax[0,2].plot(time_steps[st:st+len], energy[st:st+len], label = 'RK4')
        ax[0,2].plot(time_steps[st:st+len], energy_pre[st:st+len], label = 'RK4')
        ax[0,2].set_title("Total Energy")
        ax[0,2].legend()
        
        fig.text(0.5, 0.04, 'Time', ha='center')
        fig.text(0.08, 0.5, 'Value', va='center', rotation='vertical')

        plt.show()

    def plot_frequency(self, dt = 0.01, st = 0, len = 1000):
        # Compute the Fourier transforms of the angle and angular velocity data
        fft_theta1 = np.fft.fft(self.data['theta1'])
        fft_theta2 = np.fft.fft(self.data['theta2'])
        fft_omega1 = np.fft.fft(self.data['omega1'])
        fft_omega2 = np.fft.fft(self.data['omega2'])

        fft_theta1_pre = np.fft.fft(self.data['theta1_pre'])
        fft_theta2_pre = np.fft.fft(self.data['theta2_pre'])
        fft_omega1_pre = np.fft.fft(self.data['omega1_pre'])
        fft_omega2_pre = np.fft.fft(self.data['omega2_pre'])

        
        # Compute the frequencies corresponding to the Fourier coefficients
        freqs = np.fft.fftfreq(len(self.data['theta1']), dt)
        
        # Plot the magnitude spectra of the Fourier transforms
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle("Freqency Plot", fontsize=16)

        ax[0,0].semilogy(freqs[st:st+len], np.abs(fft_theta1)[st:st+len], label=r'$\theta_1$')
        ax[0,0].semilogy(freqs[st:st+len], np.abs(fft_theta2)[st:st+len], label=r'$\theta_2$')
        ax[0,0].set_ylabel('Magnitude')
        ax[0,0].set_title("RK4")
        ax[0,0].legend()

        ax[0,1].semilogy(freqs[st:st+len], np.abs(fft_theta1_pre)[st:st+len], label=r'$\theta_2$')
        ax[0,1].semilogy(freqs[st:st+len], np.abs(fft_theta2_pre)[st:st+len], label=r'$\theta_2$')
        ax[0,1].set_ylabel('Magnitude')
        ax[0,1].set_title("Neural Network")
        ax[0,1].legend()        
        
        ax[1,0].semilogy(freqs[st:st+len], np.abs(fft_omega1)[st:st+len], label=r'$\omega_1$')
        ax[1,0].semilogy(freqs[st:st+len], np.abs(fft_omega2)[st:st+len], label=r'$\omega_2$')
        ax[1,0].set_xlabel('Frequency (Hz)')
        ax[1,0].set_ylabel('Magnitude')
        ax[1,0].set_title("RK4")
        ax[1,0].legend()

        ax[1,1].semilogy(freqs[st:st+len], np.abs(fft_omega1_pre)[st:st+len], label=r'$\omega_1$')
        ax[1,1].semilogy(freqs[st:st+len], np.abs(fft_omega2_pre)[st:st+len], label=r'$\omega_2$')
        ax[1,1].set_xlabel('Frequency (Hz)')
        ax[1,1].set_ylabel('Magnitude')
        ax[1,1].set_title("Neural Network")
        ax[1,1].legend()

        plt.show()

    def animate(self, st=0, len=1000, L1=1, L2=1, interval=5):
            # Set up the figure and axes
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            fig.subplots_adjust(wspace=0.4)
            
            # Set limits and aspect ratio for both axes
            for ax in [ax1, ax2]:
                ax.set_xlim(-(L1+L2), L1+L2)
                ax.set_ylim(-(L1+L2), L1+L2)
                ax.set_aspect('equal')
            
            # Initialize the line objects representing the pendulums for each subplot
            line1, = ax1.plot([], [], 'o-', lw=2)
            line2, = ax1.plot([], [], 'o-', lw=2)
            line1_pre, = ax2.plot([], [], 'o-', lw=2)
            line2_pre, = ax2.plot([], [], 'o-', lw=2)
            
            # Define the update function for each subplot
            def update(i):
                x1 = L1 * np.sin(self.data['theta1'][st:st+len][i])
                y1 = -L1 * np.cos(self.data['theta1'][st:st+len][i])
                x2 = x1 + L2 * np.sin(self.data['theta2'][st:st+len][i])
                y2 = y1 - L2 * np.cos(self.data['theta2'][st:st+len][i])
                
                line1.set_data([0, x1], [0, y1])
                line2.set_data([x1, x2], [y1, y2])
                
                x1_pre = L1 * np.sin(self.data['theta1_pre'][st:st+len][i])
                y1_pre = -L1 * np.cos(self.data['theta1_pre'][st:st+len][i])
                x2_pre = x1_pre + L2 * np.sin(self.data['theta2_pre'][st:st+len][i])
                y2_pre = y1_pre - L2 * np.cos(self.data['theta2_pre'][st:st+len][i])
                
                line1_pre.set_data([0, x1_pre], [0, y1_pre])
                line2_pre.set_data([x1_pre, x2_pre], [y1_pre, y2_pre])
                
                return line1, line2, line1_pre, line2_pre
            
            # Create the animation
            anim = FuncAnimation(fig, update, frames=len, interval=interval, blit=True)
            
            # Show the animation
            plt.show()
            
            return anim


plottings = simulate_plot(labels, outputs)

plottings.plot_phase_space(st = 0, len=100)

plottings.plot_trajectory(st=1000, len=100)

plottings.plot_time_series(st=0, len = 1000)

anim = plottings.animate(len = 1000, interval = 1)

print(plottings.data['theta1'][0:10], plottings.data['theta1_pre'][0:10])

time_series_plot([i for i in range(len(theta1_original[:1000]))], theta1_original[:1000], theta1_predicted[:1000])


