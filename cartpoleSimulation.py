
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
import matplotlib.patches as p

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
mc = 5.0  # mass of cart
mp = 1.0  # mass of pendulum
t_stop = 20  # how many seconds to simulate
history_len = 500  # how many trajectory points to display
    

def derivs(state, t):
    
    x, v, theta, omega = state
    dstatedt = np.zeros(4)
    dstatedt[0] = v
    F = 1000*(np.pi - theta) - 10*omega - 10*x
    dstatedt[1] = (F + mp*np.sin(theta)*(L*omega**2 + G*np.cos(theta)))/(mc+mp*(np.sin(theta))**2)
    dstatedt[2] = omega
    dstatedt[3] = (-1*F*np.cos(theta) - mp*L*(omega**2)*np.cos(theta)*np.sin(theta) - (mc + mp)*G*np.sin(theta))/(L*(mc + mp*(np.sin(theta))**2))

    return dstatedt


# create a time array from 0..t_stop sampled at 0.1 second steps
dt = 0.02
t = np.arange(0, t_stop, dt)

x = 0
v = 0
theta = np.pi+0.1
omega = 0

#initial state
state = np.array([x, v, theta, omega])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)
print(y.shape)


#visuals
fig = plt.figure(figsize=(12, 8)) #window size
ax = fig.add_subplot(autoscale_on=False, xlim=(-2*L1, 2*L1), ylim=(-2*L1, 2*1.)) #plot size
ax.set_aspect('equal')
ax.grid()


cartpos_list = y[:,0]
theta_list = y[:,2]

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], ',-', lw=1)
rectangle = p.Rectangle((0, 0), 0, 0, color = "orange", lw = 2)
plt.gca().add_patch(rectangle)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    
    thisx = [cartpos_list[i], L*cos(theta_list[i]-np.pi/2)+cartpos_list[i]]
    thisy = [0, L*sin(theta_list[i]-np.pi/2)]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[1])#here
    history_y.appendleft(thisy[1])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    rectangle.set_width(2)
    rectangle.set_height(1)
    rectangle.set_xy([cartpos_list[i]-1, -1])
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text, rectangle


ani = animation.FuncAnimation(
    fig, animate, len(theta_list), interval=dt*100, blit=True) #here
plt.show()
