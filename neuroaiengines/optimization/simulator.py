import time
from abc import ABC, abstractclassmethod, abstractmethod
from copy import copy
from typing import Iterable, Union
import numpy as np
from neuroaiengines.utils.angles import wrap_pi
from neuroaiengines.utils.transforms import *
from numpy import cos, pi, sin
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import lfilter


class Metric(ABC):
    def __init__(self):
        self._dts = []
    @abstractmethod
    def _record(self, action, state, observation, policy):
        ...

    def record(self,dt, action, state, observation, policy):
        if len(self._dts) == 0:
            self._dts.append(0)
        else:
            self._dts.append(self._dts[-1]+dt)
        self._record(action, state, observation, policy)
        
    @property
    @abstractmethod
    def result(self):
        raise NotImplementedError
    
class BasicSimulator(object):
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        
    def run(self, t : float, metrics : Union[Metric,Iterable[Metric]] , dt=0.001, initial_action=None, initial_state=None) -> Iterable:
        """
        Runs an experiment with metrics.

        :param t:
            The time in seconds to run the experiment
        :param  metrics:
            A metric or an iterable of metric objects.
        :param dt:
            Timestep
        :param initial_action:
            The initial action. Defaults to vector zero in the action space
        :param initial_state:
            The initial state. Defaults to vector zero in the state space


        :returns:
            An iterable of metric objects with data recorded
        """
        if isinstance(metrics, Metric):
            metrics = [metrics]
        T = np.linspace(0,t, int(t/dt)+1)
        action = initial_action or np.zeros(self.env.nA)
        state = initial_state or np.zeros(self.env.nS)
        observation = self.env.observe(0, state)
        env = self.env
        policy = self.policy
        for t in T:
            action = policy.step(dt, observation)
            state = env.dynamics(dt, action)
            observation = env.observe(dt, state)
            for metric in metrics:
                metric.record(dt, action, state, observation, policy)
        return [metric.result for metric in metrics]



def validate_action(func):
    """
    Decorator to validate that the action is of the right size
    """

    def validator(env, dt, action, *args, **kwargs):
        assert len(action.shape) == 1
        assert len(action) == env.nA
        return func(env, *args, dt=dt, action=action, **kwargs)

    return validator


def validate_state(func):
    """
    Decorator to validate that the state is of the right size
    """

    def validator(env, dt, state, *args, **kwargs):
        assert len(state.shape) == 1
        assert len(state) == env.nS
        return func(env, *args, dt=dt, state=state, **kwargs)

    return validator


DEFAULT_DT = 0.001


class Env(ABC):
    """
    A basic environment, without any state or observation dynamics. Don't use directly
    """

    def __init__(self, initial_state, action_size, renderer=None, **renderer_args):
        # State
        initial_state = np.array(initial_state, dtype=np.float32)
        assert len(initial_state.shape) == 1, "Initial state must be a single vector"
        self.state = initial_state
        self.nS = len(initial_state)
        self.nA = action_size
        if renderer is not None:
            self._renderer = renderer(self, **renderer_args)
            self._render = True
        self._t = None

    def step(self, t, action):
        """
        Steps the dynamics given an action, returns an observation of the state.
        """
        # Calculate the timestep
        if self._t is None:
            dt = DEFAULT_DT

        else:
            dt = t - self._t
        self._t = t
        state = self.dynamics(dt, action)
        observation = self.observe(dt, state)
        if self._render:
            self._renderer.render()
        return observation

    @abstractmethod
    def dynamics(self, dt, action):
        raise NotImplementedError

    @abstractmethod
    def observe(self, dt, state):
        raise NotImplementedError


class KinematicPoint(Env):
    """

    An environment that moves a point around an empty plane


    state:
    [x, y, theta, vx, vtheta]
    x : the x position of the agent
    y : the y position of the agent
    theta: the orientation of the agent ([0, 2*pi], counterclockwise is positive)
    vx: linear velocity along agent's forward axis
    vtheta: angular velocity
    
    observation:
    returns the full state

    action:
    [vx, vtheta]
    
    vx: the desired linear speed of the agent
    vtheta: the desired angular velocity of the agent
    

    """

    def __init__(self, x: float, y: float, theta: float, vx: float, vtheta: float, *args, **kwargs):
        """
        Creates a kinematic point environment. This is comparable to a two wheel robot operating in a single z plane.
        Controlled with [vx, vtheta].

        :param x,y: initial position
        :param theta: initial angle
        :param vx: initial forward velocity
        :param vtheta: initial angular velocity
        
        :returns: KinematicPoint environment
        
        """

        initial_state = [x, y, theta, vx, vtheta]
        
        # noise initialization
        self.process_noise = None
        self.input_noise = None
        self.observation_noise = None  
        if "process_noise" in kwargs.keys():
            self.process_noise = kwargs["process_noise"]
        if "input_noise" in kwargs.keys():
            self.input_noise = kwargs["input_noise"]  
        if "observation_noise" in kwargs.keys():
            self.observation_noise = kwargs["observation_noise"]  
            
        super().__init__(initial_state, action_size=2, *args, **kwargs)

    @validate_action
    def dynamics(self, dt: float, action: np.array):
        """
        Computes the dynamics (kinematics in this case) of the point

        :param dt: timestep
        :param action: action in form [vx, vtheta]

        :returns: The full state of the object
        """
        vx, vtheta = action
        
        if self.input_noise != None:
            vx = vx + self.process_noise.sample()
            vtheta = vtheta + self.process_noise.sample()

        theta = self.theta
        x = self.x
        y = self.y
        dx = vx * np.cos(self.theta)
        dy = vx * np.sin(self.theta)
        dtheta = vtheta

        self.state[0] = x + dx * dt
        self.state[1] = y + dy * dt
        self.state[2] = theta + dtheta * dt
        self.state[3] = vx
        self.state[4] = vtheta
        
        # add process noise
        if self.process_noise != None:
            self.state[0] = self.state[0] + self.process_noise.sample() * dt
            self.state[1] = self.state[1] + self.process_noise.sample() * dt
            self.state[2] = self.state[2] + self.process_noise.sample() * dt

        self.state[2] = wrap_pi(self.state[2])
        return self.state

    @validate_state
    def observe(self, dt, state):
        """
        The observation function based on the state. Can be overloaded!

        :param dt: timestep
        :param state: full state

        :returns observation: observation of the state
        """
        observation = state
        return observation

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def theta(self):
        return self.state[2]

    @property
    def vx(self):
        return self.state[3]

    @property
    def vtheta(self):
        return self.state[4]


class KinematicPointWithLandmarks(KinematicPoint):
    """
    An environment that moves a point around with some landmarks.

    state:
    [x, y, theta, vx, vtheta]
    x : the x position of the agent
    y : the y position of the agent
    theta: the orientation of the agent ([0, 2*pi], counterclockwise is positive)
    vx: linear velocity along agent's forward axis
    vtheta: angular velocity

    observation:
    [a1,a2...]
    the angles of the landmarks relative to the agent

    action:
    [vx, vtheta]
    
    vx: the desired linear speed of the agent
    vtheta: the desired angular velocity of the agent
    """

    def __init__(self, landmark_pos: list, *args, **kwargs):
        """
        
        :param landmark_pos: Nx2 array of landmark positions
        """
        super().__init__(*args, **kwargs)
        landmark_pos = np.array(landmark_pos)
        assert len(landmark_pos.shape) == 2, "Incorrect shape of landmark_pos"
        assert landmark_pos.shape[1] == 2, "Incorrect shape of landmark_pos"
        self.nL = landmark_pos.shape[0]
        self._landmark_pos = landmark_pos

    def get_landmark_angles(self, state):
        """
        :param dt: timestep
        :param state: full state
        """
        observation = np.zeros(self.nL + 1)
        x = state[0]
        y = state[1]

            
        theta = state[2]
        for i, (lpx, lpy) in enumerate(self._landmark_pos,1):
            angpos_noise = [0, 0]
            if (self.observation_noise != None):
                angpos_noise = [self.observation_noise.sample(),
                                self.observation_noise.sample()]
            ang = np.arctan2(lpy + angpos_noise[0] - y,
                             lpx + angpos_noise[1] - x) - theta           
            observation[i] = ang
            
        # adding angular velocity aspect of observation
        angvel_noise = 0
        if (self.observation_noise != None):
            vel_noise = self.observation_noise.sample()
        observation[0] = state[4] + angvel_noise
        
        return observation

    def observe(self, dt, state):
        obs = self.get_landmark_angles(state)
            
        return obs




class RecorderMetric(Metric):
    def __init__(self):
        super().__init__()
        self._data = []
        
    @property
    def result(self):
        ns = len(self._dts)
        nr = len(self._data[0])
        res = np.zeros((ns, nr+1))
        # append time sequence to data
        res[:,0] = self._dts
        res[:,1:] = self._data
        return res
    
class StatMetric(RecorderMetric):
    @abstractclassmethod
    def _calculate(self):
        " Used to calculate statistic based on accumulated data"
        return
    
    @property
    def result(self):
        res = self._calculate()
        return res
    
class StateRecorderMetric(RecorderMetric):
    def _record(self, action, state, observation, policy):
        #print(state)
        self._data.append(copy(state))  

class ActionRecorderMetric(RecorderMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy(action))
        
class ObservationRecorderMetric(RecorderMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy(observation))

class PolicyStateRecorderMetric(RecorderMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy(policy.get_state()))

# auxilliary metrics for debugging
class PolicyStateSquaredErrorRecorderMetric(RecorderMetric):
    def _record(self, action, state, observation, policy):
        #print(policy.get_state()[0]-copy(state[2]), policy.get_state()[0], state[2])
        self._data.append(copy([(policy.get_state()[0] - state[2])**2]))

class ClockTimeRecorderMetric(RecorderMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_time = time.time()
        
    def _record(self, action, state, observation, policy):
        self._data.append(copy([time.time() - self.cur_time]))
        self.cur_time = time.time()

class AngUpdateStatMetric(StatMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy([state[2]]))
    
    def _calculate(self):
        sout = np.array(self._data)[:,0]
        sout_unwrapped = np.unwrap(sout, discont = np.pi)
        dyaw_vals = np.diff(sout_unwrapped)
        return dyaw_vals

# metrics for baseline comparisons      
class RMSEStatMetric(StatMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy([policy.get_state()[0], state[2]]))
    
    def _calculate(self):
        pout, sout = [np.array(self._data)[:,0], np.array(self._data)[:,1]]
        sout_unwrapped = np.unwrap(np.ravel(sout), discont = np.pi)
        squared_out = (pout - sout_unwrapped) ** 2
        calc = np.sqrt(np.mean(squared_out))
        return calc

class RunTimeStatMetric(StatMetric):        
    def _record(self, action, state, observation, policy):
        self._data.append(copy([time.time()]))
    
    def _calculate(self):
        calc = np.ptp(np.ravel(self._data))
        return calc
    
class TrialLengthStatMetric(StatMetric):
    def _record(self, action, state, observation, policy):
        pass
    
    def _calculate(self):
        calc = np.ptp(np.ravel(self._dts))
        return calc        
  
class AngRangeStatMetric(StatMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy([state[2]]))
    
    def _calculate(self):
        calc = min(2*np.pi, np.ptp(np.unwrap(np.ravel(self._data), discont = np.pi)))
        #print(np.unwrap(np.ravel(self._data).tolist(), discont = np.pi))
        return calc
    
class NRMSEStatMetric(StatMetric):
    def _record(self, action, state, observation, policy):
        self._data.append(copy([policy.get_state()[0], state[2]]))
    
    def _calculate(self):
        # N-rmse = rmse/x_range
        # calculating numerator
        pout, sout = [np.array(self._data)[:,0], np.array(self._data)[:,1]]
        sout_unwrapped = np.unwrap(np.ravel(sout), discont = np.pi)
        squared_out = (pout - sout_unwrapped) ** 2
        num_calc = np.sqrt(np.mean(squared_out))
        # calculating denmoninator
        denom_calc = min(2*np.pi, np.ptp(np.unwrap(np.ravel(np.array(self._data)[:,1]), discont = np.pi)))
        return num_calc/denom_calc
    
        
class PolicyCovarianceRecorderMetric(RecorderMetric):        
    def _record(self, action, state, observation, policy):
        self._data.append(copy(policy.get_cov()))

 
class CurrentTrajectoryNode(object):
    """
        An environment node that generates a trajectory, keeps track of agent position, and generates ring neuron inputs at every timestep
    """
    def __init__(self, landmark_pos,n_landmarks, n_features, vtx=None, i_pos=None, i_theta=0, start_after=0.2, use_vel=True, trajectory_seed=0,  rn_bump_size=pi/18, total_rns=27,dropout_mask=None, **kwargs):
        """
        parameters:
        -----------
        landmark_pos: np.array((n_landmarks,2))
            the [x,y] positions of landmarks
        n_landmarks: int
            number of landmarks
        n_features: int
            number of features per landmark
        vtx: [float vt,float vx]
            angular/linear velocity override, for generating rotation only/circle trajectories
        i_pos: [float x, float y]
            initial position
        i_theta: float
            initial heading
        start_after: float
            time at the beginning of the run where only the EPGs get input (for bump stablization)
        use_vel: bool
            Use velocity (for landmark only runs)
        trajectory_seed: int
            seed for trajectory generation
        rn_bump_size: float
            fwhm of the RN bump
        total_rns: int
            total number of RNs per landmark
        dropout_mask: iterable<bool>
            a mask where each element is a timestep. If the mask is true at a timestep, the ring neuron activations will be 0

        
        """
        self.vx = 0
        # Need to be reset
        
        self.poss = []
        self.angles = []
        # Set this here, because if you put a default in the function definition, the object persists on redefinition
        if i_pos is None:
            self.pos = np.array([0.,0.])
            self.i_pos = np.array([0.,0.])
        else:
            self.pos = np.array(i_pos)
            self.i_pos = np.array(i_pos)
        self.num_steps = 0
        
        self.theta = i_theta
        self.i_theta = i_theta
        self.last_t = None
        self.landmark_pos = landmark_pos
        self.start = start_after
        self.use_vel = use_vel
        self.offset = 0
        self.seed = trajectory_seed
        self.rng = np.random.RandomState(self.seed)
        # Generate activation
        # TODO make this based on the length of the trial, or update if it gets larger than this
        # N timesteps
        T = 10000
        if dropout_mask is not None:
            self.dropout_mask = dropout_mask
        else:
            self.dropout_mask = np.zeros(T, dtype=np.bool)
        
        if vtx is None:
            # Trajectory generation based on von mises function
            
            vm = self.rng.vonmises(0, 100, T)
            self.rotation = lfilter([1.0], [1, -.5], vm)
            self.rotation = gaussian_filter1d(self.rotation * 100, sigma=100)*5
            x = np.linspace(0, 1, int(T / 50))
            y = self.rng.rand(int(T / 50)) * (.15)
            f = interp1d(x, y, kind='cubic')
            xnew = np.linspace(0, 1, T, endpoint=True)
            self.acceleration = f(xnew)*10.5**3
            
        else:
            # constant trajectory
            vt = vtx[0]
            vx = vtx[1]
            self.rotation = np.ones(T)*vt
            self.acceleration = np.ones(T)*vx
        self.n_landmarks = n_landmarks
        self.n_features = n_features
        # Create the RN activation fn
        self.rn_activation_fn = create_activation_fn(fwhm=rn_bump_size, num_neurons=total_rns)
        self.rn_slice = generate_middle_slice(total_rns, n_features)
        # Make slices for the output so everything is sync'd
        # Access different members of the step function to ensure output
        # ex:
        #   out = traj_node.step(t)
        #   vx = out[traj_node.velocity_slice]
        nrn = n_features*n_landmarks
        nepg = 18
        nrnepg = nrn+nepg
        self.size_out = nrnepg+2
        self.velocity_slice = slice(0,1)
        self.rn_activation_slice = slice(1,1+nrn)
        self.epg_activation_slice = slice(1+nrn, 1+nrnepg)
        self.linear_velocity_slice = slice(1+nrnepg,2+nrnepg)
        self.actual_angles = []
        ###
    def step(self,t):
        """
        t: float
            time
        """
        #a = angle wrt world
        #vt = angular velocity
        #vx = linear velocity
        #dt = time passed
        #keep track of how many steps so we draw from the distribution correctly
        self.num_steps += 1
        # Calculate dt
        if self.last_t is None:
            dt = 0.001
        else:
            dt = t - self.last_t
        self.last_t = t
        # Calculate time after initialization period
        ot = t - self.start
        ot = ot if ot > 0 else 0
        # Approximate the number of steps into the trajectory
        num_steps = int((ot-self.offset)*1000)
        generate_activation = self.rn_activation_fn
        output = np.zeros(self.size_out)
        # if we've started yet
        if (t - self.offset) >= self.start:
            
            vt = self.rotation[num_steps]
            self.vx = self.acceleration[num_steps]
            
            #get position based on linear velocity
            self.pos[0] += cos(self.theta)*self.vx*dt
            self.pos[1] += sin(self.theta)*self.vx*dt


            #get new angle wrt world based on angular velocity
            self.theta += vt*dt
            
            a = wrap_pi(self.theta)
            angles = []
            if self.use_vel:
                output[self.velocity_slice] = vt
            for lpos in self.landmark_pos:
             
                angle = angle_wrt_home(self.pos, a, lpos)
                angles.append(angle)
            if not self.dropout_mask[num_steps]:

                output[self.rn_activation_slice] = generate_activation(angles, slc=self.rn_slice).ravel()
            self.actual_angles.append(self.theta)
        else:
            # Still in initialization period
            a = wrap_pi(self.theta)
            
            angles = []
            for lpos in self.landmark_pos:
                
                

                angle = angle_wrt_home(self.pos, a, lpos)
                angles.append(angle)
            output[self.epg_activation_slice] = get_epg_activation(self.theta).ravel()
        output[self.linear_velocity_slice] = self.vx
        self.poss.append(self.pos.copy())
        self.angles.append(self.theta)
        return output
    def reset(self, time=0):
        self.poss = []
        self.angles = []
        self.pos = self.i_pos.copy()
        self.vx = 0
        self.num_steps = 0
        self.theta = self.i_theta
        self.last_t = None
        self.offset = time
