import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART
import matplotlib.pyplot as plt

import os
import copy
import tqdm
import torch
import src.torch.pytorch_util as ptu

from src.envs.hexapod_dart.controllers.sin_controller import SinusoidController
from src.envs.hexapod_dart.controllers.mb_sin_controller import MBSinusoidController

# To record entire trajectory of a particular state/observation
# In this case - this will be the behvaiuoral descriptor
class DutyFactor(rd.Descriptor):
    def __init__(self,desc):
        rd.Descriptor.__init__(self,desc)
        self._states = []
        
    def __call__(self):
        if(self._simu.num_robots()>0):
            self._states.append(self._simu.robot(0).positions())
            
            
class HexapodEnv:
    
    def __init__(self, dynamics_model,
                 render=False,
                 record_state_action=False,
                 ctrl_freq=100,
                 n_waypoints=1):

        self.render = render
        self.record_state_action = record_state_action
        self.urdf_path = "src/envs/hexapod_dart/robot_model/"
        self.ctrl_freq = ctrl_freq # in Hz
        self.sim_dt = 0.001

        self.n_wps = n_waypoints
        
        self.dynamics_model = dynamics_model

    def update_dynamics_model(dynamics_model):
        self.dynamics_model = dynamics_model
        return 1
        
    def init_robot(self):
        # Load robot urdf and intialize robot
        import src, os
        path_to_src = src.__path__[0]
        split_path = os.path.split(path_to_src)
        
        abs_path_to_urdf = os.path.join(split_path[0], self.urdf_path)
        
        # print('paths ', path_to_src, self.urdf_path, abs_path_to_urdf)
        # robot = rd.Robot(self.urdf_path+"hexapod_v2.urdf", "hexapod", False)
        robot = rd.Robot(abs_path_to_urdf+"hexapod_v2.urdf", "hexapod", False)
        robot.free_from_world([0,0,0.0,0,0,0.15]) # place robot slightly above the floor
        robot.set_actuator_types('servo') # THIS IS IMPORTANT
        robot.set_position_enforced(True)
        #print("actuator types: ",robot.actuator_types()) # default was torque
        return robot
    
    def simulate(self, ctrl, sim_time, robot, render=None, video_name=None):

        if render is None:
            render = self.render
            
        # clone robot
        grobot = robot.clone()
            
        #initialize controllers
        ctrl_freq = self.ctrl_freq # 100 Hz
        sinusoid_controller = SinusoidController(ctrl, False, ctrl_freq)
        stationary_controller = SinusoidController(np.zeros(36), False, ctrl_freq)

        # Create simulator object
        simu = rd.RobotDARTSimu(self.sim_dt) # 1000 Hz simulation freq - 0.001 dt
        simu.set_control_freq(ctrl_freq) # set control frequency of 100 Hz
        simu.set_collision_detector("bullet") # DART Collision Detector is the default but does not support cylinder shapes                                             
        #print("SIMULATION TIMESTEP",simu.timestep())
        
        # create graphics                                                          
        if render:
            print("INSIDE RENDER TRUE")
            graphics = rd.gui.Graphics(simu, rd.gui.GraphicsConfiguration())
            #graphics.look_at([0.,0.,5.],[0.,0.,0.], [0.,0.,1.]) # camera_pos, look_at, up_vector
            simu.set_graphics(graphics)

            if video_name is not None: 
                camera = rd.sensor.Camera(simu, graphics.magnum_app(), graphics.width(), graphics.height(), 30)
                simu.add_sensor(camera)
                simu.run(0.01)
                
                camera.camera().record(True, True)
                camera.record_video(video_name)
                camera.look_at([0.1,1.4,1.3], [0.1,0,0.])

            
        # add the robot and the floor
        simu.add_robot(grobot)
        #floor = simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")
        simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")

        # ADD CONTROLLERS TO ROBOT ONLY AFTER ADDING ROBOT TO SIMULATION
        # add stationary controller first to stabilise robot from drop
        grobot.add_controller(stationary_controller, 1.)
        stationary_controller.configure()
        
        # Add controller to the robot   
        grobot.add_controller(sinusoid_controller, 1.)
        sinusoid_controller.configure() # to set the ctrl parameters
        sinusoid_controller.activate(False)

        # set friction parameters - only works with latest release of robot_dart
        #mu = 0.7
        #grobot.set_friction_coeffs(mu)
        #floor.set_friction_coeffs(mu)
        
        simu.run(1.0)
        
        # Change control mode from stabalise to walking
        stationary_controller.activate(False)
        sinusoid_controller.activate(True)
        
        # run
        simu.run(sim_time)

        #print("DONE SIMU")
        final_pos = grobot.positions() # [rx,ry,rz,x,y,z,joint_pos]

        grobot.reset()
        
        states_recorded = np.array(sinusoid_controller.states)
        actions_recorded = np.array(sinusoid_controller.actions)

        #print("States recorded: ", states_recorded.shape)
        #print("Action recorded: ", actions_recorded.shape)
        
        return final_pos, states_recorded, actions_recorded 


    def desired_angle(self, in_x,in_y, batch=False):
        '''
        Compute desired yaw angle (rotationabotu z axis) of the robot given final position x-y
        '''
        x = in_x
        y = in_y
            
        B = np.sqrt((x/2)**2 + (y/2)**2)
        alpha = np.arctan2(y,x)
        A = B/np.cos(alpha)
        beta = np.arctan2(y,x-A)

        if batch:
            for i in range(x.shape[0]):
                if x[i] < 0:
                    beta[i] = beta[i] - np.pi
                while beta[i] < -np.pi:
                    beta[i] = beta[i] + 2*np.pi
                while beta[i] > np.pi:
                    beta[i] = beta[i] - 2*np.pi
        else:
            if x < 0:
                beta = beta - np.pi
            # if angles out of 360 range, bring it back in
            while beta < -np.pi:
                beta = beta + 2*np.pi
            while beta > np.pi:
                beta = beta - 2*np.pi
        return beta

    def angle_dist(self, a,b, batch=False):
        dist = b-a
        if batch:
            for i in range(a.shape[0]):
                while dist[i] < -np.pi:
                    dist[i] = dist[i] + 2*np.pi
                while dist[i] > np.pi:
                    dist[i] = dist[i] - 2*np.pi
        # if the angles out of the 360 range, bring it back in
        else:
            import os
            import math
            if dist + 2*np.pi < -np.pi:
                dist += abs(math.floor(dist/(2*np.pi)))*2*np.pi
            if dist - 2*np.pi > np.pi:
                dist -= abs(math.floor(dist/(2*np.pi)))*2*np.pi
            while dist < -np.pi:
                dist = dist + 2*np.pi
            while dist > np.pi:
                dist = dist - 2*np.pi
                
        return dist 

    def compute_bd(self, obs_traj, ensemble=False, mean=True):
        bd = [0]*2*self.n_wps

        wp_idxs = [i for i in range(len(obs_traj)//self.n_wps, len(obs_traj),
                                    len(obs_traj)//self.n_wps)][:self.n_wps-1]
        wp_idxs += [-1]

        obs_wps = np.take(obs_traj, wp_idxs, axis=0)
        obs_inds = [3, 4]
        if ensemble and not mean:
            bd = obs_wps[:,:,obs_inds].flatten()
        else:
            bd = obs_wps[:,obs_inds].flatten()

        offset = 1.5 # in this case the offset is the saem for both x and y descriptors
        fullmap_size = 3 # 8m for full map size 
        bd = (bd + offset)/fullmap_size
        
        return bd
    
    def evaluate_solution(self, ctrl, render=False):

        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate(ctrl, sim_time, robot, render)

        #--------Compute BD (final x-y pos)-----------#
        x_pos = final_pos[3]
        y_pos = final_pos[4]
        # normalize BD
        offset = 1.5 # in this case the offset is the saem for both x and y descriptors
        fullmap_size = 3 # 8m for full map size 
        x_desc = (x_pos + offset)/fullmap_size
        y_desc = (y_pos + offset)/fullmap_size
    
        desc = [[x_desc,y_desc]]
        
        last_s = np.empty((1, s_record.shape[1]))
        last_s[0,:len(final_pos)] = final_pos
        full_traj = np.vstack((s_record, last_s))
        desc = self.compute_bd(full_traj)
        #------------Compute Fitness-------------------#
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric

        if render:
            print("Desc from simulation", desc) 
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None

        return fitness, desc, obs_traj, act_traj, 0 # 0 is disagr

    # for NN timestep model
    def simulate_model(self, ctrl, sim_time, mean, det):
        states_recorded = []
        actions_recorded = []

        #controller = MBSinusoidController(ctrl, False, self.ctrl_freq)
        controller = SinusoidController(ctrl, False, self.ctrl_freq)
        controller.configure()

        # initial state - everything zero except z position (have to make it correspond to what the model is trained on) 
        state = np.zeros(48)
        state[5] = -0.014 # robot com height when feet on ground is 0.136m 

        for t in np.arange(0.0, sim_time, 1/self.ctrl_freq):
            action = controller.commanded_jointpos(t)
            states_recorded.append(state)
            actions_recorded.append(action)        
            s = ptu.from_numpy(state)
            a = ptu.from_numpy(action)
            s = s.view(1,-1)
            a = a.view(1,-1)

            if det:
                # if deterministic dynamics model
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1))
            else:
                # if probalistic dynamics model - choose output mean or sample
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1), mean=mean)
                
            #print(state.shape)
            #print(pred_delta_ns)            
            state = pred_delta_ns[0] + state # the [0] just seelect the row [1,state_dim]

        final_pos = state 
        states_recorded = np.array(states_recorded)
        actions_recorded = np.array(actions_recorded)

        return final_pos, states_recorded, actions_recorded
    
    def evaluate_solution_model(self, ctrl, mean=False, det=True):

        sim_time = 3.0
        final_pos, states_rec, actions_rec = self.simulate_model(ctrl, sim_time, mean, det)
        s_record = states_rec
        #--------Compute BD (final x-y pos)-----------#
        x_pos = final_pos[3]
        y_pos = final_pos[4]
        # normalize BD
        offset = 1.5 # in this case the offset is the saem for both x and y descriptors
        fullmap_size = 3 # 8m for full map size 
        x_desc = (x_pos + offset)/fullmap_size
        y_desc = (y_pos + offset)/fullmap_size
    
        desc = [[x_desc,y_desc]]

        last_s = np.empty((1, s_record.shape[1]))
        last_s[0,:len(final_pos)] = final_pos
        full_traj = np.vstack((s_record, last_s))
        desc = self.compute_bd(full_traj)
        #------------Compute Fitness-------------------#
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric
        obs_traj = states_rec
        act_traj = actions_rec

        disagr = 0
        
        return fitness, desc, obs_traj, act_traj, disagr
    
    def simulate_model_ensemble(self, ctrl, sim_time, mean, disagr):
        states_recorded = []
        actions_recorded = []
        model_disagr = []
        
        #controller = MBSinusoidController(ctrl, False, self.ctrl_freq)
        controller = SinusoidController(ctrl, False, self.ctrl_freq)
        controller.configure()
        # initial state
        # for this ensembles - the states need to be fed in the form of ensemble_size
        # state and actions fed in as [ensemble_size, dim_size]
        # ts expand and flatten takes care of the num particles/
        state = np.zeros(48)
        state[5] = -0.014 # robot com height when feet on ground is 0.136m 
        state = np.tile(state,(self.dynamics_model.ensemble_size, 1))
        
        for t in np.arange(0.0, sim_time, 1/self.ctrl_freq):
            action = controller.commanded_jointpos(t)
            states_recorded.append(state)
            actions_recorded.append(action)
            s = ptu.from_numpy(state)
            a = ptu.from_numpy(action)

            a = a.repeat(self.dynamics_model.ensemble_size,1)
            #print("s shape: ", state.shape)
            #print("a shape:", a.shape)
            
            # if probalistic dynamics model - choose output mean or sample
            if disagr:
                pred_delta_ns, _ = self.dynamics_model.sample_with_disagreement(torch.cat((self.dynamics_model._expand_to_ts_form(s), self.dynamics_model._expand_to_ts_form(a)), dim=-1))
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                disagreement = self.compute_abs_disagreement(state, pred_delta_ns)
                #print("Disagreement: ", disagreement.shape)
                disagreement = ptu.get_numpy(disagreement) 
                #disagreement = ptu.get_numpy(disagreement[0,3]) 
                #disagreement = ptu.get_numpy(torch.mean(disagreement)) 
                model_disagr.append(disagreement)
                
            else:
                pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s,a, mean=mean)
                
            #print("Samples: ", pred_delta_ns.shape)
            #print(state.shape)
            #print(pred_delta_ns.shape)            
            state = pred_delta_ns + state 

        import pdb; pdb.set_trace()
        final_pos = state[0] # for now just pick one model - but you have all models here
        states_recorded = np.array(states_recorded)
        actions_recorded = np.array(actions_recorded)
        model_disagr = np.array(model_disagr)
        
        return final_pos, states_recorded, actions_recorded, model_disagr

    
    def compute_abs_disagreement(self, cur_state, pred_delta_ns):
        '''
        Computes absolute state dsiagreement between models in the ensemble
        cur state is [4,48]
        pred delta ns [4,48]
        '''
        next_state = pred_delta_ns + cur_state
        next_state = ptu.from_numpy(next_state)
        mean = next_state

        sample=False
        if sample: 
            inds = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0]) 
        else:
            inds = torch.tensor(np.array([0,0,0,1,1,2]))
            inds_b = torch.tensor(np.array([1,2,3,2,3,3]))

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, mean.shape[1])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
        inds_b = inds_b.repeat(1, mean.shape[1])

        means_a = (inds == 0).float() * mean[0]
        means_b = (inds_b == 0).float() * mean[0]
        for i in range(1, mean.shape[0]):
            means_a += (inds == i).float() * mean[i]
            means_b += (inds_b == i).float() * mean[i]
            
        disagreements = torch.mean(torch.sqrt((means_a - means_b)**2), dim=-2, keepdim=True)
        #disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

        return disagreements
    
    def evaluate_solution_model_ensemble(self, ctrl, mean=True, disagreement=True):
        torch.set_num_threads(1)
        sim_time = 3.0
        final_pos, states_rec, actions_rec, disagr = self.simulate_model_ensemble(ctrl, sim_time, mean, disagreement)
        s_record = states_rec[:,0,:]
        #--------Compute BD (final x-y pos)-----------#
        x_pos = final_pos[3]
        y_pos = final_pos[4]
        
        # normalize BD
        offset = 1.5 # in this case the offset is the saem for both x and y descriptors
        fullmap_size = 3 # 8m for full map size 
        x_desc = (x_pos + offset)/fullmap_size
        y_desc = (y_pos + offset)/fullmap_size
    
        desc = [[x_desc,y_desc]]

        last_s = np.empty((1, s_record.shape[1]))
        last_s[0,:len(final_pos)] = final_pos
        full_traj = np.vstack((s_record, last_s))
        desc = self.compute_bd(full_traj)
        #------------Compute Fitness-------------------#
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric

        obs_traj = states_rec
        act_traj = actions_rec

        #------------ Absolute disagreement --------------#
        # disagr is the abs disagreement trajectory for each dimension [300,1,48]
        # can also save the entire disagreement trajectory - but we will take final mean dis
        final_disagr = np.mean(disagr[-1,0,:])
        
        if disagreement:
            return fitness, desc, obs_traj, act_traj, final_disagr
        else: 
            return fitness, desc, obs_traj, act_traj

    def simulate_model_ensemble_all(self, ctrls, sim_time, mean, disagr):
        states_recorded = []
        actions_recorded = []
        model_disagrs = []
        controllers = []

        for ctrl in ctrls:
            ## Controllers
            controllers.append(SinusoidController(ctrl, False, self.ctrl_freq))
            controllers[-1].configure()
            ## Traj elements
            states_recorded.append([])
            actions_recorded.append([])
            model_disagrs.append([])
            
        # initial state
        # for this ensembles - the states need to be fed in the form of ensemble_size
        # state and actions fed in as [ensemble_size, dim_size]
        # ts expand and flatten takes care of the num particles/
        ens_size = self.dynamics_model.ensemble_size

        state = np.zeros(48)
        state[5] = -0.014 # robot com height when feet on ground is 0.136m 

        act_dim = 18

        S = np.tile(state, (ens_size*len(ctrls), 1))
        A = np.empty((ens_size*len(ctrls),
                      act_dim))

        # for t in np.arange(0.0, sim_time, 1/self.ctrl_freq):
        arange = np.arange(0.0, sim_time, 1/self.ctrl_freq)
        for t in tqdm.tqdm(arange, total=len(arange)):
            for i in range(len(ctrls)):
                A[i*ens_size:i*ens_size+ens_size] = \
                    np.tile(controllers[i].commanded_jointpos(t), (ens_size, 1))

            batch_pred_delta_ns, batch_disagreement = self.forward(A, S, mean=False,
                                                                   disagr=disagr,
                                                                   multiple=True)
            for i in range(len(ctrls)):
                ## Don't use mean predictions and keep each particule trajectory
                # Be careful, in that case there is no need to repeat each state in
                # forward multiple function
                disagreement = self.compute_abs_disagreement(S[i*ens_size:i*ens_size+ens_size],
                                                             batch_pred_delta_ns[i])
                disagreement = ptu.get_numpy(disagreement)

                model_disagrs[i].append(disagreement.copy())

                S[i*ens_size:i*ens_size+ens_size] += batch_pred_delta_ns[i]
                states_recorded[i].append(S[i*ens_size:i*ens_size+ens_size].copy())

                actions_recorded[i].append(A[i*ens_size:i*ens_size+ens_size])


        final_poses = []
        for i in range(len(ctrls)):
            final_poses.append(S[i*ens_size:i*ens_size+ens_size][0])
            
            states_recorded[i] = np.array(states_recorded[i])
            actions_recorded[i] = np.array(actions_recorded[i])
            model_disagrs[i] = np.array(model_disagrs[i])
        
        return final_poses, states_recorded, actions_recorded, model_disagrs

    def evaluate_solution_model_ensemble_all(self, ctrls, mean=True, disagreement=True):
        sim_time = 3.0
        final_poses, states_recs, actions_recs, disagrs = \
            self.simulate_model_ensemble_all(
                ctrls,
                sim_time,
                mean,
                disagreement
            )
        
        fit_list = []
        bd_list = []
        obs_trajs = []
        act_trajs = []
        disagr_trajs = []

        for i in range(len(ctrls)):
            s_record = states_recs[i][:,0,:]
            #--------Compute BD (final x-y pos)-----------#
            x_pos = final_poses[i][3]
            y_pos = final_poses[i][4]

            # normalize BD
            offset = 1.5 # in this case the offset is the saem for both x and y descriptors
            fullmap_size = 3 # 8m for full map size 
            x_desc = (x_pos + offset)/fullmap_size
            y_desc = (y_pos + offset)/fullmap_size

            desc = [[x_desc,y_desc]]

            last_s = np.empty((1, s_record.shape[1]))
            last_s[0,:len(final_poses[i])] = final_poses[i]
            full_traj = np.vstack((s_record, last_s))
            desc = self.compute_bd(full_traj)
            #------------Compute Fitness-------------------#
            beta = self.desired_angle(x_pos, y_pos)
            final_rot_z = final_poses[i][2] # final yaw angle of the robot 
            dist_metric = abs(self.angle_dist(beta, final_rot_z))

            fitness = -dist_metric

            obs_traj = states_recs[i]
            act_traj = actions_recs[i]

            #------------ Absolute disagreement --------------#
            # disagr is the abs disagreement trajectory for each dimension [300,1,48]
            # can also save the entire disagreement trajectory - but we will take final mean dis
            final_disagr = np.mean(disagrs[i][-1,0,:])

            fit_list.append(fitness)
            bd_list.append(desc)
            obs_trajs.append(obs_traj)
            act_trajs.append(act_traj)
            disagr_trajs.append(final_disagr)

        return fit_lst, bd_list, obs_trajs, act_trajs, disagr_trajs

    def forward_multiple(self, A, S, mean=True, disagr=True, ensemble=True, det_ens=False):
        ## Takes a list of actions A and a list of states S we want to query the model from
        ## Returns a list of the return of a forward call for each couple (action, state)
        assert len(A) == len(S)
        batch_len = len(A)
        if ensemble:
            ens_size = self.dynamics_model.ensemble_size
        else:
            ens_size = 1
        S_0 = np.empty((batch_len*ens_size, S.shape[1]))
        A_0 = np.empty((batch_len*ens_size, A.shape[1]))

        batch_cpt = 0
        for a, s in zip(A, S):
            S_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(s,(ens_size, 1))

            A_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(a,(ens_size, 1))
            batch_cpt += 1
        if ensemble:
            return self.forward(A_0, S_0, mean=mean, disagr=disagr, multiple=True)
        elif det_ens:
            s_0 = copy.deepcopy(S_0)
            a_0 = copy.deepcopy(A_0)
            s_0 = ptu.from_numpy(s_0)
            a_0 = ptu.from_numpy(a_0)
            return self.dynamics_model.output_pred_with_ts(
                    torch.cat((s_0, a_0), dim=-1),
                    mean=mean), [0]*len(s_0)
        else:
            s_0 = copy.deepcopy(S_0)
            a_0 = copy.deepcopy(A_0)
            s_0 = ptu.from_numpy(s_0)
            a_0 = ptu.from_numpy(a_0)
            return self.dynamics_model.output_pred(
                    torch.cat((s_0, a_0), dim=-1),
                    mean=mean), [0]*len(s_0)

    def forward(self, a, s, mean=True, disagr=True, multiple=False):
        s_0 = copy.deepcopy(s)
        a_0 = copy.deepcopy(a)

        if not multiple:
            s_0 = np.tile(s_0,(self.dynamics_model.ensemble_size, 1))
            a_0 = np.tile(a_0,(self.dynamics_model.ensemble_size, 1))

        s_0 = ptu.from_numpy(s_0)
        a_0 = ptu.from_numpy(a_0)

        # a_0 = a_0.repeat(self._dynamics_model.ensemble_size,1)

        # if probalistic dynamics model - choose output mean or sample
        if disagr:
            if not multiple:
                pred_delta_ns, disagreement = self.dynamics_model.sample_with_disagreement(
                    torch.cat((
                        self.dynamics_model._expand_to_ts_form(s_0),
                        self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ))#, disagreement_type="mean" if mean else "var")
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                return pred_delta_ns, disagreement
            else:
                pred_delta_ns_list, disagreement_list = \
                self.dynamics_model.sample_with_disagreement_multiple(
                    torch.cat((
                        self.dynamics_model._expand_to_ts_form(s_0),
                        self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ))#, disagreement_type="mean" if mean else "var")
                for i in range(len(pred_delta_ns_list)):
                    pred_delta_ns_list[i] = ptu.get_numpy(pred_delta_ns_list[i])
                return pred_delta_ns_list, disagreement_list
        else:
            pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
        return pred_delta_ns, 0

    def evaluate_solution_uni(self, ctrl, render=False, video_name=None):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate(ctrl, sim_time, robot, render, video_name)
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]

        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        if render:
            print("Desc from simulation", desc)
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None
    
        return fitness, desc, obs_traj, act_traj

    def evaluate_solution_model_uni(self, ctrl, render=False, mean=False, det=True):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate_model(ctrl, sim_time, mean, det)
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]
        
        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        #print(desc)
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        if render:
            print("Desc from simulation", desc) 
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None
    
        return fitness, desc, obs_traj, act_traj

    
    def evaluate_solution_model_ensemble_uni(self, ctrl, render=False, mean=True, disagreement=True):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        torch.set_num_threads(1)
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record, disagr = self.simulate_model_ensemble(ctrl, sim_time, mean, disagreement)

        #print("s record shape: ", s_record.shape)
        #print("a record shape: ", a_record.shape)
        s_record = s_record[:,0,:]
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]
        
        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        #print(desc)
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        #------------ Absolute disagreement --------------#
        # disagr is the abs disagreement trajectory for each dimension [300,1,48]
        # can also save the entire disagreement trajectory - but we will take final mean dis
        final_disagr = np.mean(disagr[-1,0,:])

        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None

        if disagreement:
            return fitness, desc, obs_traj, act_traj, final_disagr
        else: 
            return fitness, desc, obs_traj, act_traj














    

        
def plot_state_comparison(state_traj, state_traj_m):

    total_t = state_traj.shape[0]
    #total_t = 30
    
    for i in np.arange(3,4):
        traj_real = state_traj[:,i]
        traj_m = state_traj_m[:,i]
        plt.plot(np.arange(total_t), traj_real[:total_t], "-",label="Ground truth "+str(i))
        plt.plot(np.arange(total_t), traj_m[:total_t], '--', label="Dynamics Model "+str(i))

    return 1

if __name__ == "__main__":

    # initialize environment class
    from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
    from src.models.dynamics_models.deterministic_model import DeterministicDynModel

    variant = dict(
        mbrl_kwargs=dict(
            ensemble_size=4,
            layer_size=500,
            learning_rate=1e-3,
            batch_size=256,
        )
    )

    obs_dim = 48
    action_dim = 18
    M = variant['mbrl_kwargs']['layer_size']

    # initialize dynamics model
    prob_dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M]
    )

    # initialize dynamics model
    det_dynamics_model = DeterministicDynModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=500
    )

    model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_finalarch.pth"
    ptu.load_model(prob_dynamics_model, model_path)
    
    env = HexapodEnv(prob_dynamics_model, render=False, record_state_action=True)

    # test simulation with a random controller
    ctrl = np.random.uniform(0,1,size=36)

    fit, desc, obs_traj, act_traj = env.evaluate_solution(ctrl, render=False)
    fit_m, desc_m, obs_traj_m, act_traj_m = env.evaluate_solution_model_ensemble(ctrl, det=False, mean=True)
    print("Ground Truth: ", fit, desc)
    print("Probablistic Model: ", fit_m, desc_m)

    print(obs_traj.shape)
    print(obs_traj_m.shape)
    plot_state_comparison(obs_traj, obs_traj_m[:,0,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,1,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,2,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,3,:])

    #plt.show()
