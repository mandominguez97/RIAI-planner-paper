from geometry_msgs.msg import Pose
from .utils import generate_loiter_formation, bounds_from_cylinder, model_static_obstacles
from .multi_rrt_star_planner import MultiRRTStarPlanner
from .hungarian_tasks_planner import HungarianTasksPlanner
from .assignation_methods import AssignationMethods, RRTType
import numpy as np
from rclpy.task import Future
from concurrent.futures import ThreadPoolExecutor
import random


class Planner():
    def __init__(
            self,
            mission_frame: Pose,
            mission_radius: float,
            mission_height: float,
            n_vehicles: int,
            step_size: float,
            n_steps: int,
            space_coef: float,
            time_coef: float,
            avg_speed: float,
            spatial_tol: float,
            time_tol: float,
            cylinder_height: float,
            obstacle_radius: float
        ):
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._t_final = 100.0
        self._mission_frame = mission_frame
        self._step_size = step_size
        self._n_steps = n_steps
        self._space_coef = space_coef
        self._time_coef = time_coef
        self._limit = True
        self._bias_prob = .8
        self._avg_speed = avg_speed
        self._spatial_tol = spatial_tol
        self._time_tol = time_tol
        self._cylinder_height = cylinder_height
        self._obstacle_radius = obstacle_radius
        self._theta_gamma = 1.1
        self._mission_height = mission_height

        self._assigned_uavs = []
        self._goal_ids = []

        self._lower_limit, self._upper_limit = bounds_from_cylinder(
            mission_frame,
            mission_radius
        )
        
        loiter_center = mission_frame
        loiter_center.position.z = self._mission_height

        self._perception_trajectories = generate_loiter_formation(
            center=loiter_center,
            radius=mission_radius-2.0,
            n_drones=n_vehicles,
            n_points=200,
            speed=self._avg_speed
        )

        self.hungarian_planner = HungarianTasksPlanner()
        
        self.rtt_planner = MultiRRTStarPlanner(
            self._lower_limit,
            self._upper_limit,
            self._step_size,
            self._n_steps,
            self._space_coef,
            self._time_coef,
            self._theta_gamma,
        )


    def get_initial_trajectory(self, vehicle_poses, obstacles_poses):
        
        future = Future()

        def task():
            start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
            goal_poses = [self._perception_trajectories[n][0][0] for n in range(len(vehicle_poses))]
            goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
            
            self._assigned_uavs, self._goal_ids, trajectories = self.multi_rrt_star_plan(start_poses, goal_poses, model_static_obstacles(
                obstacles_poses,
                self._t_final,
                self._cylinder_height,
                self._obstacle_radius
            ))
            future.set_result((self._assigned_uavs, trajectories))

        self._executor.submit(task)
        return future
  

    def get_perception_trajectory(self):

        trajectories = [self._perception_trajectories[goal_id] for goal_id in self._goal_ids]
        return self._assigned_uavs, trajectories


    def get_tasks_planning(
        self, 
        vehicle_poses, 
        goal_poses, 
        obstacles_poses,
        plan_type
    ):  
        start_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, .0])) for n, p in enumerate(vehicle_poses)]
        goal_poses = [(f"{n}", np.array([p.position.x, p.position.y, p.position.z, self._t_final])) for n, p in enumerate(goal_poses)]
        strategy = None

        match plan_type:
            case AssignationMethods.RRT.value:
                strategy = self.multi_rrt_plan                
            case AssignationMethods.RRT_STAR.value:
                strategy = self.multi_rrt_star_plan    
            case AssignationMethods.RRT_STAR_HUNGARIAN.value:
                strategy = self.multi_rrt_hungarian_plan  
            case AssignationMethods.RANDOM.value:
                strategy = self.random_plan     
            case _:
                strategy = self.multi_rrt_star_plan

        future = Future()
        def task():
            assigned_uavs, goal_ids, trajectories = strategy(
                start_poses, 
                goal_poses,     
                model_static_obstacles(
                    obstacles_poses,
                    self._t_final,
                    self._cylinder_height,
                    self._obstacle_radius
                ))
            future.set_result((assigned_uavs, goal_ids, trajectories))
        self._executor.submit(task)
        return future


    def multi_rrt_hungarian_plan(self, start_poses, goal_poses, obstacles):
        
        dt = .5
        costs = [[None for _ in range(len(goal_poses))] for _ in range(len(start_poses))]

        while any(c is None for row in costs for c in row):
            results = self.rtt_planner.run_all_combinations(
                start_poses,        
                goal_poses,
                self._avg_speed,
                obstacles,
                self._bias_prob,
                self._limit,
                self._spatial_tol,
                self._time_tol
            )
            for id in results.keys(): 
                agent_idx = int(id.split('->')[0].strip())
                goal_idx = int(id.split('->')[1].strip())
                new_cost = results[id]["final_cost"]

                if costs[agent_idx][goal_idx] is None and new_cost is not None:
                    costs[agent_idx][goal_idx] = new_cost
            
        agent_idx, goal_idx = self.hungarian_planner.plan(costs)
        goal_poses_assigned = [goal_poses[goal_idx[i]][1] for i in range(len(agent_idx))]
        start_positions = [s[1] for s in start_poses]
        
        _, _, trajectories = self.rtt_planner.plan_paths(
            start_positions, 
            goal_poses_assigned,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._obstacle_radius
        )
        return agent_idx, goal_idx, trajectories


    def multi_rrt_plan(self, start_poses, goal_poses, obstacles):
        return self.rtt_planner.plan(
            start_poses,        
            goal_poses,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._cylinder_height,
            self._obstacle_radius,
            alg_type=RRTType.RRT.value
        )


    def random_plan(self, start_poses, goal_poses, obstacles):
        
        agent_idx = list(range(len(start_poses)))
        goal_idx = list(range(len(goal_poses)))
        
        num_assignments = min(len(start_poses), len(goal_poses))
        
        random.shuffle(goal_idx)
        
        agent_idx_assigned = agent_idx[:num_assignments]
        goal_idx_assigned = goal_idx[:num_assignments]
        
        goal_poses_assigned = [goal_poses[goal_idx_assigned[i]][1] for i in range(num_assignments)]
        start_positions = [start_poses[i][1] for i in range(num_assignments)]
        
        _, _, trajectories = self.rtt_planner.plan_paths(
            start_positions, 
            goal_poses_assigned,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._obstacle_radius
        )
        
        return agent_idx_assigned, goal_idx_assigned, trajectories


    def multi_rrt_star_plan(self, start_poses, goal_poses, obstacles):
        return self.rtt_planner.plan(
            start_poses,        
            goal_poses,
            self._avg_speed,
            obstacles,
            self._bias_prob,
            self._limit,
            self._spatial_tol,
            self._time_tol,
            self._cylinder_height,
            self._obstacle_radius
        )