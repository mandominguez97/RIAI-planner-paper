from .rtt_star_planner import RttStarPlanner, check_restrictions 
from .rtt_planner import RttPlanner
import numpy as np
from geometry_msgs.msg import Pose, Twist
from .utils import plot_trajectories
from planning.assignation_methods import RRTType


class MultiRRTStarPlanner():
    def __init__(
            self,
            lower_limit, 
            upper_limit,
            step_size, 
            n_steps,
            space_coef, 
            time_coef,
            theta_gamma = 1.1
        ):
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._step_size = step_size
        self._theta_gamma = theta_gamma
        self._n_steps = n_steps
        self._space_coef = space_coef
        self._time_coef = time_coef
        self._delta_t = .5


    def create_rrt_planner(
        self,
        lower_limit,
        upper_limit,
        step_size,
        n_steps,
        space_coef,
        time_coef,
        alg_type
    ):
        match alg_type:
            case RRTType.RRT.value:
                return RttPlanner(
                    lower_limit,
                    upper_limit,
                    step_size,
                    n_steps,
                    space_coef,
                    time_coef
                )
            case RRTType.RRT_STAR.value:
                return RttStarPlanner(
                    lower_limit,
                    upper_limit,
                    step_size,
                    n_steps,
                    space_coef,
                    time_coef
                )
            case _:
                return RttStarPlanner(
                    lower_limit,
                    upper_limit,
                    step_size,
                    n_steps,
                    space_coef,
                    time_coef
                )


    def plan_paths(
            self,
            start_poses, 
            goal_poses,
            speed, 
            obstacles,
            bias_prob, 
            limit,
            spatial_tol, 
            time_tol,
            obstacle_radius,
            alg_type = RRTType.RRT_STAR.value
        ):

        dt = .5
        assigned_uavs = []
        trajectories = []

        for agent_idx, (start_pose, goal_pose) in enumerate(zip(start_poses, goal_poses)):
            
            poses = []
            goal_node = None
            while goal_node is None:
                
                planner = self.create_rrt_planner(
                    self._lower_limit,
                    self._upper_limit,
                    self._step_size,
                    self._n_steps,
                    self._space_coef,
                    self._time_coef,
                    alg_type
                )
               
                goal_node, _, _ = planner.plan(
                    start_pose, goal_pose, 
                    speed, obstacles, 
                    bias_prob, limit, 
                    spatial_tol, time_tol
                )
           
            node = goal_node
            while(node._parent is not None):
                p = Pose()
                p.position.x = node._position[0]
                p.position.y = node._position[1]
                p.position.z = node._position[2]
                poses.append(p)
                node = node._parent

            dts = [dt*n for n in range(len(poses))]
            velocities = [None for _ in range(len(poses))]
            yaws = [float('nan') for _ in range(len(poses))]

            assigned_uavs.append(agent_idx)
            trajectories.append([
                poses[::-1],
                velocities,
                yaws,
                dts
            ])
            obstacles.append(
                self.path_to_obstacle(
                    goal_node, obstacle_radius)
            )

        return assigned_uavs, None, trajectories
            

    def run_all_combinations(
            self, 
            starts, 
            goals,      
            speed, 
            obstacles,
            bias_prob, 
            limit,
            spatial_tol, 
            time_tol,
            alg_type = RRTType.RRT_STAR.value
        ):
        
        results = {}

        for s_name, s in starts:
            for g_name, g in goals:

                planner = self.create_rrt_planner(
                    self._lower_limit,
                    self._upper_limit,
                    self._step_size,
                    self._n_steps,
                    self._space_coef,
                    self._time_coef,
                    alg_type
                )
                goal_node, tree, n_iterations = planner.plan(
                    s, g, 
                    speed, obstacles, 
                    bias_prob, limit, 
                    spatial_tol, time_tol
                )
                key = f"{s_name} -> {g_name}"
                if goal_node is not None:
                    final_cost = goal_node._cost
                    final_time = goal_node._position[3]
                else:
                    final_cost = None
                    final_time = None
                results[key] = {
                    "start": s,
                    "goal": g,
                    "goal_node": goal_node,
                    "tree": tree,
                    "n_iterations": n_iterations,
                    "final_cost": final_cost,
                    "final_time": final_time 
                }
                
        return results
    

    def path_to_obstacle(self, goal_node, radius):
        """
        Convierte el camino desde el start hasta goal_node
        en un "obstáculo dinámico" de radio `radius`.

        Devuelve un array de shape (N, 5):
        [x, y, z, t, r]
        """
        # Recorrer el path desde goal hasta el inicio
        path = []
        n = goal_node
        while n is not None:
            path.append(n._position.copy())
            n = n._parent

        # Invertimos para tenerlo en orden temporal creciente
        path.reverse()
        path = np.array(path)  # shape (N, 4) -> [x, y, z, t]

        xs = path[:, 0]
        ys = path[:, 1]
        zs = path[:, 2]
        ts = path[:, 3]
        rs = np.full_like(ts, radius)

        obstacle = np.column_stack([xs, ys, zs, ts, rs])
        return obstacle


    def choose_and_plan(
            self, 
            starts, 
            goals,
            speed, 
            obstacles,
            bias_prob, 
            limit,
            spatial_tol, 
            time_tol,
            obstacle_height,
            obstacle_radius,
            alg_type = RRTType.RRT_STAR.value
        ):
        """
        Devuelve best result que es un diccionario con esto por cada "Start X -> Goal Y" 

        {
                    "start": np.ndarray        # posición inicial 4D [x, y, z, t]
                    "goal": np.ndarray         # posición objetivo 4D [x, y, z, t]
                    "goal_node": Node          # nodo final alcanzado en el árbol
                    "tree": List[Node]         # todos los nodos generados para esa ruta
                    "n_iterations": int        # iteraciones realizadas
                    "final_cost": float | None # coste total del camino (si existe)
                    "final_time": float | None # tiempo final t del nodo objetivo
        }
        """
        remaining_starts = list(starts)
        remaining_goals = list(goals)
        best_results = {} 
        step = 1
  
        while remaining_starts and remaining_goals:
         
            import time
            start_time = time.time()
            
            current_results = self.run_all_combinations(
                remaining_starts, remaining_goals,
                speed, obstacles,
                bias_prob, limit,
                spatial_tol, time_tol,
                alg_type
            )

            elapsed = time.time() - start_time
            best_key = None
            best_cost = np.inf

            for key, data in current_results.items():
                cost = data["final_cost"]
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_key = key

            if best_key is None:
                break

            chosen = current_results[best_key]
            goal_node = chosen["goal_node"]
            pruned_goal = self.prune_path(goal_node, obstacles)

            chosen["goal_node"] = pruned_goal
            chosen["final_cost"] = pruned_goal._cost
            chosen["final_time"] = pruned_goal._position[3]

            best_results[best_key] = chosen
            if pruned_goal is not None:
                path_obstacle = self.path_to_obstacle(pruned_goal, obstacle_radius)
                obstacles.append(path_obstacle)

            start_name, goal_name = best_key.split(" -> ")

            remaining_starts = [s for s in remaining_starts if s[0] != start_name]
            remaining_goals  = [g for g in remaining_goals  if g[0] != goal_name]
            step += 1

        return best_results
    

    def prune_path(self, goal_node, obstacles):
        if goal_node is None or goal_node._parent is None or goal_node._parent._parent is None:
            return goal_node

        aux_node = goal_node
        while aux_node._parent._parent is not None:
            if check_restrictions(aux_node, aux_node._parent._parent, obstacles, self._upper_limit, self._delta_t):
                aux_node.set_parent(aux_node._parent._parent, self._space_coef, self._time_coef)
            else:
                aux_node = aux_node._parent
        
        
        return goal_node

    def plan(
        self, 
        starts, 
        goals,
        speed, 
        obstacles,
        bias_prob, 
        limit,
        spatial_tol, 
        time_tol,
        obstacle_height,
        obstacle_radius,
        alg_type = RRTType.RRT_STAR.value
    ):
        results = self.choose_and_plan(
            starts, goals,
            speed, obstacles,
            bias_prob, limit,
            spatial_tol, time_tol,
            obstacle_height, obstacle_radius,
            alg_type
        )
        dt = .5
        assigned_uav_ids = []
        goal_ids = []
        trajectories = []
        
        for id_str in results.keys():  
            positions = []
            node = results[id_str]["goal_node"]
            while(node._parent is not None): 
                p = Pose()
                p.position.x = node._position[0]
                p.position.y = node._position[1]
                p.position.z = node._position[2]
                positions.append(p)
                node = node._parent
            p = Pose()
            p.position.x = node._position[0]
            p.position.y = node._position[1]
            p.position.z = node._position[2]
            positions.append(p)
            
            positions = positions[::-1]
            dts = [dt*n for n in range(len(positions))]
            velocities = [Twist() for _ in range(len(positions))]
            yaws = [float('nan') for _ in range(len(positions))]
            
            assigned_uav_ids.append(int(id_str.split('->')[0].strip()))
            goal_ids.append(int(id_str.split('->')[1].strip()))
            trajectories.append([
                positions,
                velocities,
                yaws,
                dts
            ])

        return assigned_uav_ids, goal_ids, trajectories