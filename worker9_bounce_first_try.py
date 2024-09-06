import queue
import time
import numpy as np
import multiprocessing as mp
from puck import Puck
from puck_server import Puck_Server
import sympy as sp


V_MIN = 10.0
V_MAX = 42.0
A_MAX = 100.0
RADIUS = 1.0


Pucks = {}



def add_or_update_puck(puck):
    puck_id = puck.get_id()
    
    if not puck.is_alive():
        print(f"Puck with id {puck_id} is not alive. Deleting from the dictionary if exists.")
        delete_puck(puck_id)
        return

    updated_data = {
        'id': puck.get_id(),
        'name': puck.get_name(),
        'position': puck.get_position(),
        'velocity': puck.get_velocity(),
        'acceleration': puck.get_acceleration(),
        'timestamp': puck.get_time(),
        'fuel': puck.get_fuel(),
        'alive': puck.is_alive()
    }

    if puck_id in Pucks:
        Pucks[puck_id].update(updated_data)
    else:
        Pucks[puck_id] = {
            **updated_data,
            'proximity_traffic': False,
            'tca': None,
            'Dtca': None
        }

def delete_puck(puck_id):
    if puck_id in Pucks:
        del Pucks[puck_id]
        
def print_pucks():
    for puck_id, puck_data in Pucks.items():
        print(f"Puck ID {puck_id}: {puck_data}")

def is_proximity(puck_id, own_id):
    if puck_id == own_id:
        return
    distance_threshold=3.0
    own_puck = Pucks[own_id]
    S1 = own_puck['position']
    V1 = own_puck['velocity']
    S2 = Pucks[puck_id]['position']
    V2 = Pucks[puck_id]['velocity']
    
    dS = S2 - S1
    dV = V2 - V1
    
    try:
        dot_dS_dV = np.dot(dS, dV)
        dot_dV_dV = np.dot(dV, dV)

        tca = -dot_dS_dV / dot_dV_dV
        Dtca = dS - dV * tca
        
        
        if tca < 0:
            tca = None
            Dtca = None
        scalar_Dtca = np.linalg.norm(Dtca)
        
  
        Pucks[puck_id]['tca'] = tca
        Pucks[puck_id]['Dtca'] = Dtca
        
        
        if tca is not None:
            timeframe = worker_Bauermeister.timeframe
            fuel = Pucks[puck_id]['fuel']
            adjusted_timeframe = timeframe + 1 if fuel <= 0 else timeframe

            Pucks[puck_id]['proximity_traffic'] = (
                0 <= tca <= adjusted_timeframe and scalar_Dtca <= distance_threshold
            )
        else:
            Pucks[puck_id]['proximity_traffic'] = False
            
    except ZeroDivisionError:
        
        Pucks[puck_id]['tca'] = None
        Pucks[puck_id]['Dtca'] = None
        Pucks[puck_id]['proximity_traffic'] = False
        
def minimum_resolution_function(A, relative_velocity):
    return (relative_velocity, A)

def maximum_resolution_function(A, B, relative_velocity):
    return (relative_velocity, A - (2 * B))

def minimum_on_boundary_resolution_function(index, A, B_orthogonal, relative_velocity):
    multiplier = 0.4 if index == 0 else -0.4
    B = multiplier * B_orthogonal
    return (relative_velocity, B + A)


def conflict_resolutions_and_limitations(own_puck):
    resolution_vector_functions = []
    limitation_vector_functions = []

    for puck_id, puck_data in Pucks.items():
        if puck_data == own_puck or not puck_data['proximity_traffic']:
            continue

        intruder = puck_data
        relative_position = intruder['position'] - own_puck['position']
        relative_velocity = intruder['velocity'] - own_puck['velocity']

        
        if np.linalg.norm(relative_velocity) == 0 or np.dot(relative_position, relative_velocity) >= 0:
            B = 2 * relative_velocity / np.linalg.norm(relative_velocity)
            B_orthogonal = np.array([-B[1], B[0]])

            A_0 = relative_position + B + B_orthogonal
            A_1 = relative_position + B - B_orthogonal

            alpha_0_criterion = np.dot(B_orthogonal, A_0)
            alpha_1_criterion = np.dot(-B_orthogonal, A_1)

            if alpha_0_criterion < 0 and alpha_1_criterion >= 0:
                limitation_vector_functions.append((minimum_resolution_function(A_1, relative_velocity), maximum_resolution_function(A_0, B, relative_velocity)))
            elif alpha_1_criterion < 0 and alpha_0_criterion >= 0:
                limitation_vector_functions.append((minimum_resolution_function(A_0, relative_velocity), maximum_resolution_function(A_1, B, relative_velocity)))

            continue  

        
        B = 2 * relative_velocity / np.linalg.norm(relative_velocity)
        B_orthogonal = np.array([-B[1], B[0]])
        
        A_0 = relative_position + B + B_orthogonal
        A_1 = relative_position + B - B_orthogonal

        alpha_0_criterion = np.dot(B_orthogonal, A_0)
        alpha_1_criterion = np.dot(-B_orthogonal, A_1)
        
        if alpha_0_criterion > 0 and alpha_1_criterion > 0:
            resolution_vector_functions.append((minimum_resolution_function(A_0, relative_velocity), minimum_resolution_function(A_1, relative_velocity)))
        elif alpha_0_criterion == 0:
            resolution_vector_functions.append((minimum_on_boundary_resolution_function(0, A_0, B_orthogonal, relative_velocity), minimum_resolution_function(A_1, relative_velocity)))
        elif alpha_1_criterion == 0:
            resolution_vector_functions.append((minimum_resolution_function(A_0, relative_velocity), minimum_on_boundary_resolution_function(1, A_1, B_orthogonal, relative_velocity)))
        elif alpha_0_criterion < 0 and alpha_1_criterion >= 0:
            limitation_vector_functions.append((minimum_resolution_function(A_1, relative_velocity), maximum_resolution_function(A_0, B, relative_velocity)))
        elif alpha_1_criterion < 0 and alpha_0_criterion >= 0:
            limitation_vector_functions.append((minimum_resolution_function(A_0, relative_velocity), maximum_resolution_function(A_1, B, relative_velocity)))
        else:
            print("Unexpected case, problem puck:", puck_id)
    
   
    return resolution_vector_functions, limitation_vector_functions



def starting_point(vector_function):
    return vector_function[0]

def linear_intersector(vector_function1, vector_function2):
    p1, d1 = vector_function1[:2]
    p2, d2 = vector_function2[:2]

    if np.array_equal(p1, p2):
        return p1, 0

    A = np.array([d1, -d2]).T
    b = p2 - p1

    if np.linalg.det(A) == 0:
        return None

    t_values = np.linalg.solve(A, b)
    t1, t2 = t_values

    if t1 >= 0 and t2 >= 0:
        return p1 + t1 * d1, t1
    else:
        return None

def closest_approach(vector_function):
    p, d = vector_function[:2]
    
    if np.all(d == 0):  
        return p
    
    t = -np.dot(p, d) / np.dot(d, d)
    
    if t > 0:
        return p + t * d
    else:
        return p

def sort_by_distance(points):
    unique_points = list(set(map(tuple, points)))
    sorted_points = sorted(unique_points, key=lambda point: point[0]**2 + point[1]**2)
    return [np.array(point) for point in sorted_points]

def insert_into_sorted_by_distance(list_of_points, new_point): #only used in SEC collision
    new_point_distance = new_point[0]**2 + new_point[1]**2
    for i, point in enumerate(list_of_points):
        if new_point_distance < point[0]**2 + point[1]**2:
            list_of_points.insert(i, new_point)
            return
    list_of_points.append(new_point)

def linear_combiner(list_of_resolutions):
    if not list_of_resolutions:
        return np.array([0, 0])

    intersection_points = []
    starting_points = [starting_point(vector_function) for vector_function in list_of_resolutions]
    closest_points = [closest_approach(vector_function) for vector_function in list_of_resolutions]


    for i in range(len(list_of_resolutions)):
        for j in range(i + 1, len(list_of_resolutions)):
            intersection = linear_intersector(list_of_resolutions[i], list_of_resolutions[j])
            if intersection is not None:
                intersection_points.append(intersection[0])

    all_points_of_interest = intersection_points + starting_points + closest_points
    
    
#hier alle MINI SCHNITTPUNKTE

    true_mini_points = []

    for point in all_points_of_interest:
        temp_line = (np.array([0, 0]), point)
    
    
    if all(
        (intersection_result := linear_intersector(temp_line, vector_function)) is None or intersection_result[1] < 1
        for vector_function in list_of_resolutions
    ):
        true_mini_points.append(point)

  

    return sort_by_distance(true_mini_points)



def secondary_collision_protection(list_of_points, list_of_limit_functions):
    valid_points = []
    for point in list_of_points:
        if point is None:
            continue
        
        if np.array_equal(point, np.array([0, 0])):
            valid_points.append(point)
            continue

        is_valid_point = True
        for mini_function, max_function in list_of_limit_functions:
            temp_vector_function = (np.array([0, 0]), point)
            intersection_result_max = linear_intersector(temp_vector_function, max_function)

            if intersection_result_max is None or (intersection_result_max[1] > 1):
                continue
            else:
                intersection_result_mini = linear_intersector(temp_vector_function, mini_function)
                if intersection_result_mini is None:
                    is_valid_point = False
                    break

                if 0 < intersection_result_mini[1] <= 1:
                    continue
                else:
                    insert_into_sorted_by_distance(list_of_points, intersection_result_mini[0])
                    is_valid_point = False
                    break

        if is_valid_point:
            print("Secondary collision prevented")
            valid_points.append(point)

    return valid_points

def limiter(list_of_points):
    if not list_of_points:
        return []

    own_puck_velocity = Pucks[id]['velocity']

    for point in list_of_points:
        if point is None:
            continue

        point_length = np.linalg.norm(point)
        vector_sum = point + own_puck_velocity
        vector_sum_length = np.linalg.norm(vector_sum)

        if point_length <= A_MAX and V_MIN <= vector_sum_length <= V_MAX:
            return point

    return np.array([0,0])


def adaptive_performance_control(runtime):
    max_timeframe = 3
    min_timeframe = 0.4  
    if runtime >= 0.02:
        worker_Bauermeister.timeframe = max(min_timeframe, worker_Bauermeister.timeframe * 0.7) 
    if runtime <= 0.01:
        worker_Bauermeister.timeframe = min(max_timeframe, worker_Bauermeister.timeframe + (max_timeframe - worker_Bauermeister.timeframe) / 50)

def mirror(vector, mirror_at):
    x_min = 0 
    x_max = 120 
    y_min = 0 
    y_max = 120 

    mirrored_vector = None

    if mirror_at == 'x_min':
        mirrored_vector = np.array([2 * x_min - vector[0], vector[1]])
    elif mirror_at == 'x_max':
        mirrored_vector = np.array([2 * x_max - vector[0], vector[1]])
    elif mirror_at == 'y_min':
        mirrored_vector = np.array([vector[0], 2 * y_min - vector[1]])
    elif mirror_at == 'y_max':
        mirrored_vector = np.array([vector[0], 2 * y_max - vector[1]])
    else:
        raise ValueError("Invalid mirror_at value. Expected 'x_min', 'x_max', 'y_min', or 'y_max'.")

    return mirrored_vector

def create_mirror_puck(mirror_at):
    
    if 'mirror_Pucks' not in globals():
        global mirror_Pucks
        mirror_Pucks = {}

    
    own_puck = Pucks["id"]

    
    mirrored_position = mirror(own_puck['position'], mirror_at)
    mirrored_velocity = mirror(own_puck['velocity'], mirror_at)
    
   
    if mirror_at == 'x_min':
        mirror_id = 100
    elif mirror_at == 'y_max':
        mirror_id = 101
    elif mirror_at == 'x_max':
        mirror_id = 102
    elif mirror_at == 'y_min':
        mirror_id = 103
    else:
        raise ValueError("Invalid mirror_at value. Expected 'x_min', 'x_max', 'y_min', or 'y_max'.")

    
    mirrored_puck = {
        'id': mirror_id,
        'name': mirror_at,
        'position': mirrored_position,
        'velocity': mirrored_velocity
    }
    
    
    mirror_Pucks[mirror_id] = mirrored_puck

   
    return mirror_Pucks

def worker_Bauermeister(id, secret, q_request, q_reply):
    print("HELLO")
    while True:
        if not getattr(worker_Bauermeister, 'initialized', False):
            q_request.put(('GET_BOX', id))
            q_request.put(('GET_SIZE', id))
            q_request.put(('SET_NAME', 'Bauermeister', secret, id))

            worker_Bauermeister.cycle_count = 0
            worker_Bauermeister.box = None
            worker_Bauermeister.n_workers = None
            worker_Bauermeister.timeframe = 0.5
            try:
                while True:
                    reply = q_reply.get(timeout=0.02)
                    match reply:
                        case ('GET_BOX', box):
                            worker_Bauermeister.box = box
                        case ('GET_SIZE', n_workers):
                            worker_Bauermeister.n_workers = n_workers
                        case _:
                            continue

                    if worker_Bauermeister.box is not None and worker_Bauermeister.n_workers is not None:
                        break
            except queue.Empty:
                worker_Bauermeister.initialized = False
                break

            for n in range(worker_Bauermeister.n_workers):
                q_request.put(('GET_PUCK', n, id))
            
            worker_Bauermeister.initialized = True
    
    while True:
        print("starting the main loop")
        start_time = time.time()

        attempts = 0
        while attempts < 10:
            try:
                response = q_reply.get(block=False)
                if response is not None:
                    match response:
                        case ('GET_BOX', box):
                            worker_Bauermeister.box = box
                        case ('GET_SIZE', n_workers):
                            worker_Bauermeister.n_workers = n_workers
                        case ('GET_PUCK', puck):
                            add_or_update_puck(puck)
            except queue.Empty:
                break
            attempts += 1
            time.sleep(0.001)

        q_request.put(('GET_PUCK',id, id))
        q_request.put(('SET_ACCELERATION', np.array([0, 0]), secret, id))
        
        if worker_Bauermeister.cycle_count % 10 == 0:
            
            for puck_id, puck_data in Pucks.items():
                if puck_data['alive']:
                    q_request.put(('GET_PUCK', puck_id, id))
        else:
            for puck_id, puck_data in Pucks.items():
                if puck_data['alive'] and puck_data['proximity_traffic']:
                    q_request.put(('GET_PUCK', puck_id, id))


        if (worker_Bauermeister.cycle_count-1) % 10 == 0:
           for puck_id in Pucks.items():
               is_proximity(puck_id, id)
        
        worker_Bauermeister.cycle_count += 1

        

        resolution_vector_functions, limitation_vector_functions = conflict_resolutions_and_limitations(Pucks[id])

        print(len(resolution_vector_functions), "RESOLUTIONS:", resolution_vector_functions)
        print(len(limitation_vector_functions), "LIMITATIONS:", limitation_vector_functions)

        sorted_preliminary_solutions = linear_combiner(resolution_vector_functions)
        #ALLE TRUE MINI FUNCTION POINTS SORTED AB HIER

        print("SORTIERT:", len(sorted_preliminary_solutions), sorted_preliminary_solutions)

        if len(limitation_vector_functions) > 0:
            print("check secondary collisions")
            result_unsorted = secondary_collision_protection(sorted_preliminary_solutions, limitation_vector_functions)
            result = sort_by_distance(result_unsorted)
        else:
            result = sorted_preliminary_solutions

        print("SEC COLLISION CHECKED:", len(result), result)

        final_result = limiter(result)

        q_request.put(('SET_ACCELERATION', final_result, secret, id))
        print("FINALRESULT:", final_result)

        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time)
        time.sleep(sleep_time)


