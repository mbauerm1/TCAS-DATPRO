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


own_id = None
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

def is_proximity(puck_id):
 
    if puck_id == own_id:
        return
    
    own_puck = Pucks[own_id]
    S1 = own_puck['position']
    V1 = own_puck['velocity']
    S2 = Pucks[puck_id]['position']
    V2 = Pucks[puck_id]['velocity']
    
    dS = S2 - S1
    dV = V2 - V1
    
    try:
        tca = -np.dot(dS, dV) / np.dot(dV, dV)
        Dtca = dS - dV * np.dot(dS, dV) / np.dot(dV, dV)
        
        if tca < 0:
            tca = None
            Dtca = None
        
        Pucks[puck_id]['tca'] = tca
        Pucks[puck_id]['Dtca'] = Dtca
        
        if tca is not None:
            if Pucks[puck_id]['fuel'] <= 0:
                if 0 <= tca <= 2:
                    Pucks[puck_id]['proximity_traffic'] = True
                else:
                    Pucks[puck_id]['proximity_traffic'] = False
            else:
                if 0 <= tca <= 1:
                    Pucks[puck_id]['proximity_traffic'] = True
                else:
                    Pucks[puck_id]['proximity_traffic'] = False
        else:
            Pucks[puck_id]['proximity_traffic'] = False
            
    except ZeroDivisionError:
        Pucks[puck_id]['tca'] = None
        Pucks[puck_id]['Dtca'] = None
        Pucks[puck_id]['proximity_traffic'] = False

    
    
    

        
        
        
        

def worker_Bauermeister(id, secret, q_request, q_reply):
    while True:
        
        if not getattr(worker_Bauermeister, 'initialized', False):
           
            q_request.put(('GET_BOX', id))
            q_request.put(('GET_SIZE', id))
            q_request.put(('SET_NAME', 'Bauermeister', secret, id))

            worker_Bauermeister.cycle_count = 0
            worker_Bauermeister.box = None
            worker_Bauermeister.n_workers = None

            
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
#initialisierung abgeschlossen            


    # Main Loop:
    while True:
        start_time = time.time()
      
#erstmal reply queue abhören:
        attempts = 0
        while attempts < 35:  
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
            time.sleep(0.0001)  

        
#ab hier alle Pucks verfügbar:
    #own_id finden
    if own_id is None:
        for puck_id, puck_data in Pucks.items():
            if puck_data.get('name') == 'Bauermeister':
                own_id = puck_id
                break
     


#PUCK_Update der proximity pucks, komplettes update nur aller 0,5 sekunden
    if worker_Bauermeister.cycle_count % 25 == 0:
        q_request.put(('GET_PUCK', n, own_id))
        for puck_id, puck_data in Pucks.items():
            if puck_data['alive']:
                is_proximity(puck_id)
                q_request.put(('GET_PUCK', n, puck_id))

    else:
        for puck_id, puck_data in Pucks.items():
            q_request.put(('GET_PUCK', n, own_id))
            if puck_data['alive'] and puck_data['proximity_traffic']:
                is_proximity(puck_id)
                q_request.put(('GET_PUCK', n, puck_id)) 

        worker_Bauermeister.cycle_count += 1
        
    #jetzt die eigentliche collision avoidance:
   
    def minimum_resolution_function(A, relative_velocity):
        return A + relative_velocity, relative_velocity

    def maximum_resolution_function(A, relative_velocity):
        new_A = A - relative_velocity * (2 / np.linalg.norm(relative_velocity))
        return new_A, relative_velocity

   
    def minimum_on_boundary_resolution_function(index, A, B_orthogonal, relative_velocity):
        adjustment = 0.4 / np.linalg.norm(B_orthogonal)
        new_A = A + B_orthogonal * adjustment if index == 0 else A - B_orthogonal * adjustment
        return new_A, relative_velocity

    
    def conflict_resolutions_and_limitations():
        own_puck = Pucks[own_id]
        resolution_vector_functions = []
        limitation_vector_functions = []

        for puck_id, puck_data in Pucks.items():
            if puck_id == own_id or not puck_data['proximity_traffic']:
                continue

            intruder = puck_data
            relative_position = intruder['position'] - own_puck['position']
            relative_velocity = intruder['velocity'] - own_puck['velocity']

            if np.linalg.norm(relative_velocity) == 0:
                continue

            B = relative_velocity / np.linalg.norm(relative_velocity)
            B_orthogonal = np.array([-B[1], B[0]])

            A_0 = relative_position + B + B_orthogonal
            A_1 = relative_position + B - B_orthogonal

            alpha_0_criterion = np.dot(B_orthogonal, A_0)
            alpha_1_criterion = np.dot(-B_orthogonal, A_1)

            if alpha_0_criterion > 0 and alpha_1_criterion > 0:  # Own Puck in track 
                resolution_vector_functions.append((*minimum_resolution_function(A_0, relative_velocity), "mini"))
                resolution_vector_functions.append((*minimum_resolution_function(A_1, relative_velocity), "mini"))
                
            elif alpha_0_criterion == 0:  # Own Puck on boundary 0
                resolution_vector_functions.append((*minimum_on_boundary_resolution_function(0, A_0, B_orthogonal, relative_velocity), "mini"))
                resolution_vector_functions.append((*minimum_resolution_function(A_1, relative_velocity), "mini"))
                
            elif alpha_1_criterion == 0:  # Own Puck on boundary 1
                resolution_vector_functions.append((*minimum_resolution_function(A_0, relative_velocity), "mini"))
                resolution_vector_functions.append((*minimum_on_boundary_resolution_function(1, A_1, B_orthogonal, relative_velocity), "mini"))
            
            elif alpha_0_criterion < 0 and alpha_1_criterion >= 0:
                limitation_vector_functions.append(((*minimum_resolution_function(A_1, relative_velocity), "mini"),(*maximum_resolution_function(A_0, relative_velocity), "max"))) 
                
               
                    
            elif alpha_1_criterion < 0 and alpha_0_criterion >= 0:  
                limitation_vector_functions.append(((*minimum_resolution_function(A_0, relative_velocity), "mini"),(*maximum_resolution_function(A_1, relative_velocity), "max"))) 
                
            else:
                print("kann eigentlich garnicht vorkommen, Problempuck:", puck_id)

        return resolution_vector_functions, limitation_vector_functions

    def flight_envelope_protection():
        own_velocity = Pucks[own_id]['velocity']
        own_velocity_length = np.linalg.norm(own_velocity)
        
        
        V_max_length = V_MAX - 1
        V_max_vector = (own_velocity / own_velocity_length) * V_max_length
        
       
        V_min_length = V_MIN + 1
        V_min_vector = (own_velocity / own_velocity_length) * V_min_length
        
      
        V_max_perpendicular_vector = np.array([-V_max_vector[1], V_max_vector[0]])
        
       
        V_max_protection_start = V_max_vector - 3 * V_max_perpendicular_vector
        V_max_protection = V_max_protection_start, V_max_perpendicular_vector
        
       
        V_min_protection_start = V_min_vector - 3 * V_max_perpendicular_vector
        V_min_protection = V_min_protection_start, V_max_perpendicular_vector
        
        return V_max_protection, V_min_protection

    
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

        if t1 > 0 and t2 > 0:
            return p1 + t1 * d1, t1
        else:
            return None

    #
    def closest_approach(vector_function):
        p, d = vector_function[:2]
        t = -np.dot(p, d) / np.dot(d, d)
        return p + t * d if t > 0 else p

    
    def sort_by_distance(points):
        unique_points = list(set(map(tuple, points)))
        sorted_points = sorted(unique_points, key=lambda point: point[0]**2 + point[1]**2)
        return [np.array(point) for point in sorted_points]

    def insert_into_sorted_by_distance(list_of_points, new_point):
        new_point_distance = new_point[0]**2 + new_point[1]**2
        for i, point in enumerate(list_of_points):
            if new_point_distance < point[0]**2 + point[1]**2:
                list_of_points.insert(i, new_point)  # Insert at the correct position
                return
        list_of_points.append(new_point)  # Append at the end if it's the farthest

    
    def linear_combiner(list_of_resolutions):
        if not list_of_resolutions:
            return np.array([0, 0])

        intersection_points = []
        starting_points = [starting_point(vector_function) for vector_function in list_of_resolutions]
        closest_points = [closest_approach(vector_function) for vector_function in list_of_resolutions]

        #  flight envelope protection
        V_max_protection, V_min_protection = flight_envelope_protection()
        list_of_resolutions.append(V_max_protection)
        list_of_resolutions.append(V_min_protection)

        print("ANZAHL RESOLUTIONS incl envelope protection",len(list_of_resolutions))
        for i in range(len(list_of_resolutions)):
            for j in range(i + 1, len(list_of_resolutions)):
                intersection = linear_intersector(list_of_resolutions[i], list_of_resolutions[j])
                if intersection is not None:
                    intersection_points.append(intersection[0])

        all_points_of_interest = intersection_points + starting_points + closest_points
        preliminary_solutions = []

        for point in all_points_of_interest:
            temp_line = (np.array([0, 0]), point)
            if all(linear_intersector(temp_line, vector_function) is None for vector_function in list_of_resolutions):
                preliminary_solutions.append(point)

        return sort_by_distance(preliminary_solutions)


    #  to check max-min constraints
    def secondary_collision_protection(list_of_points, list_of_limit_functions):
        valid_points = []
        for point in list_of_points:
            if point is None:
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






    #
    def limiter(list_of_points):
        if not list_of_points:
            return []  
        
        own_puck_velocity = Pucks[own_id]['velocity']
        valid_points = []

        for point in list_of_points:
            if point is None:
                continue 

            point_length = np.linalg.norm(point)
            vector_sum = point + own_puck_velocity
            vector_sum_length = np.linalg.norm(vector_sum)

            if point_length <= A_MAX and V_MIN <= vector_sum_length <= V_MAX:
                valid_points.append(point)

        return valid_points



    resolution_vector_functions, limitation_vector_functions = conflict_resolutions_and_limitations()

    print(len(resolution_vector_functions),"RESOLUTIONS:", resolution_vector_functions)
    print(len(limitation_vector_functions),"LIMITATIONS:", limitation_vector_functions)

    # Combine linear solutions
    sorted_preliminary_solutions = linear_combiner(resolution_vector_functions)

    print("SORTIERT:", len(sorted_preliminary_solutions),sorted_preliminary_solutions)

    if len(limitation_vector_functions) > 0:
        print("check secondary collisions")
        result_unsorted = secondary_collision_protection(sorted_preliminary_solutions, limitation_vector_functions)
        result = sort_by_distance(result_unsorted)
    else:
        result = sorted_preliminary_solutions

    print("SEC COLLISIOn CHECKED:",len(result),result)

    final_result = limiter(result)
   
    q_request.put(('SET_ACCELERATION', final_result, secret, own_id))
    print("FINALRESULT:", final_result)

    #
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")        
        
        

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time)  
        time.sleep(sleep_time)
        
        
        
  
        
        
        
        
        
        
        
        

