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
Pucks = {}  #


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
            # Send initial requests
            q_request.put(('GET_BOX', id))
            q_request.put(('GET_SIZE', id))
            q_request.put(('SET_NAME', 'Bauermeister', secret, id))

            worker_Bauermeister.cycle_count = 0
            worker_Bauermeister.box = None
            worker_Bauermeister.n_workers = None

          
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
        
        
  
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time)  
        time.sleep(sleep_time)
        
        
        
  


def calculate_avoidance_vector():
    
    vector = [0, 0]  # 
    for puck_id, puck_data in Pucks.items():
        if puck_data['proximity_traffic']:
            
            pass  
    return vector


