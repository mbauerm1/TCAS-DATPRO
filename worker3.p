import queue
import time

import multiprocessing as mp
from puck import Puck
from puck_server import Puck_Server



V_MIN = 10.0
V_MAX = 42.0
A_MAX = 100.0
RADIUS = 1.0


own_id = None
Pucks = {}

#
def add_or_update_puck(puck):
    puck_id = puck.get_id()
    
    # 
    if not puck.is_alive():
        print(f"Puck with id {puck_id} is not alive..")
        delete_puck(puck_id)
        return

    puck_data = {
        'id': puck.get_id(),
        'name': puck.get_name(),
        'position': puck.get_position(),
        'velocity': puck.get_velocity(),
        'acceleration': puck.get_acceleration(),
        'timestamp': puck.get_time(),
        'fuel': puck.get_fuel(),
        'alive': puck.is_alive(),
        'proximity_traffic': False
    }

    
    Pucks[puck_id] = puck_data

def delete_puck(puck_id):
    if puck_id in Pucks:
        del Pucks[puck_id]

def print_pucks():
    for puck_id, puck_data in Pucks.items():
        print(f"Puck ID {puck_id}: {puck_data}")
        
        
        
        
        

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
        check_proximity()  # TO DO

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
            time.sleep(0.0001)  #
        
#ab hier alle Pucks verfügbar:
    #own_id finden
    if own_id is None:
    for puck_id, puck_data in Pucks.items():
        if puck_data.get('name') == 'Bauermeister':
            own_id = puck_id
            break
     


        
        
   #TO DO AVOIDANCE CALCULATION

       
        q_request.put(('AVOIDANCE_VECTOR', avoidance_vector, id))








#PUCK_Update fuer die naechste Runde:
        if worker_Bauermeister.cycle_count % 25 == 0:
            get_pucks(all)
        else:
            get_pucks(all_alive)
        
        #hier is_intruder einbauen
        
        worker_Bauermeister.cycle_count += 1
        
        

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time)  # 0.02 seconds = 20 milliseconds
        time.sleep(sleep_time)
        
        
        
  
        
        
        
        
        
        
        
        
        
def check_proximity():
    # Dummy routine to check for proximity traffic
    
    for puck_id, puck_data in Pucks.items():
        # Determine if the puck is in proximity traffic
        if True:  # Replace with your criteria
            Pucks[puck_id]['proximity_traffic'] = True
        else:
            Pucks[puck_id]['proximity_traffic'] = False


def calculate_avoidance_vector():
    # Dummy routine
    vector = [0, 0]  
    for puck_id, puck_data in Pucks.items():
        if puck_data['proximity_traffic']:
           
            pass 
    return vector


