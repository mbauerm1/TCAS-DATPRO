import queue
import time
from puck import Puck
import puck_add_uppdate_delete
import multiprocessing as mp



V_MIN = 10.0
V_MAX = 42.0
A_MAX = 100.0
RADIUS = 1.0


def worker_Bauermeister(id, secret, q_request, q_reply):
    while True:
        
        if not getattr(worker_Bauermeister, 'initialized', False):
            #
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

        
        responses = []
        
        attempts = 0
        while attempts < 30: #max 30 Versuche und time delay, verhindert endlosschleife
            try:
                response = q_reply.get(block=False)
                if response is not None:
                    # Collect the response
                    responses.append(response)
            except queue.Empty:
                    break
            attempts += 1
            time.sleep(0.0001)  
        
      
    
        

       #aller 25 Ticks proximity traffic updaten
        if worker_Bauermeister.cycle_count % 25 == 0:
            get_pucks(all)
        else:
            get_pucks(all_alive)
        
        #hier is_intruder einbauen
        
        worker_Bauermeister.cycle_count += 1
        
  
        
   #TO DO AVOIDANCE CALCULATION

       
        q_request.put(('AVOIDANCE_VECTOR', avoidance_vector, id))



        
    
        elapsed_time = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time)  
        time.sleep(sleep_time)
        
        
        
        
        
        
        
        
        
        
        
        
def check_proximity():
    # Dummy routine 
    
    for puck_id, puck_data in Pucks.items():
        
        if True:  
            Pucks[puck_id]['proximity_traffic'] = True
        else:
            Pucks[puck_id]['proximity_traffic'] = False


def calculate_avoidance_vector():
    # Dummy routine
    vector = [0, 0]  
    for puck_id, puck_data in Pucks.items():
        if puck_data['proximity_traffic']:
            
            pass  # Implement your avoidance logic here
    return vector


