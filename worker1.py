import queue
import time
from puck import Puck
import puck_add_uppdate_delete2


V_MIN = 10.0
V_MAX = 42.0
A_MAX = 100.0
RADIUS = 1.0

def worker_Bauermeister(id, secret, q_request, q_reply):
    if not hasattr(worker_Bauermeister, 'initialized'):
        
        q_request.put(('GET_BOX', id))
        q_request.put(('GET_SIZE', id))
        q_request.put(('SET_NAME', 'Bauermeister', secret, id))
        
        for n in range(30):  
            q_request.put(('GET_PUCK', n, id))
        
        worker_Bauermeister.initialized = True
        worker_Bauermeister.cycle_count = 0
        

    #
    while True:
        start_time = time.time()
        check_proximity
        
        responses = [] # diese Liste löschen. lieber immer das dict aktuallisieren
        while True:
            try:
                response = q_reply.get(block=False)
                if response is not None:
                    
                    responses.append(response)
            except queue.Empty:
                
                break
        
        
        worker_Bauermeister.cycle_count += 1

        
        if worker_Bauermeister.cycle_count % 25 == 0:
            worker_Bauermeister.proximity_traffic = check_proximity(responses)
        
       
        avoidance_vector = calculate_avoidance_vector(worker_Bauermeister.proximity_traffic)

        
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
    
    vector = [0, 0]  #
    for puck_id, puck_data in Pucks.items():
        if puck_data['proximity_traffic']:
            #
            pass  
    return vector


