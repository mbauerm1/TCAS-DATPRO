import queue
import time
import numpy as np
from puck import Puck
from puck_server import Puck_Server
import sympy as sp
import math

def add_or_update_puck(puck, pucks):
    """
    Fügt einen neuen Puck zum 'pucks'-Dictionary hinzu oder aktualisiert einen bestehenden.

    Im Dictionary, das alle Puck-Daten speichert, indiziert durch die Puck-ID.

    Die Routine führt folgende Schritte durch:
    1. Ruft die ID des Pucks ab.
    2. Prüft, ob der Puck aktiv ist (außer es handelt dem eigenen ) oder ob der Puck None ist:
       - Falls ja, wird der Puck aus dem 'pucks'-Dictionary gelöscht.
    3. Erstellt ein Dictionary 'updated_data' mit den aktuellen Eigenschaften des Pucks.
    4. Falls der Puck bereits im 'pucks'-Dictionary vorhanden ist, wird der Eintrag des Pucks mit 'updated_data' aktualisiert.
    5. Falls der Puck noch nicht im Dictionary vorhanden ist, wird ein neuer Eintrag mit 'updated_data' erstellt
       und zusätzliche Felder wie 'proximity_traffic', 'tca' und 'Dtca' werden initialisiert.
    """
    puck_id = puck.get_id()
 
    if not puck.is_alive() and puck_id != worker_Bauermeister.id or puck is None:
       
        delete_puck(puck_id, pucks)
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

    if puck_id in pucks:
        pucks[puck_id].update(updated_data)
    else:
        pucks[puck_id] = {
            **updated_data,
            'proximity_traffic': False,
            'tca': None,
            'Dtca': None
        }
    
def delete_puck(puck_id, pucks):
    if puck_id in pucks:
        del pucks[puck_id]

def print_pucks(pucks):
    for puck_id, puck_data in pucks.items():
        print(f"Puck ID {puck_id}: {puck_data}")

       
def mirror(vector, mirror_at):
    """
    Spiegelt einen gegebenen Vektor an einer Spielfeldwand.

    Args:
        vector (numpy.ndarray)
        mirror_at (str): String, der die Achse angibt, an der der Vektor gespiegelt werden soll.
                        
    Returns:
        numpy.ndarray: Der gespiegelte Vektor als NumPy-Array. Gibt None zurück, wenn der Eingabevektor None ist
                       oder der Wert für mirror_at ungültig ist.
    Raises:
        ValueError: Wenn ein ungültiger Wert für mirror_at angegeben wird.
    """
    if vector is None:
       return None
    mirrored_vector = None

    if mirror_at == 'x_min':
        mirrored_vector = np.array([2 * worker_Bauermeister.X_MIN - vector[0], vector[1]])
    elif mirror_at == 'x_max':
        mirrored_vector = np.array([2 * worker_Bauermeister.X_MAX - vector[0], vector[1]])
    elif mirror_at == 'y_min':
        mirrored_vector = np.array([vector[0], 2 * worker_Bauermeister.Y_MIN - vector[1]])
    elif mirror_at == 'y_max':
        mirrored_vector = np.array([vector[0], 2 * worker_Bauermeister.Y_MAX - vector[1]])
    else:
        raise ValueError("Invalid mirror_at value. Expected 'x_min', 'x_max', 'y_min', or 'y_max'.")

    return mirrored_vector

def mirror_of_function(vector_function, mirror_at):
    """

    Ähnlich wie bei der `mirror`-Funktion, diesmal aber für eine Vektor-Funktion.
    Args:
        vector_function (tuple):- position (numpy.ndarray): Der "Startpunk"t des Vektors ( bzw Verschiebungsvektor)
                                 - direction (numpy.ndarray): Die "Richtung" ist der relevante vektor für Schnittpunkte o.ä. siehe unten
        mirror_at (str): String

    Returns:
        tuple: - mirrored_position (numpy.ndarray): Der gespiegelte Startpunkt.
               - mirrored_direction (numpy.ndarray): Die gespiegelte Richtung.
               Gibt None zurück, wenn die Eingabevektor-Funktion None ist oder ungültige Werte enthält.
    Raises:
        ValueError: Wenn ein ungültiger Wert für mirror_at angegeben wird.
    """
    if vector_function is None:
        return None
    
    position, direction = vector_function
    if position is None or direction is None:
        return None

    mirrored_position = mirror(position, mirror_at)
   
    if mirror_at == 'x_min' or mirror_at == 'x_max':
        mirrored_direction = np.array([-direction[0], direction[1]])
    elif mirror_at == 'y_min' or mirror_at == 'y_max':
        mirrored_direction = np.array([direction[0], -direction[1]])
    else:
        raise ValueError("Invalid mirror_at value. Expected 'x_min', 'x_max', 'y_min', or 'y_max'.")

    return (mirrored_position, mirrored_direction)

def create_mirror_pucks(pucks):
    """
    Für jede Spiegelrichtung ('x_min', 'x_max', 'y_min', 'y_max') wird ein neuer gespiegelter Puck erstellt
    und zu `worker_Bauermeister.mirror_Pucks` hinzugefügt.

    Args:
        pucks (dict): Ein Dictionary mit allen vorhandenen Pucks, einschließlich des eigenen Pucks.

    Returns:
        dict: Ein "geupdatetes" Dictionary mit den gespiegelten Pucks.
    """
    mirror_directions = {
        'x_min': 100,
        'y_max': 101,
        'x_max': 102,
        'y_min': 103
    }

    own_puck = pucks[worker_Bauermeister.id]

    for mirror_at, mirror_id in mirror_directions.items():
        mirrored_position = mirror(own_puck['position'], mirror_at)
        
        own_velocity = own_puck['velocity'] 
        
        if mirror_id == 100 or mirror_id == 102:
            mirrored_velocity = np.array([-own_velocity[0], own_velocity[1]])
        elif mirror_id == 101 or mirror_id == 103:
            mirrored_velocity = np.array([own_velocity[0], -own_velocity[1]])
        
        mirrored_puck = {
            'id': mirror_id,
            'name': mirror_at,
            'position': mirrored_position,
            'velocity': mirrored_velocity
        }

        worker_Bauermeister.mirror_Pucks[mirror_id] = mirrored_puck
#        print(worker_Bauermeister.Pucks[id])
#        print(f"Created mirrored puck for {mirror_at}: {mirrored_puck}")
    return worker_Bauermeister.mirror_Pucks

def is_proximity(puck_id, own_id, pucks):
    """
    Bestimmt, ob ein gegebener Puck (puck_id) für einen eigenen Puck (own_id) eine potenzielle Gefahr darstellt.

    Analog zur Lösung in der Vorlesung berechnet diese routine die "time to closest approach" (TCA) und den minimalen Abstand (Dtca) zwischen zwei Pucks.
    Basierend auf diesen Werten wird beurteilt, ob der betrachtete Puck eine Gefahr für den eigenen Puck darstellt und 
    entsprechende bool-Markierungen im Dictionary gesetzt.

    Args:
        puck_id (int): Die ID des zu überprüfenden Pucks.
        own_id (int): Die ID des eigenen Pucks.
        pucks (dict)

    Ablauf:
    - Berechnung der relativen Position und Geschwindigkeit zwischen dem eigenen Puck und dem zu prüfenden Puck.
    - Bestimmung tca und Dtca und aus diesen Werten auch die "quick solution", ein einzelner Vektor
    - Spezielle Fälle (`own_id` von 100 bis 103) für die gespiegelten OUcks werden seperat gehandhabt
    - dafür wird für einen gespiegelten Puck tca und Dtca überprüft und ggf der Puck als proximity traffic markiert, mit der jeweiligen Spiegelachse.
    - die jeweilige reflektions Flag z.b. "left_danger" wird dann für die spätere Verarbeitung auf "True" gesetzt
    - Pucks die "out of Fuel" sind werden früher als proximity traffic getagged, da diese sich nurnoch geradlinig bewegen kann so schon frühzeitig eine Treibstoffsparende Lösung gefunden werden.
    - Wenn eine Gefahr festgestellt wird, wird eine schnelle Lösung zur Vermeidung in eine Liste von schnellen Lösungen eingefügt.

    Exceptions:
        ZeroDivisionError: Tritt auf, wenn die Geschwindigkeit beider Pucks gleich ist (Division durch null). In diesem Fall wird der Puck als ungefährlich markiert.

    Returns:
        None: Diese Funktion aktualisiert das `pucks`-Dictionary und die internen Zustände des `worker_Bauermeister`-Objekts.
    """

    #print("PROXY CALLED für id:", puck_id, "range", worker_Bauermeister.range, "time", worker_Bauermeister.timeframe)
    
    if puck_id == own_id or puck_id == worker_Bauermeister.id:
        return
    if own_id == 100 or own_id == 101 or own_id == 102 or own_id == 103:
        own_puck = worker_Bauermeister.mirror_Pucks[own_id]
    else:
        own_puck = worker_Bauermeister.Pucks[own_id]
    
    S1 = own_puck['position']
    V1 = own_puck['velocity']
    S2 = pucks[puck_id]['position']
    V2 = pucks[puck_id]['velocity']
    
    dS = S2 - S1
    dV = V2 - V1
    dSx,dSy = dS
    #length_dS_sqred = dSx**2 +dSy**2
    try:
        dot_dS_dV = np.dot(dS, dV)
        dot_dV_dV = np.dot(dV, dV)

        tca = -dot_dS_dV / dot_dV_dV if dot_dV_dV != 0 else None
        Dtca = dS + dV *tca if tca is not None else None
        
        if tca is None or tca < 0:
            tca = None
            Dtca = None
            #print("collision in Vergangenheit")
        
        scalar_Dtca = np.linalg.norm(Dtca) if Dtca is not None else None

        pucks[puck_id]['tca'] = tca
        pucks[puck_id]['Dtca'] = Dtca
        pucks[puck_id]['scalar_Dtca'] = scalar_Dtca
        
        
        if tca is not None and scalar_Dtca is not None :
            #print(f" Puck {puck_id} wird gecheckt, tca: {tca}, DTCA {Dtca}")
            timeframe = worker_Bauermeister.timeframe
            fuel = pucks[puck_id]['fuel']
            adjusted_timeframe = timeframe + 0.5 if fuel <= 0 else timeframe
            proximity_condition = (tca <= adjusted_timeframe and scalar_Dtca <= worker_Bauermeister.range) 
            #print("PROXY CONDITION IS:", proximity_condition)
            quick_solution = None
            if proximity_condition:
                quick_solution = -Dtca/(tca/4)
                #print("quick solution", quick_solution)
            if own_id == 100:
#                print(f"Setting proximity_traffic_x_min for puck {puck_id} to {proximity_condition}")
                pucks[puck_id]['proximity_traffic_x_min'] = proximity_condition
                if proximity_condition:
                    #print("left Danger")
                    worker_Bauermeister.left_danger = True
            elif own_id == 101:
#                print(f"Setting proximity_traffic_y_max for puck {puck_id} to {proximity_condition}")
                pucks[puck_id]['proximity_traffic_y_max'] = proximity_condition
                if proximity_condition:
                    #print("ceiling Danger")
                    worker_Bauermeister.ceiling_danger = True
            elif own_id == 102:
#                print(f"Setting proximity_traffic_x_max for puck {puck_id} to {proximity_condition}")
                pucks[puck_id]['proximity_traffic_x_max'] = proximity_condition
                if proximity_condition:
                    #print("right Danger")
                    worker_Bauermeister.right_danger=True
            elif own_id == 103:
#                print(f"Setting proximity_traffic_y_min for puck {puck_id} to {proximity_condition}")
                pucks[puck_id]['proximity_traffic_y_min'] = proximity_condition
                if proximity_condition:
                    #print("floor Danger")
                    worker_Bauermeister.floor_danger = True
            else:
#                print(f"Setting proximity_traffic for puck {puck_id} to {proximity_condition}")
                pucks[puck_id]['proximity_traffic'] = proximity_condition
                
            if quick_solution is not None:
                if own_id == 100:
                    quick_solution = mirror(quick_solution,'x_min')
                elif own_id == 101:
                    quick_solution = mirror(quick_solution, 'y_max')
                elif own_id == 102:
                    quick_solution = mirror(quick_solution, 'x_max')
                elif own_id == 103:
                    quick_solution = mirror(quick_solution, 'y_min')
                worker_Bauermeister.quick_solution_list.append(quick_solution)
            
        else:
            if own_id == 100:
                pucks[puck_id]['proximity_traffic_x_min'] = False
            elif own_id == 101:
                pucks[puck_id]['proximity_traffic_y_max'] = False
            elif own_id == 102:
                pucks[puck_id]['proximity_traffic_x_max'] = False
            elif own_id == 103:
                pucks[puck_id]['proximity_traffic_y_min'] = False
            else:
                pucks[puck_id]['proximity_traffic'] = False
            proximity_condition = False
        
        if proximity_condition:
            worker_Bauermeister.proximity_list.append(puck_id)
            #print(f"Puck {puck_id} is proximity traffic. TCA: {round(tca,2)}, Dtca: {round(scalar_Dtca,3)} Dtca: {Dtca}")
            
 
    except ZeroDivisionError:
        print("ZERO DEVISION ERROR")
        pucks[puck_id]['tca'] = None
        pucks[puck_id]['Dtca'] = None
        pucks[puck_id]['proximity_traffic'] = False




def V_MIN_template():
    """
    Erstellt eine Schablone der "stall regions" an der im V Min limiter die Vektorfunktionen beschnitten werden.

    Es hanbdelt sich dabei um zwei mögliche Barrieren basierend auf der Nähe der eigenen Geschwindigkeit zur V min. (abzüglich einer Safety margin)
    Wenn die Geschwindigkeit klein ist, werden die Barrieren nahe am Startpunkt erzeugt um auch kleine Korrekturen in richtung vmin zuzulassen.
    Bei größerer Geschwindigkeit werden die Barrieren weiter entfernt und angewinkelt erzeugt ( siehe Skizze im Protokoll), um nicht einen zu großen Azimuth zu beschränken.
    
    Returns:
        None: Die Barrieren werden in einer Attributvariable des `worker_Bauermeister`-Objekts gespeichert.
    """


    r = worker_Bauermeister.V_MIN+ worker_Bauermeister.safety_margin
    velocity = worker_Bauermeister.Pucks[worker_Bauermeister.id]['velocity']
    scaler = 0.708*r #(sqrt(2)/2)
    scalar_velocity = np.linalg.norm(velocity)
    normed_velocity = velocity / scalar_velocity
    
    if scalar_velocity < 2 * scaler:
     
        vx, vy = velocity
        orthogonal_v = np.array([-vy, vx])
        start_close = normed_velocity * r
        
        barrier_close1 = (start_close, orthogonal_v)
        barrier_close2 = (start_close, -orthogonal_v)
        worker_Bauermeister.barriers = barrier_close1, barrier_close2
        

    else:
        
        center = normed_velocity * scaler
        cx, cy = center
        orthogonal = 1.1* np.array([-cy, cx])
        
        barrier1 = (2 * center, -center + orthogonal)
        barrier2 = (2 * center, -center - orthogonal)
        worker_Bauermeister.barriers = barrier1, barrier2
       

def V_MIN_limiter(vector_function, target_list):
    """
    Begrenzung der Vektor-Funktion anhand der V_MIN-template bzw der barrier-functions.

    Die Funktion beschneidet die Vektorfunktion entsprechend und die resultierenden Teile werden zu einer Liste 
    hinzugefügt. ( da aus einer Vektorfunktion hier auch zwei werden können). 
    
    Args:
        vector_function (tuple)
        target_list (list): Die Liste, in die die beschnittenen Teile der Vektor-Funktion eingefügt werden.

    Returns:
        None: Die begrenzten Teile werden der `target_list`angehängt.
    """

    #print(f"V_MIN limiter input: {vector_function}")
    if vector_function is None or worker_Bauermeister.barriers is None:
        #print("vmin limiter return none")
        return None
    
    p, d = vector_function
    dx, dy = d
    if dx == 0 and dy == 0:
        #print("direction vector war nur, return None")
        return None
    
    b1, b2 = worker_Bauermeister.barriers
    
    temp_line = (np.array([0, 0]), p)
    
    check1 = linear_intersector(temp_line, b1)
    check2 = linear_intersector(temp_line, b2)
    inside_condition = False
    
    if (check1 is None or check1[1] > 1) and (check2 is None or check2[1] > 1):
        inside_condition = True
    
    intcp1 = linear_intersector(vector_function, b1)
    intcp2 = linear_intersector(vector_function, b2)
    
    if inside_condition:
        if intcp1 is None and intcp2 is None:
            #print("completely inside")
            target_list.append(vector_function)
        elif intcp1 is None:
            #print("start inside, shortened", intcp2[0], (1 - intcp2[1]) * d)
            target_list.append((intcp2[0], (1 - intcp2[1]) * d))
            
        else:
           # print("start inside, shortened,", intcp1[0], (1 - intcp1[1]) * d)
            target_list.append((intcp1[0], (1 - intcp1[1]) * d))
            
    else:  # Start is outside
        if intcp1 is None and intcp2 is None:
            #print("passed ohne einschnitte", p, d)
            target_list.append((p, d))
            
        elif intcp1 is None:
           # print("start outside, shortened", p, d * intcp2[1])
            target_list.append((p, d * intcp2[1]))
           
        elif intcp2 is None:
           # print("start outside, shortened")
            target_list.append((p, d * intcp1[1]))
           
        else:
            #print("doppelter barrier intercept")
            if intcp1[1] < intcp2[1]:
                target_list.append((p, d * intcp1[1]))
                target_list.append((intcp2[0], (1 - intcp2[1]) * d))
                    
            elif intcp1[1] > intcp2[1]:
                target_list.append((p, d * intcp2[1]))
                target_list.append((intcp1[0], (1 - intcp1[1]) * d))
                
            else:
                target_list.append((p, d))
    #print("result min limiter:", target_list)
           
def A_MAX_limiter(relative_velocity, A):
    """
    
    Berechnet die (Zeit-) Punkte an denen frühestmöglich die Schar der Lösungsvektoren mit 
    maximaler Beschleunigung üerreicht werden kann. Gibt die neue begrenzte Vektor-Funktion zurück, falls 
    sie ereichbar ist, sonst None.Berechnung siehe Protokoll.

    Args:
        Vektorfunktion( tuple)

    Returns:
        tuple or None: Vektorfunktion mit "erreichbaren" Lösungen
    """

    if A is None or relative_velocity is None:
        return None
    a_max = worker_Bauermeister.A_MAX
    Vx, Vy = relative_velocity
    Ax, Ay = A

    A_dot_V = np.dot(A, relative_velocity)
    norm_A = np.linalg.norm(A)
    norm_V = np.linalg.norm(relative_velocity)
    
   
    coeff_t6 = 0.25 * a_max**2 
    coeff_t2 = -norm_V**2  
    coeff_t1 = -2 * A_dot_V 
    coeff_t0 = -norm_A**2 

    
    coefficients = [coeff_t6, 0, 0, 0, coeff_t2, coeff_t1, coeff_t0]
    
    roots = np.roots(coefficients)
    
    real_positive_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    
    if real_positive_roots:
        t_min = min(real_positive_roots) 
        new_vector_function =  relative_velocity, (A / t_min)
        return new_vector_function
    else:
        return None

    
    
    
    

def V_MAX_limiter(vector_function):
    """
    Beschneidet einen Vektor, wenn er das maximale Geschwindigkeitslimit (V_MAX) überschreitet. 
    Je nachdem ob der Schnittpunkt von außen oder innen erreicht wird ( outside condition) wird der ensprechende Funktionsteil zurückgegeben.
    Args:
        vector_function (tuple)

    Returns:
        vector_function or None (falls die Funktion komplett außerhalb liegt)
    """
    
    #print("V_MAX LIMITER input:", vector_function)
    
    if vector_function is None:
        print("Input was None")
        return None
    
    p, d = vector_function
    px, py = p
    radius = worker_Bauermeister.V_MAX - worker_Bauermeister.safety_margin
    outside_condition = (px**2 + py**2) >= radius**2
    intercepts = circular_intersector_list(vector_function, radius)
    
    if intercepts is not None:
        t = intercepts[0][1]  
        point = intercepts[0][0]  
        
        if not outside_condition:  
            #print("V_MAX LIMITER, slightly shortened:", p, t * d)
            return p, t * d  
        
        else:  
            if t <= 1:
               # print("Partially outside V_MAX, shortened:", point, d * (1 - t))
                return point, d * (1 - t)  
            else:
                #print("Completely outside V_MAX")
                return None  
    else:
        #print("No intercepts, returning original vector:", vector_function)
        return vector_function 
    
def minimum_resolution_function(relative_velocity, A, target_list):
    """
    Hier wird die Vektorfunktion erzeugt (siehe Protokoll) und nacheinander durch alle Limiter überprüft. 
    Ist die Funktion komplett unnereichbar, überschreitet v_max oder v_min im kompletten bereich wird None zurückgegeben.

    Args:
        relative_velocity (numpy.array) ("startpunkt der Funktion")
        A (numpy.array) ("Funktionsvektor, verschoben um relative_velocity")
        target_list (list): Die Liste, in der die resultierenden gültigen Vektor-Funktionen gespeichert werden.

    Returns:
        None: Das Ergebnis wird direkt in die `target_list` eingetragen. Gibt None zurück, wenn keine gültige
              Funktion gefunden wird.
    """
    #print("res function vor Limiter:", relative_velocity,A)
    A_MAX_checked = A_MAX_limiter(relative_velocity, A)
    if A_MAX_checked is not None:
        #print("A Max check returnt etwas", A_MAX_checked)
        V_MAX_checked = V_MAX_limiter(A_MAX_checked)
        if V_MAX_checked is not None:
            #print("Vmax check returned etwas", V_MAX_checked)
            V_MIN_limiter(V_MAX_checked, target_list)
            
        else:
            print("OUTPUT LIMTIERWAR NOONE !!!!")
            return None
    else:
        print("OUTPUT LIMITER WAR NOONE !!!!")
        return None
        
    
    
def min_limitation_function(relative_velocity, A):
    """
    Erstellt die eine mini limitation funktion. Der einfachheit halber werden diese Lösungen in der Version nicht durch die Limiter gecheckt,
    da davon auszugehen ist, dass die Lösung die in der secondary_collsion_protection erzeugt wird ein relativ langer Vektor ist, 
    und somit eher in brenzlichen Situationen zum tragen kommt, nämlich wenn zum Lösen eines Konfliktes ein anderer Puck gekreuzt werden muss.

    Args:
        relative_velocity (numpy.array)
        A (numpy.array)

    Returns:
        tuple
    """
    return relative_velocity, A

def max_limitation_function(relative_velocity, A, B):
    """
    Erstellt eine max limitation funktion. Hierbei wird der hintere Eckpunkt an der dem eigenen Puck zugewanden Seite angepeilt ( Siehe Skizze im Protokoll).
    Dabei war die annahme, dass der Puck sich (dank der proximity-Klassifizierung mit positiver tca) auf einem tangentialen Pfad nähert.

    Args:
        relative_velocity (numpy.array)
        A (numpy.array)
        B (numpy.array): Vektor um den "Peilpunkt" auf die hintere Ecke zu verschieben

    Returns:
        zwei Listen: resolution_vector_functions oder limitation_tuples. Diese können auch leer sein.
    """
    return relative_velocity, A - (2 * B)
    

def conflict_resolutions_and_limitations(own_puck, pucks):
    """
    Diese zentrale Routine bestimmt Konfliktlösungen basierend auf der relativen Position und Geschwindigkeit.
    Es wird mit dem alpha_criterion geprüft ob der eigene Puck innerhalb des tracks der Eckigen Schutzzone liegt, oder nicht. 
    Da ich dennoch gelegentlich Kollisionen beobachtet habe, die eigentlich nicht passieren dürften, habe ich zusätzlich das 
    avoidance_required -Kriterum erstellt um unabhängig vom alpha_criterion bei allen Pucks die sich bis auf weniger als zwei Längeneinehiten 
    annähern ebenfalls ein Manöver zu starten. In dem Fall werden "resolutions" erzeugt, denen gefolgt werden muss. 
    Anderenfalls werden "limitation_tuples" erzeugt, bestehend aus einer Max- und einer Minifunktion, die lediglich für den Schutz vor sekundären kollisionen bestimmt sind.
    Anders als bei den ersten Versionen finden sich hier noch große Parameter bzw Funktionen wie der "safety_squish" die deutlich drastischere Reaktionen hervorrufen sollen.

    Args:
        own_puck (dict)
        pucks (dict)

    Returns:
        tuple: Eine Liste von Vektor-Funktionen für Konfliktvermeidung (resolution_vector_functions) und eine Liste 
        von Einschränkungen (limitation_tuples), die angewendet werden müssen.
    """
    
    resolution_vector_functions = []
 
    limitation_tuples = []
    
    for puck_id, puck_data in pucks.items():
        if puck_id == own_puck['id'] or not puck_data['proximity_traffic']:
            continue
        
        #print("resolving puck nr", puck_id)
        intruder = puck_data
        relative_position = intruder['position'] - own_puck['position']
        #print("rel Pos", relative_position)
        len_rel_pos = np.linalg.norm(relative_position)
        safety_squish = 3/(-2*len_rel_pos)+1
        relative_position = 0.8*relative_position*safety_squish
        
        
        relative_velocity = (intruder['velocity'] - own_puck['velocity'])
        if relative_position is None or relative_velocity is None:
            print("rel velo war null")
            return
        relative_velocity= 1.8*relative_velocity
        #print("relative velocity", relative_velocity)
        #print(f"Type of relative_position: {type(relative_position)}, value: {relative_position}")
        #print(f"Type of relative_velocity: {type(relative_velocity)}, value: {relative_velocity}")
       # print("DTCA UND TCA:", tca, Dtca, "for Puck nr:", puck_id)
        #two_tick_velocity = relative_velocity
        
        scalar_Dtca = intruder['scalar_Dtca'] 
        if scalar_Dtca is not None:
            avoidance_required = (scalar_Dtca<4)
        else:
            avoidance_required = False
            
        scalar_relative_velocity = np.linalg.norm(relative_velocity)
        
        if scalar_relative_velocity == 0:
            scalar_relative_position = np.linalg.norm(relative_position)
            normed_relative_position = relative_position/ scalar_relative_position
            nrpx, nrpy = normed_relative_position
            orthonormal_position = np.array([-nrpy, nrpx])
            special_limit_function = relative_position - normed_relative_position - orthonormal_position , 2* orthonormal_position
            limitation_tuples.append( (special_limit_function, None)  )
            continue
            
        B = relative_velocity * 3/ scalar_relative_velocity 
        B_orthogonal = 1.5 * np.array([-B[1], B[0]]) 
        #print(f"B: {B}")
        tca = intruder['tca']
        
        A = relative_position + B 
        if tca is not None:
            tca = tca*0.8
            A_0 = (A + B_orthogonal+2*relative_velocity)/tca
            A_1 = (A - B_orthogonal+2*relative_velocity)/tca
            #print(f" tca not None: A_0: {A_0},A_1: {A_1}")
        else:
            limitation_tuples.append( ((relative_position/2,relative_velocity),None))
            minimum_resolution_function(-relative_position,-relative_velocity, resolution_vector_functions)
            
            continue
            #print("RESOLUTIONS ENTHÄLT PUCK MIT TCA NONE")
            #A_0 = (A + B_orthogonal+two_tick_velocity)*10
            #A_1 = (A - B_orthogonal+two_tick_velocity)*10
        #print(f"A_0: {A_0}, A_1:{A_1}")
        alpha_0_criterion = np.dot(B_orthogonal, A_0)
        alpha_1_criterion = np.dot(-B_orthogonal, A_1)
        #relative_velocity = 1.5* relative_velocity
        
        
        if  np.dot(relative_position, relative_velocity) > 0 and not avoidance_required:
            #print("Puck moves away")
            limitation_tuples.append( ((relative_position/2,relative_velocity),None))
                
        V_MIN_template()
        
        if alpha_0_criterion > 0 and alpha_1_criterion > 0 or avoidance_required:
           # print("mini avoidance for Puck nr", puck_id)
            minimum_resolution_function(3*relative_velocity/tca,A_0, resolution_vector_functions)
            minimum_resolution_function(3*relative_velocity/tca,A_1, resolution_vector_functions)
            #print("current esolution fcts:", resolution_vector_functions)
        
        elif alpha_0_criterion <= 0:
            #print("out of track0 for Puck", puck_id)
            limitation_tuples.append((max_limitation_function( relative_velocity, A_0, B ), min_limitation_function( relative_velocity, A_1) ) ) 
            
        elif alpha_1_criterion <=0:
            #print("out of track1 for Puck", puck_id)
            limitation_tuples.append( (max_limitation_function( relative_velocity, A_1, B ), min_limitation_function( relative_velocity, A_0) ))
            
        else:
            print("Unexpected case, problem puck:", puck_id)

    #print("RESULTATE CONFLICT AVOIDANCE:", "res:", resolution_vector_functions, "lim:", limitation_tuples)  
    
    return resolution_vector_functions, limitation_tuples


def linear_intersector(vector_function1, vector_function2):
    """
    Diese Routine bestimmt den Schnittpunkt zweier Vektor-Funktionen ( falls er existiert)
    
    Jeder Vektor wird durch einen Startpunkt und eine Richtung beschrieben. Die Routine berechnet, 
    ob und wo sich diese beiden Vektoren oder Strahlen schneiden. Wenn die Strahlen parallel oder sich aufgrund der endlichen Länge nicht schneiden , 
    wird `None` zurückgegeben.

    Args:
        vector_function1 (tuple): bestehend aus einem Startpunkt (p1) und einer Richtung (d1).
        vector_function2 (tuple): bestehend aus einem Startpunkt (p2) und einer Richtung (d2).

    Returns:
        tuple or None: Gibt den Schnittpunkt der beiden Strahlen als Tupel zurück (Schnittpunkt, t1, t2), 
        wobei `t1` und `t2` die Parameter entlang der beiden Vektoren sind. Gibt `None` zurück, wenn 
        keine Schnittstelle vorhanden ist oder die Strahlen parallel sind.
    """
#    print("INTERSECTOR CALLED")
#    print(f"vector funct 1{vector_function1} und vector funct 2 {vector_function2}")

    if vector_function1 is None or vector_function2 is None:
        return None
    
    p1, d1 = vector_function1[:2]
    p2, d2 = vector_function2[:2]
    
    dx1, dy1 = d1
    dx2, dy2 = d2
    px1, py1 = p1
    px2, py2 = p2

    # Calculate the determinant
    denominator = dx1 * dy2 - dy1 * dx2
#    print("DENOMINATOR", denominator)

    if abs(denominator) < 1e-10:
        return None

    px21 = px2 - px1
    py21 = py2 - py1
    t1 = ((px21) * dy2 - (py21) * dx2) / denominator
    t2 = ((px21) * dy1 - (py21) * dx1) / denominator

    if t1 < 0 or t2 < 0:
        return None

    # Calculate the intersection point
    intersection_point = np.array([px1 + t1 * dx1, py1 + t1 * dy1])
#    print("Intersection point:", intersection_point)
    return intersection_point, t1 ,t2


def circular_intersector_list(vector_function, radius):
    """
    Ähnlich wie der linear intersector, hier jedoch mit einem Kreis. Hauptsächlich genutzt für den V_MAX_limiter

    Args:
        vector_function
        radius (float)

    Returns:
        list or None: Eine Liste von Schnittpunkten (Punkt, t) oder None, wenn kein Schnittpunkt vorhanden ist.
    """
    
    if vector_function is None or radius is None:
        return None

    a,b = vector_function
    bx,by = b
  
    if bx ==0 and by==0:
        return None
    
    denominator = np.dot(b, b)
    p = 2 * np.dot(a, b) / denominator
    q = (np.dot(a, a) - radius**2) / denominator
    sqrt_part = (p**2 / 4) - q

    if sqrt_part >= 0:
       
        sqrt = math.sqrt(sqrt_part)
        t_1 = -p / 2 + sqrt
        t_2 = -p / 2 - sqrt
        point1 = a + t_1 * b
        point2 = a + t_2 * b
        intercepts = []
        
        if t_1 >= 0:
            intercepts.append((point1, t_1))
        if t_2 >= 0:
            intercepts.append((point2, t_2))

    
        intercepts.sort(key=lambda x: x[1])

        # 
        return intercepts if intercepts else None

    
    return None

  

def closest_approach(vector_function):
    """
    implementiert wie in der Vorlesung gezeigt
    """
    if vector_function is None:
        return None
    p, d = vector_function[:2]
    dx,dy = d
   
    if dx == dy == 0: 
        return p
    
    d_dot_d = np.dot(d, d)
    
    if d_dot_d == 0:
        t =100
    else:
        t = -np.dot(p, d) / d_dot_d
    
    if t > 0:
        if t >= 1:
            return p+d
        else:
            return p + t * d
    else:
        return p    
   

def sort_by_distance(list_of_points):
    """
    Entfernt Duplikate aus einer Liste von Punkten und sortiert die verbleibenden
    Punkte basierend auf ihrem quadratischen Abstand zum Ursprung (0, 0).
    
    Args:
    list_of_points (List[np.arrays])
    
    Returns:
    list_of_points (List[np.arrays]) , aber ohne Duplikate und nach der Lönge sortiert
    """
    unique_points = list(set(map(tuple, list_of_points)))
    sorted_points = sorted(unique_points, key=lambda point: point[0]**2 + point[1]**2)
    return [np.array(point) for point in sorted_points]

def insert_into_sorted_by_distance(list_of_points, new_point):#
    """
    Ordnet einen Punkt nach seiner Länge nach in eine Sortierte Liste ein, ohne die Reihenfolge zu stören
    """
    new_point_distance = new_point[0]**2 + new_point[1]**2
    for i, point in enumerate(list_of_points):
        if new_point_distance < point[0]**2 + point[1]**2:
            list_of_points.insert(i, new_point)
            return
    list_of_points.append(new_point)

def linear_combiner(id, list_of_resolutions):
    """
    Zentrale Routine, die eine Liste von Lösungsvektorfunktionen verarbeitet und die mögliochen Konfliktlösungspunkte (Schnittpunkte oder closest approach Points) 
    identifiziert.Diese Punkte werden danach noch überprüft ob es auch wirkliche Minimumunkte sind, indem ihr Positionsvektor nach Schnittpunkten 
    mit allen Mini-Funktioenen geprüft wird.

    Args:
        id 
        list_of_resolutions (list of tuples)

    Returns:
        list: Eine sortierte Liste von gültigen Minipunkten (numpy-Arrays),falls keine gültigen Punkte gefunden werden, wird der 
        Nullvektor zurückgegeben.
    """
    #print("COMBINER CALLED")
#    print("list of Resolutions:", list_of_resolutions)
    list_of_resolutions = [
        vector_function for vector_function in list_of_resolutions
        if vector_function is not None 
    ]
  
    if not list_of_resolutions:
       # print("Liste war leer, empty return")
        
        return np.array([0, 0])
    
    #print("_________________________ECHTE AVOIDANCE__________________________________")
    

    closest_points = [closest_approach(vector_function) for vector_function in list_of_resolutions]
    #print("closest points:", closest_points)
    intersection_points = []
    for i in range(len(list_of_resolutions)):
        for j in range(i + 1, len(list_of_resolutions)):
            intersection = linear_intersector(list_of_resolutions[i], list_of_resolutions[j])
            if intersection is not None and intersection[1] <=1 and intersection[2] <=1:
                intersection_points.append(intersection[0])
    
    #print("intersection points:", intersection_points)
            
    all_points_of_interest = intersection_points+ closest_points #+ max_min_limit_points
    
    for quick_point in worker_Bauermeister.quick_solution_list:
        if quick_point is not None:
            all_points_of_interest.append(quick_point)
            #print("QUICK POINT APPENDED!")
#    print("ALL points of interest:", all_points_of_interest) 
#    print("all points of interest", all_points_of_interest)
    
    true_mini_points = []

    for point in all_points_of_interest:
        temp_line = (np.array([0, 0]), point)
 #       print("TEMP LINE", temp_line)
        is_valid_point = True  
        
        for vector_function in list_of_resolutions:
            
            intercept_check = linear_intersector(temp_line, vector_function)
        
            if intercept_check is not None and intercept_check[1] > 1:
                is_valid_point = False  
                break
    
        if is_valid_point:
            true_mini_points.append(point)
            
#    print("true MINI", true_mini_points)
    result = sort_by_distance(true_mini_points)

    #print("RESULT COMBINER:", result)
    return result

def secondary_collision_protection(list_of_points, limitation_tuples):
    """
    Überprüft Lösungspunkte auf potenzielle sekundäre Kollisionen und passt sie bei Bedarf an.

    Diese Funktion iteriert durch eine Liste von potenziellen Lösungspunkten (true_mini_points) und überprüft, 
    ob einer dieser Punkte ( bzw Vektoren) zu einer Kollision mit einer Max-function führt. 
    Ist das der Fall ist wird geprüft oder der Vektor lang genug ist um diese Kollsion direkt zu Lösen, falls nicht
    wird der Punkt so verlängert, das er es tut und angehängt. An dieser Stelle ist garantiert, das dieser neue Punkt den Konflikt lösen könnte, allerdings
    nicht ob es dafür zu einer A_MAX, V_MAX oder V_MIN überschreitung kommt. ( erklärung dazu siehe oben)
    Andernfalls wird der Punkt zurückgegeben, wenn keine sekundäre Kollision  erkannt wird. Da die input Liste sortiert wird, kann so der kürzeste Vektor 
    werden, der den Konflikt löst.

    Args:
        list_of_points (list of np.ndarray)
        limitation_tuples (list of tuples)

    Returns:
        np.ndarray or None: Der Minipunkt, der keine Kollision verursacht, oder None, wenn alle Punkte 
        potenzielle Kollisionen erzeugen.
    """
    print("secondary coll check called, with input:", list_of_points)

    for point in list_of_points:
        if point is None:
            continue
        
       
        if np.array_equal(point, np.array([0, 0])):
            return point 
        
        for max_function, mini_function in limitation_tuples:
            temp_vector_function = (np.array([0, 0]), point) 
            intersection_result_max = linear_intersector(temp_vector_function, max_function)

           
            if intersection_result_max is None or (intersection_result_max[1] > 1):
                continue
            
            
            if mini_function is None:
                break
            
           
            temp_mini_list = []
            relative_velocity_temp, A_temp = mini_function
            minimum_resolution_function( relative_velocity_temp, A_temp, temp_mini_list)
            temp_mini_list = [function for function in temp_mini_list if function is not None]
            
            if not temp_mini_list:
                break
            
            
            for temp_mini_function in temp_mini_list:
                intersection_result_mini = linear_intersector(temp_vector_function, temp_mini_function)
                
               
                if intersection_result_mini is None:
                    break
                
                
                if 0 < intersection_result_mini[1] <= 1:
                    continue
                else:
                    
                    insert_into_sorted_by_distance(list_of_points, intersection_result_mini[0])
                    break
            else:
               
                continue
            
            
            break
        else:
            
            return point
    
   
    #print("No valid points found.")
    return None


           
def quick_secondary_collision_protection(list_of_points, limitation_tuples):
    """
    Schnelle Überprüfung auf sekundäre Kollisionen ohne Korrektur, für zeitkritische Szenarien.
    Funtkionsweise ähnlich wie oben, nur das hier Punkte bei einer sekundären Kollision einfach ausgeschlossen werden, ohne Korrekturversuch

    Args:
        list_of_points (list of np.ndarray)
        limitation_tuples (list of tuples)

    Returns:
        np.ndarray or None
    """
    for point in list_of_points:
        if point is None:
            continue
        temp_line = (np.array([0, 0]), point)
        for lim_tuple in limitation_tuples:
            if lim_tuple is None:
                continue
            max_function, _ = lim_tuple
            intersect_result = linear_intersector(temp_line, max_function)
            
            
            if intersect_result is None or intersect_result[1] >= 1:
                return point
    return None



def adaptive_performance_control(runtime):
    """
    
    Diese Funktion passt den "Horizont" des Pucks (d.h. den Zeitrahmen und den Erfassungsbereich) dynamisch an, 
    um die Rechenzeit basierend auf der aktuellen Laufzeit zu optimieren. Wenn die Berechnungen zu lange dauern, 
    wird der Horizont verkleinert, um die Leistung zu verbessern. Andernfalls werden sie erweitert, 
    um besser ausweichen zu können. Der Horizont steuert auch ob bounces beachtet werden oder nicht.

    Args:
        runtime (float): Die Laufzeit der aktuellen Berechnung.
    
    Modifies:
        worker_Bauermeister.timeframe (float) zeitlicher Horizont
        worker_Bauermeister.range (float) räumlicher Horizont
    """
#    print("ADAPTIVE PERF RUNTIME", runtime)
    max_timeframe = 5
    min_timeframe = 1  
    
    max_range = 10 
    min_range = 4
    
    if runtime >= 0.018:
        worker_Bauermeister.timeframe = max(min_timeframe, worker_Bauermeister.timeframe * 0.018/runtime)
        worker_Bauermeister.range = max(min_range, worker_Bauermeister.range-(worker_Bauermeister.range - min_range)/2 )
        #print(f"decreased timeframe {worker_Bauermeister.timeframe} and range {worker_Bauermeister.range}")
    if runtime <= 0.014: #evtl timeframe als zweite condition einführen
        worker_Bauermeister.range = min(max_range, (worker_Bauermeister.range*1.1))
        worker_Bauermeister.timeframe = min(max_timeframe, worker_Bauermeister.timeframe + (max_timeframe - worker_Bauermeister.timeframe) / 50)
        #print(f"increased timeframe {worker_Bauermeister.timeframe} and range {worker_Bauermeister.range}")
        
   
 
def moderator():
    """
    Diese Funktion wird in ruhigen Phasen aufgerufen, wenn der Treibstoffstand des Pucks 
    hoch genug ist. Sie berechnet eine Korrektur der Geschwindigkeit, um den Puck in einem 
    moderaten Geschwindigkeitsbereich zu halten.

    Returns:
        np.array: Der Geschwindigkeitskorrekturvektor, der angewendet werden soll.
    """
    if worker_Bauermeister.Pucks[worker_Bauermeister.id]['fuel']>25:
        moderation_factor = 0.2
        own_velocity = worker_Bauermeister.Pucks[worker_Bauermeister.id]['velocity']
        scalar_velocity = np.linalg.norm(own_velocity)
        scaled_velocity = 26/scalar_velocity*own_velocity
        correction_factor =-moderation_factor *((scalar_velocity-26)/16)**5
    
        command =scaled_velocity*correction_factor
    else:
        command = np.array([0,0])
    #print(f"es wurde moderiert: {command}")
    return command

def Q_spamer_preparation():
    """
    Diese Funktion sucht nach dem Puck mit der größten Zeit bis zur nächsten Kollision (TCA) (die außerhalb meines 
    zeitlichen Horizontes liegt und nicht "None" ist) und bereitet einen Vektor vor um den Puck unter v_min zu bringen
    und zwar noch bevor er meinen Horizont erreichen kann. Wenn kein Puck gefunden wird, gibt die Funktion `None` zurück.

    Returns:
        tuple or None: Ein Tupel mit dem negierten Geschwindigkeitsvektor und der ID des Ziel-Pucks, 
                       oder `None`, falls kein passender Puck gefunden wird.
    """
    #print("Q SPAMER CALLED")
    pucks = worker_Bauermeister.Pucks
    max_tca = worker_Bauermeister.timeframe+0.5
    target_puck_id = None
    target_velocity = None
    for puck_id, puck_data in pucks.items():
        tca = pucks[puck_id]['tca'] 
        if tca is None:
            continue
        if tca > max_tca:
            max_tca = tca
            target_puck_id = puck_id
            target_velocity = puck_data.get('velocity')  
    if target_puck_id is not None:
        return   -2*target_velocity, target_puck_id
    else:
        return None
   
             


def worker_Bauermeister(id, secret, q_request, q_reply):
    while True:
        if not getattr(worker_Bauermeister, 'initialized', False):
            
            print("HELLO WORLD") #der Start meines ersten "größeren" Python Projekts
            #zuerst werden wichtige Parameter des Spieles abgefragt:
            q_request.put(('GET_BOX', id))
            q_request.put(('GET_SIZE', id))
            q_request.put(('SET_NAME', 'Bauermeister', secret, id))
            
            #dann einige Variablen/Eigenschaften/Listen/Dicsionaries erzeugt
            worker_Bauermeister.Pucks = {}
            worker_Bauermeister.mirror_Pucks = {}
#            worker_Bauermeister.preliminary_solutions_list = []
            worker_Bauermeister.proximity_list = []
            worker_Bauermeister.quick_solution_list = []
            worker_Bauermeister.Q_counter = 1
            
            worker_Bauermeister.box = None
            worker_Bauermeister.n_workers = None
            worker_Bauermeister.cycle_count = 0
            worker_Bauermeister.id = id
            
            worker_Bauermeister.timeframe = 1.3
            worker_Bauermeister.range = 4
            worker_Bauermeister.safety_margin = 1
            worker_Bauermeister.barriers = None
            #falls etwas schief geht gehe ich von folgenden werten aus: 
            worker_Bauermeister.X_MIN = 0 
            worker_Bauermeister.X_MAX = 120 
            worker_Bauermeister.Y_MIN = 0
            worker_Bauermeister.Y_MAX = 75
            
            worker_Bauermeister.V_MAX = 42
            worker_Bauermeister.V_MIN = 10
            worker_Bauermeister.A_MAX = 100#
            #anganfangs hatte ich hier eine kurze sleep phase um auf antwort zu warten, es hat aber auch ohne funktionert
            try:
                while True:
                    reply = q_reply.get()
                    
                    match reply:  
                        case ('GET_BOX', box):
                            worker_Bauermeister.box = box
                            X_MIN, X_MAX = box.get_x_limits()
                            Y_MIN, Y_MAX = box.get_y_limits()
                            
                            worker_Bauermeister.X_MIN = X_MIN
                            worker_Bauermeister.X_MAX = X_MAX
                            worker_Bauermeister.Y_MIN = Y_MIN
                            worker_Bauermeister.Y_MAX = Y_MAX
                            
                        case ('GET_SIZE', n_workers):
                            worker_Bauermeister.n_workers = n_workers
                            print("ANZAHL PUCKS", n_workers)
                        case _:
                            continue

                    if worker_Bauermeister.box is not None and worker_Bauermeister.n_workers is not None: # wenn erfolgreich, aus dem loop aussteigen
                        break
            except queue.Empty: # wenn keine Antwort erhalten wurde, wird es erneut versucht bis der worker initialisiert ist
                worker_Bauermeister.initialized = False
                break
            # mit bekannter Puck anzahl werden nun alle Pucks erstmalig abgefragt
            for n in range(worker_Bauermeister.n_workers):
                q_request.put(('GET_PUCK', n, id))
                
            worker_Bauermeister.initialized = True
            #print("INITIALIZED", worker_Bauermeister.initialized )
            time.sleep(0.005)
            #print("END OF INIT LOOP")
            break


    # MAIN LOOP
    
    
        
    while True:
       
       # print("known alive Pucks:", len(worker_Bauermeister.Pucks) )
        #print("Start of Main Loop")
        start_time = time.time() #Zeitstempel für performance berechnungen und entscheidungen ob später genug Zeit für ausführliche routinen ist oder nicht
        print("CYCLES:", worker_Bauermeister.cycle_count)
        print(" ----> PROXY IDS:", worker_Bauermeister.proximity_list, "<----")
        #for puck_id in worker_Bauermeister.proximity_list:
            #print(worker_Bauermeister.Pucks[puck_id])
#        print("TIME before REQUEST QUEUE", round(((time.time() - start_time) / 0.02), 2))

        
        # PUCKS REQUESTEN
        
        #zunächst werden alle Pucks abgefragt um aktuelle Daten zu haben. Das geht schneller als ich dachte, anfangs hatte ich das nur für jeden 10. cycle geplant.
        if worker_Bauermeister.cycle_count % 2 == 0 or not worker_Bauermeister.proximity_list:
            for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                if puck_data['alive']:
                    q_request.put(('GET_PUCK', puck_id, id))
            q_request.put(('GET_PUCK', id, id))
            
        else:
            for puck_id, puck_data in worker_Bauermeister.Pucks.items(): 
                if puck_data['alive'] and puck_data['proximity_traffic']:
                    q_request.put(('GET_PUCK', puck_id, id))
            q_request.put(('GET_PUCK', id, id))
#        print("TIME after REQUEST QUEUE", round(((time.time() - start_time) / 0.02), 2))
        
        
        
        # REPLY QUEUE ABHÖREN
        
        
        while True:
            try:
                response = q_reply.get(block=False)
               

                if response is not None:
                    
                    match response:
                        case ('GET_PUCK', puck):
                            add_or_update_puck(puck, worker_Bauermeister.Pucks)
                        case ('SET_ACCELERATION', a):
                            if a is not None:
                                ax, ay = a  #deleten später
                                if ax**2 + ay**2 > 0.001: 
                                    print(a,">>>>>>>>>>>>>>>ACC COMMAND RECIEVED<<<<<<<<<<<<<<<<") # zur Kontrolle ob ausgewichen wurde, kleine Lösungen kommen warscheinlich vom Moderator
                        case ('GET_BOX', box):
                            worker_Bauermeister.box = box
                        case ('GET_SIZE', n_workers):
                            worker_Bauermeister.n_workers = n_workers  
                            
            except queue.Empty:
                break 
        if not worker_Bauermeister.Pucks[worker_Bauermeister.id]['alive']: #um nach Kollision das log besser lesen zu können und abstürze zu vermeiden
            break
        # create mirror pucks
        create_mirror_pucks(worker_Bauermeister.Pucks)        
     

        #PROXY CHECK
        
        
        if worker_Bauermeister.cycle_count % 2 == 0: # um rechenzeit zu sparen nur jedes zweite mal
            worker_Bauermeister.proximity_list= []
            worker_Bauermeister.quick_solution_list = []
            
            
            for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                
                is_proximity(puck_id, worker_Bauermeister.id, worker_Bauermeister.Pucks)
                
            # ab hier wird die notwendigkeit einer der bounce überprüfung gecheckt, nämlich genau dann, wenn sich eine( oder mehr) Wand in meinem Horizont befindet
            very_very_large_time = 100 

            pos_x, pos_y = worker_Bauermeister.Pucks[id]['position']
            vel_x, vel_y = worker_Bauermeister.Pucks[id]['velocity']

            distance_to_left_wall = pos_x 
            distance_to_right_wall = worker_Bauermeister.X_MAX - pos_x
            distance_to_floor = pos_y 
            distance_to_ceiling = worker_Bauermeister.Y_MAX - pos_y

            time_to_left_wall = distance_to_left_wall / abs(vel_x) if vel_x < 0 else very_very_large_time
            time_to_right_wall = distance_to_right_wall / abs(vel_x) if vel_x > 0 else very_very_large_time
            time_to_floor = distance_to_floor / abs(vel_y) if vel_y < 0 else very_very_large_time
            time_to_ceiling = distance_to_ceiling / abs(vel_y) if vel_y > 0 else very_very_large_time
            
            worker_Bauermeister.left_danger = False # innerhalb der is_proximity routine werden diese Flags korrekt gesetzt
            worker_Bauermeister.right_danger = False
            worker_Bauermeister.ceiling_danger = False
            worker_Bauermeister.floor_danger = False

            if distance_to_left_wall < worker_Bauermeister.range or time_to_left_wall < worker_Bauermeister.timeframe:
                for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                    if puck_data['alive']:
                        is_proximity(puck_id, 100, worker_Bauermeister.Pucks)
            if distance_to_right_wall < worker_Bauermeister.range or time_to_right_wall < worker_Bauermeister.timeframe:
                for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                    if puck_data['alive']:
                        is_proximity(puck_id, 102, worker_Bauermeister.Pucks)
            if distance_to_floor < worker_Bauermeister.range or time_to_floor < worker_Bauermeister.timeframe:
                for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                    if puck_data['alive']:
                        is_proximity(puck_id, 103, worker_Bauermeister.Pucks)
            if distance_to_ceiling < worker_Bauermeister.range or time_to_ceiling < worker_Bauermeister.timeframe:
                for puck_id, puck_data in worker_Bauermeister.Pucks.items():
                    if puck_data['alive']:
                        is_proximity(puck_id, 101, worker_Bauermeister.Pucks)


        # CALL RESOLUTION ON NORMAL PUCKS
        
        # hier beginnt die eigentliche Ausweichberechnung
        if id in worker_Bauermeister.Pucks:
#            print("OWN ID FOUND!")
            resolution_vector_functions, limitation_tuples = conflict_resolutions_and_limitations(worker_Bauermeister.Pucks[id], worker_Bauermeister.Pucks)
        else:
            resolution_vector_functions = []
            limitation_tuples = []
#            print(f"OWN ID {id} NOT IN DICTIONARY!!!")
       
        #CALL RESLOUTION ON MIRROR PUCKS
        if worker_Bauermeister.left_danger: # hier erfolgt die erzeugung der Ausweichempfehlungen für die jeweiligen gespiegelten Pucks
            left_resolution_vector_functions, left_limitation_tuples = conflict_resolutions_and_limitations(worker_Bauermeister.mirror_Pucks[100], worker_Bauermeister.Pucks)
            for function in left_resolution_vector_functions:
                mirrored_function = mirror_of_function(function, 'x_min')  #und die Rückspiegelung der Vektorfunktionen 
                resolution_vector_functions.append(mirrored_function)
    
   
            for function_tuple in left_limitation_tuples:
                max_function, mini_function = function_tuple
                mirrored_max_function = mirror_of_function(max_function, 'x_min')
                mirrored_mini_function = mirror_of_function(mini_function, 'x_min')
                mirrored_tuple = (mirrored_max_function, mirrored_mini_function)
                limitation_tuples.append(mirrored_tuple)

        if worker_Bauermeister.right_danger:
            right_resolution_vector_functions, right_limitation_tuples = conflict_resolutions_and_limitations(worker_Bauermeister.mirror_Pucks[102], worker_Bauermeister.Pucks )
            for function in right_resolution_vector_functions:
                mirrored_function = mirror_of_function(function, 'x_max')
                resolution_vector_functions.append(mirrored_function)
            for function_tuple in right_limitation_tuples:
                max_function, mini_function = function_tuple
                mirrored_max_function = mirror_of_function(max_function, 'x_max')
                mirrored_mini_function = mirror_of_function(mini_function, 'x_max')
                mirrored_tuple = (mirrored_max_function, mirrored_mini_function)
                limitation_tuples.append(mirrored_tuple)

        if worker_Bauermeister.floor_danger:
            floor_resolution_vector_functions, floor_limitation_tuples = conflict_resolutions_and_limitations(worker_Bauermeister.mirror_Pucks[103], worker_Bauermeister.Pucks)
            for function in floor_resolution_vector_functions:
                mirrored_function = mirror_of_function(function, 'y_min')
                resolution_vector_functions.append(mirrored_function)
            for function_tuple in floor_limitation_tuples:
                max_function, mini_function = function_tuple
                mirrored_max_function = mirror_of_function(max_function, 'y_min')
                mirrored_mini_function = mirror_of_function(mini_function, 'y_min')
                mirrored_tuple = (mirrored_max_function, mirrored_mini_function)
                limitation_tuples.append(mirrored_tuple)

        if worker_Bauermeister.ceiling_danger:
            ceiling_resolution_vector_functions, ceiling_limitation_tuples = conflict_resolutions_and_limitations(worker_Bauermeister.mirror_Pucks[101], worker_Bauermeister.Pucks)
            for function in ceiling_resolution_vector_functions:
                mirrored_function = mirror_of_function(function, 'y_max')
                resolution_vector_functions.append(mirrored_function)
            for function_tuple in ceiling_limitation_tuples:
                max_function, mini_function = function_tuple
                mirrored_max_function = mirror_of_function(max_function, 'y_max')
                mirrored_mini_function = mirror_of_function(mini_function, 'y_max')
                mirrored_tuple = (mirrored_max_function, mirrored_mini_function)
                limitation_tuples.append(mirrored_tuple)


        if resolution_vector_functions is None: #um Fehlermeldungen zu vermeiden
            resolution_vector_functions = []
        if limitation_tuples is None:
            limitation_tuples = []
        #print("TIME after RESOLUTIONS", round(((time.time() - start_time) / 0.02), 2))   
        if not resolution_vector_functions:
            #print("no avoidance necessary")
            final_result = np.array([0, 0]) #ich gebe (0,0) zurück (und nicht "None" damit ich eine möglicherweise vorher erteiltes Ausweichkommando überschreibe
        else: # hier beginnt die verarbeitung der Funktionen
            preliminary_solutions_list = linear_combiner(id, resolution_vector_functions)
            
            if preliminary_solutions_list is None or len(preliminary_solutions_list) == 0:
                preliminary_solutions_list = [np.array([0, 0])]  
            timecheck = time.time() - start_time
            # hier wird je nach verbleibender Zeit geprüft ob es sekundäre Kollisionen gibt
            if len(limitation_tuples) > 0 and len(preliminary_solutions_list) > 0:
                if timecheck < 0.01:
                    secondary_collision_checked_list = secondary_collision_protection(preliminary_solutions_list, limitation_tuples)
                    final_result = secondary_collision_checked_list
                elif timecheck < 0.017:
                    final_result = quick_secondary_collision_protection(preliminary_solutions_list, limitation_tuples)
                else:
                    final_result = preliminary_solutions_list[0] #im zeitkritischsten Fall wird einfach der erstbeste Vektor verwendet
            else:
                final_result = preliminary_solutions_list[0]

        #sollte kein Ausweichen nötig sein und maximal ein Puck in der Nähe wird die geschwindigkeit angepasst
        if len(limitation_tuples)==0 and len(worker_Bauermeister.proximity_list ) <=1: 
           #print("MODERATOR CALLED")
           final_result = moderator()
            
            
        #print("FINAL RESULT:", final_result)
        
        q_request.put(('SET_ACCELERATION',  final_result, secret, id))
         
        #print("TIME after COMPLETE AVOIDANCE",  round(((time.time() - start_time) / 0.02), 2), "<--")
        
        #im Anschlluss wird die geschwindigkeit der Berehcnung geprüft und ggf angepasst, in dem sich der Horizont verändert:
        elapsed_time = time.time() - start_time
        adaptive_performance_control(elapsed_time)
        elapsed_time2 = time.time() - start_time
        if elapsed_time2 < 0.0016: # auf meinem Rehcner benötigt der q_spamer 0.02 bis 0.03 Sekunden, daher dieses zeitlimit
            Q_data = Q_spamer_preparation()
            if Q_data is not None:
                new_target_velocity, target_puck_id = Q_data 
                #print("QSPAM:", new_target_velocity, target_puck_id)
                #start_spam = time.time()
                for s in range(worker_Bauermeister.Q_counter, worker_Bauermeister.Q_counter +10):
                    q_request.put(('SET_ACCELERATION',   new_target_velocity , s,target_puck_id))
                worker_Bauermeister.Q_counter = worker_Bauermeister.Q_counter +10
                if worker_Bauermeister.Q_counter > 10**6:
                    worker_Bauermeister.Q_counter =1
             
                #print("SPAMTIME", time.time()-start_spam)
        
        
        elapsed_time3 = time.time() - start_time
        sleep_time = max(0, 0.02 - elapsed_time3)
        time.sleep(sleep_time) 
        # hier wird der rythmus an den serverrythmus synchronisiert um nicht in die erste Zeit eines neuen Ticks zu geraten und eine bereits vorhandenes Kommando neu zu berehcnen und dann veraltete Daten zu senden.

        worker_Bauermeister.cycle_count += 1
