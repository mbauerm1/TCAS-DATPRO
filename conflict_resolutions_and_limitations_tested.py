import numpy as np
import matplotlib.pyplot as plt


Pucks = {}

def minimum_resolution_function(A, relative_velocity):
    return relative_velocity, A

def maximum_resolution_function(A, B, relative_velocity):
    return relative_velocity, A - (2 * B)

def minimum_on_boundary_resolution_function(index, A, B_orthogonal, relative_velocity):
    multiplier = 0.4 if index == 0 else -0.4
    B = multiplier * B_orthogonal
    return relative_velocity, B + A

def conflict_resolutions_and_limitations(own_id):
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
        print("B ortho", B_orthogonal)
        A_0 = relative_position + B + B_orthogonal
        A_1 = relative_position + B - B_orthogonal
        print("A_0", A_0, "A_1", A_1)
        alpha_0_criterion = np.dot(B_orthogonal, A_0)
        alpha_1_criterion = np.dot(-B_orthogonal, A_1)
        print("alpha0", alpha_0_criterion)
        print("alpha1", alpha_1_criterion)

        if alpha_0_criterion > 0 and alpha_1_criterion > 0:
            resolution_vector_functions.append(minimum_resolution_function(A_0, relative_velocity))
            resolution_vector_functions.append(minimum_resolution_function(A_1, relative_velocity))
            print("Colision! Mini avoidance beide richtungen")
        elif alpha_0_criterion == 0:
            resolution_vector_functions.append(minimum_on_boundary_resolution_function(0, A_0, B_orthogonal, relative_velocity))
            resolution_vector_functions.append(minimum_resolution_function(A_1, relative_velocity))
            print("Gerade noch auf der boundary von A_0, avoidance in beide richtungen")
        elif alpha_1_criterion == 0:
            resolution_vector_functions.append(minimum_resolution_function(A_0, relative_velocity))
            resolution_vector_functions.append(minimum_on_boundary_resolution_function(1, A_1, B_orthogonal, relative_velocity))
            print("Gerade noch auf der boundary von A_1, avoidance in beide richtungen")
        elif alpha_0_criterion < 0 and alpha_1_criterion >= 0:
            
            limitation_vector_functions.append((minimum_resolution_function(A_1, relative_velocity), maximum_resolution_function(A_0, -B_orthogonal, relative_velocity)))
           
            print("keine kollision, aber secondary collision möglich")
        
        elif alpha_1_criterion < 0 and alpha_0_criterion >= 0:
            
            limitation_vector_functions.append((minimum_resolution_function(A_0, relative_velocity), maximum_resolution_function(A_1, B_orthogonal, relative_velocity)))
            
            print("keine kollision, aber secondary collision möglich")
        else:
            print("Unexpected case, problem puck:", puck_id)

    return resolution_vector_functions, limitation_vector_functions, relative_position, relative_velocity, A_0, A_1

 # hier beliebige Puckdaten eingeben
Pucks[1] = {
    'id': 1,
    'name': 'OwnPuck',
    'position': np.array([0.0, 0.0]),
    'velocity': np.array([0.0, 0.0]),
    'acceleration': np.array([0.0, 0.0]),
    'timestamp': 123456.789,
    'fuel': 50.0,
    'alive': True,
    'proximity_traffic': False,
    'tca': None,
    'Dtca': None
}

 # hier beliebige Puckdaten eingeben
Pucks[2] = {
    'id': 2,
    'name': 'TestPuck',
    'position': np.array([0, -4]),
    'velocity': np.array([1.1, 0]),
    'acceleration': np.array([0.0, 0.0]),
    'timestamp': 123456.789,
    'fuel': 50.0,
    'alive': True,
    'proximity_traffic': True,  
    'tca': None,
    'Dtca': None
}


own_id = 1
resolution_vector_functions, limitation_vector_functions, relative_position, relative_velocity, A_0, A_1 = conflict_resolutions_and_limitations(own_id)

print("resolutions", resolution_vector_functions, "limitations", limitation_vector_functions)
print("pos", relative_position, "VEL", relative_velocity)


plt.figure(figsize=(10, 10))
t = 5  


plt.quiver(relative_position[0], relative_position[1], relative_velocity[0], relative_velocity[1], angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5)
plt.scatter(relative_position[0], relative_position[1], color='red')  # Mark the starting point



for func in resolution_vector_functions:
    start, direction = func
    plt.quiver(start[0], start[1], direction[0] * t, direction[1] * t, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)

for func in limitation_vector_functions:
    for limit_func in func:
        start, direction = limit_func
        plt.quiver(start[0], start[1], direction[0] * t, direction[1] * t, angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.5)


plt.quiver(relative_position[0], relative_position[1], A_0[0] - relative_position[0], A_0[1] - relative_position[1], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.5, label='A_0')
plt.quiver(relative_position[0], relative_position[1], A_1[0] - relative_position[0], A_1[1] - relative_position[1], angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.5, label='A_1')


plt.scatter(0, 0, color='black', s=50, zorder=5)

plt.xlim(-5, 5)
plt.ylim(-5, 7)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Relative Velocity and Vector Functions')
plt.legend()
plt.grid()
plt.show()
