import numpy as np

def initialize_drones(num_drones, min_distance=0.5, space_limits=(-4,4), max_altitude=0):
    drones = np.zeros((num_drones, 3))
    min_z = 0
    max_z = max_altitude
    
    drones[0] = np.array([
        np.random.uniform(space_limits[0], space_limits[1]),
        np.random.uniform(space_limits[0], space_limits[1]),
        np.random.uniform(min_z, max_z)
    ])
    
    for i in range(1, num_drones):
        valid_position = False
        max_attempts = 100
        attempts = 0
        
        while not valid_position and attempts < max_attempts:
            candidate = np.array([
                np.random.uniform(space_limits[0], space_limits[1]),
                np.random.uniform(space_limits[0], space_limits[1]),
                np.random.uniform(min_z, max_z)
            ])
            
            distances = np.linalg.norm(drones[:i] - candidate, axis=1)
            if np.all(distances >= min_distance):
                valid_position = True
                drones[i] = candidate
            
            attempts += 1
            
        if not valid_position:
            raise ValueError(f"Could not find valid position for drone {i} after {max_attempts} attempts")
    
    perturbations = np.random.uniform(-0.1, 0.1, drones.shape)
    drones += perturbations
    
    drones[:, 2] = np.clip(drones[:, 2], min_z, max_z)
    return drones