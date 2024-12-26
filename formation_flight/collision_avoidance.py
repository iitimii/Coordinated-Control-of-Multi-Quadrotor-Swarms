import numpy as np

def collision_avoidance(target_xyzs, current_states, safe_distance, 
                       influence_distance=None, repulsion_strength=1.0,
                       max_adjustment=0.5):
    """
    Calculate collision-free adjusted positions for multiple drones using smooth repulsion forces.
    
    Args:
        target_xyzs (np.ndarray): Target positions for each drone (n_drones, 3)
        current_states (np.ndarray): Current drone states including position (n_drones, state_dim)
        safe_distance (float): Minimum allowed distance between drones
        influence_distance (float): Distance at which repulsion begins (defaults to 1.5 * safe_distance)
        repulsion_strength (float): Scaling factor for repulsion force
        max_adjustment (float): Maximum position adjustment per step
        
    Returns:
        np.ndarray: Adjusted target positions accounting for collision avoidance
    """
    if influence_distance is None:
        influence_distance = 1.5 * safe_distance
        
    adjusted_positions = np.copy(target_xyzs)
    num_drones = target_xyzs.shape[0]
    
    # Calculate all pairwise distances and vectors at once
    current_positions = current_states[:, :3]
    position_differences = current_positions[:, np.newaxis] - current_positions
    distances = np.linalg.norm(position_differences, axis=2)
    
    # Create a mask for pairs that need repulsion (excluding self-pairs)
    needs_repulsion = (distances < influence_distance) & (distances > 0)
    
    for i in range(num_drones):
        # Calculate total repulsion force from all nearby drones
        total_repulsion = np.zeros(3)
        
        for j in range(num_drones):
            if needs_repulsion[i, j]:
                distance = distances[i, j]
                delta_pos = position_differences[i, j]
                
                # Smooth repulsion force that increases as distance decreases
                repulsion_factor = np.exp(-(distance - safe_distance) / safe_distance)
                normalized_direction = delta_pos / (distance + 1e-6)
                
                # Calculate repulsion force with smooth falloff
                repulsion = (repulsion_strength * repulsion_factor * 
                           normalized_direction * (influence_distance - distance))
                
                # Add weighted repulsion based on how close drones are to their targets
                progress_to_target_i = np.linalg.norm(target_xyzs[i] - current_positions[i])
                progress_to_target_j = np.linalg.norm(target_xyzs[j] - current_positions[j])
                
                # Give higher priority to drones closer to their targets
                priority_weight = progress_to_target_j / (progress_to_target_i + progress_to_target_j + 1e-6)
                total_repulsion += repulsion * priority_weight
        
        # Apply repulsion with magnitude limiting
        repulsion_magnitude = np.linalg.norm(total_repulsion)
        if repulsion_magnitude > max_adjustment:
            total_repulsion = total_repulsion * (max_adjustment / repulsion_magnitude)
            
        adjusted_positions[i] += total_repulsion
    
    return adjusted_positions
