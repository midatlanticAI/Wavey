#!/usr/bin/env python3
"""
WAVE MAZE ALPHABET SOUP
Big complex mazes where entities must discover wave-based navigation
Pure wave physics determines pathfinding success/failure
"""
import random
import math
from collections import defaultdict

class MazeEnvironment:
    """Large complex maze with wave propagation physics"""
    
    def __init__(self, size=200, complexity=0.3):
        self.size = size
        self.maze = self.generate_complex_maze(size, complexity)
        self.goals = self.place_multiple_goals()
        self.wave_reflectors = self.identify_wave_reflectors()
        self.wave_channels = self.identify_wave_channels()
        
    def generate_complex_maze(self, size, complexity):
        """Generate large complex maze with multiple paths"""
        # Start with all walls
        maze = [[1 for _ in range(size)] for _ in range(size)]
        
        # Create multiple starting regions
        regions = []
        for _ in range(8):  # 8 open regions
            center_x = random.randint(10, size-10)
            center_y = random.randint(10, size-10)
            radius = random.randint(8, 15)
            regions.append((center_x, center_y, radius))
            
            # Clear circular region
            for x in range(max(0, center_x - radius), min(size, center_x + radius)):
                for y in range(max(0, center_y - radius), min(size, center_y + radius)):
                    dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        maze[x][y] = 0  # Open space
        
        # Create connecting corridors between regions
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if random.random() < 0.6:  # 60% chance to connect regions
                    self.carve_corridor(maze, regions[i], regions[j])
        
        # Add random corridors for complexity
        for _ in range(size * 2):
            start_x = random.randint(1, size-2)
            start_y = random.randint(1, size-2)
            if maze[start_x][start_y] == 0:  # Start from open space
                self.random_walk_corridor(maze, start_x, start_y, length=random.randint(10, 30))
        
        # Add dead ends and loops
        for _ in range(size):
            x = random.randint(1, size-2)
            y = random.randint(1, size-2)
            if maze[x][y] == 1 and random.random() < complexity:
                # Chance to break wall and create complexity
                maze[x][y] = 0
        
        return maze
    
    def carve_corridor(self, maze, region1, region2):
        """Carve corridor between two regions"""
        x1, y1, _ = region1
        x2, y2, _ = region2
        
        # Simple L-shaped corridor
        current_x, current_y = x1, y1
        
        # Move horizontally first
        while current_x != x2:
            maze[current_x][current_y] = 0
            if current_x < x2:
                current_x += 1
            else:
                current_x -= 1
        
        # Then move vertically
        while current_y != y2:
            maze[current_x][current_y] = 0
            if current_y < y2:
                current_y += 1
            else:
                current_y -= 1
    
    def random_walk_corridor(self, maze, start_x, start_y, length):
        """Create random walk corridor"""
        x, y = start_x, start_y
        
        for _ in range(length):
            maze[x][y] = 0
            
            # Random direction
            directions = [(0,1), (0,-1), (1,0), (-1,0)]
            dx, dy = random.choice(directions)
            
            new_x = max(1, min(self.size-2, x + dx))
            new_y = max(1, min(self.size-2, y + dy))
            
            x, y = new_x, new_y
    
    def place_multiple_goals(self):
        """Place multiple goals in open areas"""
        goals = []
        attempts = 0
        
        while len(goals) < 5 and attempts < 1000:  # 5 goals maximum
            x = random.randint(5, self.size-5)
            y = random.randint(5, self.size-5)
            
            if self.maze[x][y] == 0:  # Open space
                # Check it's not too close to other goals
                too_close = False
                for gx, gy, _ in goals:
                    if math.sqrt((x-gx)**2 + (y-gy)**2) < 20:
                        too_close = True
                        break
                
                if not too_close:
                    reward = random.uniform(100, 500)  # Variable goal rewards
                    goals.append((x, y, reward))
            
            attempts += 1
        
        return goals
    
    def identify_wave_reflectors(self):
        """Identify wall patterns that create interesting wave reflections"""
        reflectors = []
        
        for x in range(1, self.size-1):
            for y in range(1, self.size-1):
                if self.maze[x][y] == 1:  # Wall
                    # Count surrounding walls
                    wall_count = sum(1 for dx in [-1,0,1] for dy in [-1,0,1] 
                                   if self.maze[x+dx][y+dy] == 1)
                    
                    # Corners and edges create interesting reflections
                    if wall_count >= 6:  # Dense wall area
                        reflectors.append((x, y, 'dense'))
                    elif wall_count <= 3:  # Isolated wall
                        reflectors.append((x, y, 'isolated'))
        
        return reflectors
    
    def identify_wave_channels(self):
        """Identify corridor patterns that channel waves effectively"""
        channels = []
        
        for x in range(2, self.size-2):
            for y in range(2, self.size-2):
                if self.maze[x][y] == 0:  # Open space
                    # Check for corridor-like patterns
                    
                    # Horizontal corridor
                    if (self.maze[x][y-1] == 1 and self.maze[x][y+1] == 1 and
                        self.maze[x-1][y] == 0 and self.maze[x+1][y] == 0):
                        channels.append((x, y, 'horizontal'))
                    
                    # Vertical corridor
                    elif (self.maze[x-1][y] == 1 and self.maze[x+1][y] == 1 and
                          self.maze[x][y-1] == 0 and self.maze[x][y+1] == 0):
                        channels.append((x, y, 'vertical'))
        
        return channels
    
    def is_valid_position(self, x, y):
        """Check if position is valid (not a wall)"""
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.maze[int(x)][int(y)] == 0
    
    def calculate_wave_propagation(self, source_x, source_y, freq, amplitude, target_x, target_y):
        """Calculate wave propagation through maze considering walls and reflections"""
        
        # Direct line-of-sight component
        distance = math.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
        if distance == 0:
            return amplitude
        
        # Check for direct path (simplified ray casting)
        direct_blocked = False
        steps = int(distance)
        if steps > 0:
            for i in range(1, steps):
                check_x = source_x + (target_x - source_x) * i / steps
                check_y = source_y + (target_y - source_y) * i / steps
                if not self.is_valid_position(check_x, check_y):
                    direct_blocked = True
                    break
        
        total_amplitude = 0
        
        # Direct path component
        if not direct_blocked:
            # Wave attenuation with distance
            direct_amplitude = amplitude / (1 + distance * 0.1)
            # Phase delay
            phase = freq * distance * 0.1
            total_amplitude += direct_amplitude * math.cos(phase)
        
        # Reflection components from nearby walls
        reflection_amplitude = 0
        for wall_x, wall_y, wall_type in self.wave_reflectors:
            wall_dist_from_source = math.sqrt((wall_x - source_x)**2 + (wall_y - source_y)**2)
            wall_dist_to_target = math.sqrt((target_x - wall_x)**2 + (target_y - wall_y)**2)
            
            if wall_dist_from_source < 50 and wall_dist_to_target < 50:  # Close enough for reflection
                # Reflection coefficient based on wall type
                if wall_type == 'dense':
                    reflection_coeff = 0.7
                elif wall_type == 'isolated':
                    reflection_coeff = 0.3
                else:
                    reflection_coeff = 0.5
                
                # Reflected wave amplitude (decreases with total path length)
                total_reflection_distance = wall_dist_from_source + wall_dist_to_target
                reflected_amplitude = (amplitude * reflection_coeff / 
                                     (1 + total_reflection_distance * 0.05))
                
                # Reflection phase
                reflection_phase = freq * total_reflection_distance * 0.1
                reflection_amplitude += reflected_amplitude * math.cos(reflection_phase)
        
        total_amplitude += reflection_amplitude * 0.5  # Reduce reflection contribution
        
        # Channel enhancement for waves traveling through corridors
        for channel_x, channel_y, channel_type in self.wave_channels:
            channel_dist = math.sqrt((channel_x - target_x)**2 + (channel_y - target_y)**2)
            
            if channel_dist < 10:  # Wave passes through channel
                # Channels focus waves
                channel_enhancement = 1.5 if channel_type in ['horizontal', 'vertical'] else 1.0
                total_amplitude *= channel_enhancement
                break
        
        return max(0, total_amplitude)

class WaveMazeEntity:
    """Entity that must discover wave-based maze navigation"""
    
    def __init__(self, entity_id, start_x, start_y):
        self.id = entity_id
        self.x = start_x
        self.y = start_y
        self.energy = 100.0
        self.age = 0
        
        # Wave sensing for navigation
        self.wave_sensors = []
        for _ in range(random.randint(4, 10)):  # Random number of wave sensors
            freq = random.uniform(0.5, 5.0)
            sensitivity = random.uniform(0.3, 1.0)
            directional_bias = random.uniform(0, 2*math.pi)  # Preferred sensing direction
            self.wave_sensors.append({
                'frequency': freq,
                'sensitivity': sensitivity, 
                'direction_bias': directional_bias
            })
        
        # Wave emission for echolocation/communication
        self.wave_emitters = []
        for _ in range(random.randint(1, 3)):
            freq = random.uniform(1.0, 8.0)
            max_power = random.uniform(0.5, 1.5)
            self.wave_emitters.append({
                'frequency': freq,
                'max_power': max_power
            })
        
        # Navigation state
        self.path_memory = []  # Remember successful paths
        self.wall_memory = []  # Remember wall locations
        self.goal_memory = []  # Remember goal locations and rewards
        self.failed_directions = []  # Remember failed movement attempts
        
        # Learning parameters
        self.exploration_tendency = random.uniform(0.2, 0.8)
        self.memory_retention = random.uniform(0.1, 0.9)
        self.risk_tolerance = random.uniform(0.1, 0.7)
        
        # Navigation discoveries
        self.navigation_strategies = []
        self.wave_navigation_discoveries = []
        
    def emit_navigation_waves(self, maze_env):
        """Emit waves for echolocation/navigation"""
        for emitter in self.wave_emitters:
            if self.energy > 10:  # Need energy to emit
                # Emit wave in current environment
                # This is handled by the maze environment's wave physics
                
                # Cost energy for emission
                emission_cost = emitter['max_power'] * 2
                self.energy -= emission_cost
    
    def sense_maze_waves(self, maze_env, all_entities):
        """Sense wave reflections and other entities' waves"""
        sensed_data = {'directions': {}, 'goals': [], 'obstacles': [], 'entities': []}
        
        # Sample waves in different directions around the entity
        sample_directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        for dx, dy in sample_directions:
            direction_strength = 0
            sample_distance = 15  # How far to sample
            
            for sensor in self.wave_sensors:
                # Calculate sensor alignment with this direction
                sensor_angle = sensor['direction_bias']
                direction_angle = math.atan2(dy, dx)
                angle_diff = abs(sensor_angle - direction_angle)
                alignment = math.cos(angle_diff)  # 1.0 for perfect alignment, 0 for perpendicular
                
                if alignment > 0:  # Sensor can detect in this direction
                    sample_x = self.x + dx * sample_distance
                    sample_y = self.y + dy * sample_distance
                    
                    # Check for wave reflections from walls
                    wave_amplitude = 0
                    
                    # Emit wave and calculate reflection
                    for emitter in self.wave_emitters:
                        if abs(emitter['frequency'] - sensor['frequency']) < 1.0:  # Matching freq
                            reflection = maze_env.calculate_wave_propagation(
                                self.x, self.y, emitter['frequency'], emitter['max_power'],
                                sample_x, sample_y
                            )
                            wave_amplitude += reflection
                    
                    # Apply sensor sensitivity and alignment
                    detected_strength = wave_amplitude * sensor['sensitivity'] * alignment
                    direction_strength += detected_strength
            
            sensed_data['directions'][(dx, dy)] = direction_strength
        
        # Detect goals through wave signatures (goals emit distinctive patterns)
        for goal_x, goal_y, reward in maze_env.goals:
            distance = math.sqrt((goal_x - self.x)**2 + (goal_y - self.y)**2)
            
            if distance < 50:  # Within sensing range
                # Goals emit characteristic wave patterns
                goal_wave_strength = 0
                for sensor in self.wave_sensors:
                    # Goals emit at specific frequencies
                    goal_freq = 2.5 + (reward / 100.0)  # Reward affects frequency
                    if abs(sensor['frequency'] - goal_freq) < 0.5:
                        goal_signal = reward / (1 + distance * 0.1)
                        goal_wave_strength += goal_signal * sensor['sensitivity']
                
                if goal_wave_strength > 0.5:  # Strong enough to detect
                    sensed_data['goals'].append({
                        'direction': (goal_x - self.x, goal_y - self.y),
                        'strength': goal_wave_strength,
                        'distance': distance,
                        'estimated_reward': reward
                    })
        
        return sensed_data
    
    def compute_wave_navigation_decision(self, sensed_data, maze_env):
        """Compute movement decision based purely on wave sensing"""
        
        if self.energy < 5:
            return False  # Too tired to move
        
        # Analyze sensed wave patterns
        direction_scores = {}
        
        # Score each direction based on wave reflections
        for (dx, dy), strength in sensed_data['directions'].items():
            score = 0
            
            # Low reflection = open path (good for movement)
            if strength < 0.3:
                score += 2.0  # Open path bonus
            elif strength > 1.0:
                score -= 1.0  # High reflection = wall/obstacle
            
            # Check if we've failed this direction recently
            failed_recently = False
            for failed_x, failed_y, failed_age in self.failed_directions:
                if abs(failed_x - (self.x + dx)) < 2 and abs(failed_y - (self.y + dy)) < 2:
                    if self.age - failed_age < 20:  # Recent failure
                        failed_recently = True
                        break
            
            if failed_recently:
                score -= 0.5
            
            direction_scores[(dx, dy)] = score
        
        # Bias toward detected goals
        for goal_info in sensed_data['goals']:
            goal_dx, goal_dy = goal_info['direction']
            goal_distance = goal_info['distance']
            
            # Normalize goal direction
            if goal_distance > 0:
                goal_dx_norm = goal_dx / goal_distance
                goal_dy_norm = goal_dy / goal_distance
                
                # Find best matching direction
                best_match_score = -1
                best_direction = None
                
                for (dx, dy) in direction_scores.keys():
                    match_score = goal_dx_norm * dx + goal_dy_norm * dy  # Dot product
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_direction = (dx, dy)
                
                if best_direction and best_match_score > 0:
                    # Boost score for direction toward goal
                    goal_bonus = goal_info['strength'] * best_match_score
                    direction_scores[best_direction] += goal_bonus
        
        # Add exploration randomness
        for direction in direction_scores:
            direction_scores[direction] += random.uniform(0, self.exploration_tendency)
        
        # Choose best direction
        if direction_scores:
            best_direction = max(direction_scores.items(), key=lambda x: x[1])
            (dx, dy), score = best_direction
            
            if score > 0.5:  # Threshold for movement
                # Attempt to move
                new_x = self.x + dx * 2  # Move 2 units in chosen direction
                new_y = self.y + dy * 2
                
                # Check if movement is valid
                if maze_env.is_valid_position(new_x, new_y):
                    self.x = new_x
                    self.y = new_y
                    self.energy -= 3  # Movement cost
                    
                    # Remember successful path
                    self.path_memory.append((self.x, self.y, score, self.age))
                    if len(self.path_memory) > 50:
                        self.path_memory = self.path_memory[-50:]
                    
                    # Learn from successful navigation
                    self.learn_navigation_pattern(sensed_data, (dx, dy), success=True)
                    
                    return True
                else:
                    # Movement failed - remember this
                    self.failed_directions.append((new_x, new_y, self.age))
                    if len(self.failed_directions) > 20:
                        self.failed_directions = self.failed_directions[-20:]
                    
                    self.learn_navigation_pattern(sensed_data, (dx, dy), success=False)
        
        return False
    
    def learn_navigation_pattern(self, sensed_data, chosen_direction, success):
        """Learn from navigation attempt outcomes"""
        
        # Record wave pattern associated with this decision
        wave_pattern = {
            'wave_strengths': dict(sensed_data['directions']),
            'chosen_direction': chosen_direction,
            'success': success,
            'age': self.age,
            'location': (self.x, self.y)
        }
        
        self.wave_navigation_discoveries.append(wave_pattern)
        
        # Keep memory manageable
        if len(self.wave_navigation_discoveries) > 30:
            self.wave_navigation_discoveries = self.wave_navigation_discoveries[-30:]
        
        # Develop navigation strategies from patterns
        if len(self.wave_navigation_discoveries) >= 5:
            self.develop_navigation_strategies()
    
    def develop_navigation_strategies(self):
        """Develop improved navigation strategies from experience"""
        
        successful_patterns = [p for p in self.wave_navigation_discoveries if p['success']]
        failed_patterns = [p for p in self.wave_navigation_discoveries if not p['success']]
        
        if len(successful_patterns) >= 3:
            # Identify successful wave pattern characteristics
            successful_strengths = []
            for pattern in successful_patterns:
                for direction, strength in pattern['wave_strengths'].items():
                    if direction == pattern['chosen_direction']:
                        successful_strengths.append(strength)
            
            if successful_strengths:
                avg_successful_strength = sum(successful_strengths) / len(successful_strengths)
                
                # Develop strategy
                new_strategy = {
                    'type': 'wave_strength_threshold',
                    'optimal_strength_range': (avg_successful_strength - 0.2, avg_successful_strength + 0.2),
                    'confidence': len(successful_patterns) / max(1, len(failed_patterns)),
                    'discovery_age': self.age
                }
                
                # Add if not already discovered similar strategy
                similar_exists = False
                for existing in self.navigation_strategies:
                    if (existing['type'] == 'wave_strength_threshold' and
                        abs(existing['optimal_strength_range'][0] - new_strategy['optimal_strength_range'][0]) < 0.1):
                        similar_exists = True
                        break
                
                if not similar_exists:
                    self.navigation_strategies.append(new_strategy)
    
    def check_goal_reached(self, maze_env):
        """Check if entity reached any goals"""
        for i, (goal_x, goal_y, reward) in enumerate(maze_env.goals):
            distance = math.sqrt((self.x - goal_x)**2 + (self.y - goal_y)**2)
            
            if distance < 5:  # Close enough to goal
                # Reached goal!
                self.energy += reward
                
                # Remember this goal location
                self.goal_memory.append({
                    'location': (goal_x, goal_y),
                    'reward': reward,
                    'discovery_age': self.age,
                    'path_taken': self.path_memory[-10:]  # Last 10 steps
                })
                
                # Remove goal from environment (consumed)
                maze_env.goals.pop(i)
                
                return reward
        
        return 0

class WaveMazeSimulation:
    """Large maze simulation with pure wave-based navigation"""
    
    def __init__(self, maze_size=200, num_entities=30, seed=None):
        if seed:
            random.seed(seed)
        
        self.maze_env = MazeEnvironment(maze_size, complexity=0.4)
        self.entities = []
        self.tick = 0
        
        # Create entities at random valid starting positions
        for i in range(num_entities):
            attempts = 0
            while attempts < 100:
                start_x = random.randint(10, maze_size - 10)
                start_y = random.randint(10, maze_size - 10)
                
                if self.maze_env.is_valid_position(start_x, start_y):
                    entity = WaveMazeEntity(i, start_x, start_y)
                    self.entities.append(entity)
                    break
                
                attempts += 1
        
        # Track discoveries
        self.navigation_breakthroughs = []
        self.goal_discoveries = []
        self.maze_mapping_progress = []
        
    def simulation_step(self):
        """Single step of wave-based maze navigation"""
        self.tick += 1
        
        for entity in self.entities[:]:
            entity.age += 1
            
            # Emit navigation waves
            entity.emit_navigation_waves(self.maze_env)
            
            # Sense wave environment
            sensed_data = entity.sense_maze_waves(self.maze_env, self.entities)
            
            # Make navigation decision based on waves
            moved = entity.compute_wave_navigation_decision(sensed_data, self.maze_env)
            
            # Check if reached any goals
            goal_reward = entity.check_goal_reached(self.maze_env)
            if goal_reward > 0:
                self.goal_discoveries.append({
                    'entity_id': entity.id,
                    'tick': self.tick,
                    'reward': goal_reward,
                    'strategies_used': len(entity.navigation_strategies)
                })
            
            # Energy decay
            entity.energy -= 1.5  # Base energy cost per tick
            
            # Remove entities that run out of energy
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Track navigation breakthroughs
        if self.tick % 100 == 0:
            self.detect_navigation_breakthroughs()
    
    def detect_navigation_breakthroughs(self):
        """Detect when entities make navigation breakthroughs"""
        
        # Strategy development breakthrough
        total_strategies = sum(len(e.navigation_strategies) for e in self.entities)
        if total_strategies > len(self.entities) * 0.5:  # Average > 0.5 strategies per entity
            self.navigation_breakthroughs.append({
                'type': 'strategy_development',
                'tick': self.tick,
                'total_strategies': total_strategies,
                'entities_with_strategies': len([e for e in self.entities if e.navigation_strategies])
            })
        
        # Maze exploration breakthrough
        explored_positions = set()
        for entity in self.entities:
            for path_x, path_y, _, _ in entity.path_memory:
                explored_positions.add((int(path_x//10), int(path_y//10)))  # 10x10 grid exploration
        
        exploration_coverage = len(explored_positions) / ((self.maze_env.size // 10) ** 2)
        if exploration_coverage > 0.1:  # Explored > 10% of maze
            self.maze_mapping_progress.append({
                'tick': self.tick,
                'coverage': exploration_coverage,
                'total_positions': len(explored_positions)
            })

def run_wave_maze_soup(maze_size=200, max_ticks=3000, seed=None):
    """Run large maze navigation with pure wave physics"""
    if not seed:
        seed = random.randint(1000, 9999)
    
    print("=== WAVE MAZE ALPHABET SOUP ===")
    print(f"Large complex maze navigation via pure wave physics")
    print(f"Maze: {maze_size}x{maze_size}, Max ticks: {max_ticks}, Seed: {seed}")
    print("Entities must discover wave-based navigation...")
    print()
    
    simulation = WaveMazeSimulation(maze_size, num_entities=25, seed=seed)
    
    print(f"Initial setup: {len(simulation.entities)} entities, {len(simulation.maze_env.goals)} goals")
    print(f"Wave reflectors: {len(simulation.maze_env.wave_reflectors)}")
    print(f"Wave channels: {len(simulation.maze_env.wave_channels)}")
    print()
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 500 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 ALL MAZE NAVIGATORS EXTINCT")
                break
            
            # Basic stats
            pop = len(simulation.entities)
            avg_energy = sum(e.energy for e in simulation.entities) / pop
            total_strategies = sum(len(e.navigation_strategies) for e in simulation.entities)
            goals_remaining = len(simulation.maze_env.goals)
            
            print(f"T{tick:4d}: Pop={pop:2d}, Energy={avg_energy:.1f}, Goals={goals_remaining}, Strategies={total_strategies}")
            
            # Navigation progress
            total_path_memory = sum(len(e.path_memory) for e in simulation.entities)
            total_discoveries = sum(len(e.wave_navigation_discoveries) for e in simulation.entities)
            recent_goals = len([g for g in simulation.goal_discoveries if g['tick'] > tick - 500])
            
            if total_path_memory > 0:
                print(f"        🗺️  {total_path_memory} paths remembered, {total_discoveries} wave discoveries")
            if recent_goals > 0:
                print(f"        🎯 {recent_goals} goals found recently")
            
            # Breakthroughs
            recent_breakthroughs = len([b for b in simulation.navigation_breakthroughs if b['tick'] > tick - 500])
            if recent_breakthroughs > 0:
                print(f"        ⚡ {recent_breakthroughs} navigation breakthroughs")
            
            # Coverage
            if simulation.maze_mapping_progress:
                latest_coverage = simulation.maze_mapping_progress[-1]['coverage']
                print(f"        📍 Maze exploration: {latest_coverage:.1%} coverage")
    
    # Final analysis
    print(f"\n=== WAVE MAZE NAVIGATION RESULTS ===")
    
    if simulation.entities:
        print(f"✅ Maze navigators survived: {len(simulation.entities)} entities")
        
        # Navigation achievements
        total_goals_found = len(simulation.goal_discoveries)
        total_strategies_developed = sum(len(e.navigation_strategies) for e in simulation.entities)
        
        print(f"\n🎯 Navigation achievements:")
        print(f"   Goals found: {total_goals_found}/{total_goals_found + len(simulation.maze_env.goals)}")
        print(f"   Navigation strategies developed: {total_strategies_developed}")
        print(f"   Navigation breakthroughs: {len(simulation.navigation_breakthroughs)}")
        
        # Show breakthrough types
        if simulation.navigation_breakthroughs:
            breakthrough_types = defaultdict(int)
            for breakthrough in simulation.navigation_breakthroughs:
                breakthrough_types[breakthrough['type']] += 1
            
            print(f"\n⚡ Navigation breakthroughs:")
            for breakthrough_type, count in breakthrough_types.items():
                print(f"   {breakthrough_type.replace('_', ' ').title()}: {count} occurrences")
        
        # Best navigators
        print(f"\n🏆 Best maze navigators:")
        top_navigators = sorted(simulation.entities, 
                              key=lambda e: len(e.navigation_strategies) + len(e.goal_memory), 
                              reverse=True)[:3]
        
        for entity in top_navigators:
            strategies = len(entity.navigation_strategies)
            goals_found = len(entity.goal_memory)
            paths = len(entity.path_memory)
            
            print(f"   Entity {entity.id}: {strategies} strategies, {goals_found} goals found, {paths} paths")
            print(f"     Energy: {entity.energy:.1f}, Age: {entity.age}")
            
            # Show their best strategy if any
            if entity.navigation_strategies:
                best_strategy = max(entity.navigation_strategies, key=lambda s: s.get('confidence', 0))
                print(f"     Best strategy: {best_strategy['type']} (confidence: {best_strategy.get('confidence', 0):.2f})")
        
        # Maze exploration summary
        if simulation.maze_mapping_progress:
            final_coverage = simulation.maze_mapping_progress[-1]['coverage']
            print(f"\n📍 Final maze exploration: {final_coverage:.1%} coverage")
    
    else:
        print("💀 Wave maze navigation experiment failed - no survivors")
    
    return simulation

if __name__ == "__main__":
    run_wave_maze_soup(maze_size=150, max_ticks=4000)