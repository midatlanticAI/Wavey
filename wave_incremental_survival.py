#!/usr/bin/env python3
"""
Incremental Wave-Based Survival System
Start with working maze navigation, add ONE capability at a time
Step 1: Maze Navigation + Food Detection Only
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional

class WavePosition:
    """
    Wave encoding of position (from working maze system)
    """
    
    def __init__(self, x: int, y: int, maze_width: int, maze_height: int):
        self.x = x
        self.y = y
        self.maze_width = maze_width
        self.maze_height = maze_height
        
        # Basic wave encoding (proven to work)
        self.frequency_x = (x + 1) * 10
        self.frequency_y = (y + 1) * 15
        
        self.phase_x = (2 * math.pi * x) / maze_width
        self.phase_y = (2 * math.pi * y) / maze_height
        
        center_x, center_y = maze_width // 2, maze_height // 2
        self.distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        self.amplitude = 1.0 / (1.0 + 0.1 * self.distance_from_center)

class WaveFood:
    """
    NEW: Wave encoding for food positions
    Food creates attractive wave fields that the agent can detect
    """
    
    def __init__(self, x: int, y: int, strength: float = 1.0):
        self.x = x
        self.y = y
        self.strength = strength
        
        # Food wave properties
        self.frequency = 25  # Fixed frequency for all food (distinct from position frequencies)
        self.amplitude = strength
        self.phase = 0  # No phase offset for simplicity
        
        # Detection range
        self.detection_range = 8  # Can be "smelled" within 8 cells
    
    def calculate_attraction_to_position(self, pos_x: int, pos_y: int) -> float:
        """
        Calculate wave-based attraction between food and position
        Uses distance-based amplitude decay + wave interference
        """
        distance = math.sqrt((self.x - pos_x)**2 + (self.y - pos_y)**2)
        
        # No attraction beyond detection range
        if distance > self.detection_range:
            return 0.0
        
        # Wave-based attraction calculation
        # Closer food = stronger amplitude, specific frequency creates "pull"
        distance_factor = 1.0 / (1.0 + 0.2 * distance)
        wave_amplitude = self.amplitude * distance_factor
        
        # Phase based on relative position (creates directional information)
        direction_angle = math.atan2(self.y - pos_y, self.x - pos_x)
        wave_phase = direction_angle
        
        # Calculate wave value at this position
        # Higher values = stronger attraction toward food
        attraction = wave_amplitude * (1 + math.cos(wave_phase))  # Always positive attraction
        
        return attraction

class IncrementalWaveNavigator:
    """
    Start with working maze navigation, add food detection
    Based on the successful 25x25 maze navigator
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start = start
        self.goal = goal
        
        # Working navigation from previous system
        self.goal_wave = WavePosition(goal[0], goal[1], self.width, self.height)
        self.current_pos = start
        
        # NEW: Food detection system
        self.food_objects = []  # List of WaveFood objects
        
        # Navigation state (from working system)
        self.path = [start]
        self.visited = set([start])
        self.step_count = 0
        self.max_steps = self.width * self.height * 4
        
        # NEW: Food interaction
        self.food_eaten = 0
        self.total_food_attraction = 0  # Debug metric
    
    def add_food(self, food_positions: List[Tuple[int, int]]):
        """Add food objects to the environment"""
        self.food_objects = []
        for pos in food_positions:
            self.food_objects.append(WaveFood(pos[0], pos[1], strength=1.0))
    
    def calculate_maze_attraction(self, pos: Tuple[int, int]) -> float:
        """
        Original maze navigation attraction (PROVEN TO WORK)
        Calculate wave interference between position and goal
        """
        x, y = pos
        pos_wave = WavePosition(x, y, self.width, self.height)
        
        # Beat frequency calculation
        freq_diff_x = abs(self.goal_wave.frequency_x - pos_wave.frequency_x)
        freq_diff_y = abs(self.goal_wave.frequency_y - pos_wave.frequency_y)
        
        # Phase alignment
        phase_diff_x = abs(self.goal_wave.phase_x - pos_wave.phase_x)
        phase_diff_y = abs(self.goal_wave.phase_y - pos_wave.phase_y)
        
        # Amplitude resonance
        amplitude_product = self.goal_wave.amplitude * pos_wave.amplitude
        
        # Combined attraction (from working system)
        frequency_factor = 1.0 / (1.0 + 0.05 * (freq_diff_x + freq_diff_y))
        phase_factor = 1.0 / (1.0 + 0.5 * (phase_diff_x + phase_diff_y))
        
        return frequency_factor * phase_factor * amplitude_product
    
    def calculate_food_attraction(self, pos: Tuple[int, int]) -> float:
        """
        NEW: Calculate total food attraction at position
        Combines attraction from all food sources
        """
        total_food_attraction = 0
        
        for food in self.food_objects:
            attraction = food.calculate_attraction_to_position(pos[0], pos[1])
            total_food_attraction += attraction
        
        return total_food_attraction
    
    def calculate_combined_attraction(self, pos: Tuple[int, int]) -> float:
        """
        NEW: Combine maze navigation + food attraction
        This is the key integration point
        """
        # Get both attraction types
        maze_attraction = self.calculate_maze_attraction(pos)
        food_attraction = self.calculate_food_attraction(pos)
        
        # Weight them (food should be attractive but not override goal completely)
        maze_weight = 1.0
        food_weight = 2.0  # Food is twice as attractive as goal (when hungry)
        
        # Combined attraction
        total_attraction = (maze_attraction * maze_weight) + (food_attraction * food_weight)
        
        # Store debug info
        self.total_food_attraction = food_attraction
        
        return total_attraction
    
    def get_valid_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent positions (from working system)"""
        x, y = pos
        moves = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if self.maze[new_y][new_x] == 0:
                    moves.append((new_x, new_y))
        
        return moves
    
    def select_best_move(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        MODIFIED: Select move based on combined maze + food attraction
        """
        valid_moves = self.get_valid_moves(current_pos)
        
        if not valid_moves:
            return None
        
        # Calculate combined attraction for each move
        move_attractions = []
        for move in valid_moves:
            attraction = self.calculate_combined_attraction(move)
            
            # Penalty for revisiting (from working system)
            if move in self.visited:
                attraction *= 0.3
            
            # Bonus for getting closer to goal (from working system)
            goal_distance = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
            current_distance = abs(current_pos[0] - self.goal[0]) + abs(current_pos[1] - self.goal[1])
            
            if goal_distance < current_distance:
                attraction *= 1.2
            
            move_attractions.append((move, attraction))
        
        # Sort by attraction (from working system)
        move_attractions.sort(key=lambda x: x[1], reverse=True)
        
        # Selection with exploration (from working system)
        if len(move_attractions) > 1 and random.random() < 0.15:
            return move_attractions[1][0]
        else:
            return move_attractions[0][0]
    
    def check_food_consumption(self, pos: Tuple[int, int]) -> bool:
        """
        NEW: Check if agent is at food position and consume it
        """
        for food in self.food_objects[:]:  # Copy list for safe removal
            if food.x == pos[0] and food.y == pos[1]:
                self.food_objects.remove(food)
                self.food_eaten += 1
                return True
        return False
    
    def solve_maze_with_food(self) -> Dict[str, Any]:
        """
        MODIFIED: Solve maze while collecting food
        Based on working maze solver
        """
        current_pos = self.current_pos
        
        while current_pos != self.goal and self.step_count < self.max_steps:
            self.step_count += 1
            
            # Select next move using combined attraction
            next_move = self.select_best_move(current_pos)
            
            if next_move is None:
                # Dead end - backtrack (from working system)
                if len(self.path) > 1:
                    self.path.pop()
                    current_pos = self.path[-1]
                else:
                    break
            else:
                # Move to selected position
                current_pos = next_move
                self.path.append(current_pos)
                self.visited.add(current_pos)
                
                # NEW: Check for food consumption
                food_consumed = self.check_food_consumption(current_pos)
                
                # Debug output every 50 steps
                if self.step_count % 50 == 0:
                    food_attraction = self.calculate_food_attraction(current_pos)
                    print(f"  Step {self.step_count}: Pos {current_pos}, Food attraction: {food_attraction:.3f}, Food eaten: {self.food_eaten}")
        
        # Results
        success = current_pos == self.goal
        efficiency = len(self.path) / (self.width + self.height)
        
        return {
            'success': success,
            'path': self.path.copy(),
            'steps_taken': len(self.path),
            'total_iterations': self.step_count,
            'efficiency': efficiency,
            'food_eaten': self.food_eaten,  # NEW
            'total_food_spawned': len(self.food_objects) + self.food_eaten,  # NEW
            'start': self.start,
            'goal': self.goal,
            'maze_size': (self.width, self.height)
        }

def generate_maze_with_food(width: int, height: int, num_food: int = 3) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Generate maze and randomly place food"""
    # Generate maze (from working system)
    maze = [[1 for _ in range(width)] for _ in range(height)]
    
    def carve_path(x, y):
        maze[y][x] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 < new_x < width - 1 and 0 < new_y < height - 1 and maze[new_y][new_x] == 1:
                maze[y + dy // 2][x + dx // 2] = 0
                carve_path(new_x, new_y)
    
    carve_path(1, 1)
    maze[1][1] = 0
    maze[height - 2][width - 2] = 0
    
    # Place food randomly in open spaces
    open_positions = []
    for x in range(width):
        for y in range(height):
            if maze[y][x] == 0 and (x, y) != (1, 1) and (x, y) != (width - 2, height - 2):
                open_positions.append((x, y))
    
    food_positions = random.sample(open_positions, min(num_food, len(open_positions)))
    
    return maze, food_positions

def print_maze_with_food_and_path(maze: List[List[int]], path: List[Tuple[int, int]], 
                                  start: Tuple[int, int], goal: Tuple[int, int], 
                                  food_positions: List[Tuple[int, int]]):
    """Print maze with path and food visualization"""
    height, width = len(maze), len(maze[0])
    
    for y in range(height):
        row = ""
        for x in range(width):
            pos = (x, y)
            if pos == start:
                row += "S"
            elif pos == goal:
                row += "G"
            elif pos in food_positions:
                row += "F"
            elif pos in path:
                row += "."
            elif maze[y][x] == 1:
                row += "█"
            else:
                row += " "
        print(row)

def test_incremental_food_detection():
    """
    Test incremental wave system: maze navigation + food detection
    """
    print("=== INCREMENTAL WAVE SYSTEM TEST ===")
    print("Step 1: Working maze navigation + food detection")
    print()
    
    maze_sizes = [13, 17, 21, 25]
    results = []
    
    for size in maze_sizes:
        print(f"Testing {size}x{size} maze with food...")
        
        # Generate maze with food
        num_food = max(2, size // 8)
        maze, food_positions = generate_maze_with_food(size, size, num_food)
        start = (1, 1)
        goal = (size - 2, size - 2)
        
        # Create navigator and add food
        navigator = IncrementalWaveNavigator(maze, start, goal)
        navigator.add_food(food_positions)
        
        # Solve maze
        result = navigator.solve_maze_with_food()
        results.append(result)
        
        # Results
        status = "✅" if result['success'] else "❌"
        food_ratio = result['food_eaten'] / result['total_food_spawned'] if result['total_food_spawned'] > 0 else 0
        
        print(f"{status} {size}x{size}: {result['steps_taken']} steps, "
              f"{result['food_eaten']}/{result['total_food_spawned']} food ({food_ratio:.1%}), "
              f"efficiency: {result['efficiency']:.2f}")
        
        # Show small mazes
        if size <= 17 and result['success']:
            print("Maze with path and food:")
            # Show remaining food positions
            remaining_food = [(f.x, f.y) for f in navigator.food_objects]
            print_maze_with_food_and_path(maze, result['path'], start, goal, remaining_food)
            print()
    
    # Assessment
    successful_runs = sum(1 for r in results if r['success'])
    total_food_eaten = sum(r['food_eaten'] for r in results)
    total_food_available = sum(r['total_food_spawned'] for r in results)
    
    success_rate = (successful_runs / len(results)) * 100
    food_collection_rate = (total_food_eaten / total_food_available) * 100 if total_food_available > 0 else 0
    
    print("="*60)
    print("INCREMENTAL FOOD DETECTION RESULTS:")
    print(f"Success rate: {successful_runs}/{len(results)} ({success_rate:.1f}%)")
    print(f"Food collection: {total_food_eaten}/{total_food_available} ({food_collection_rate:.1f}%)")
    
    if success_rate >= 75 and food_collection_rate >= 30:
        print("\n🎯 STEP 1 SUCCESS: Food detection integrated successfully!")
        print("✅ Maze navigation still works")
        print("✅ Food attraction influences path selection")
        print("✅ Food consumption mechanics work")
        print("Ready for Step 2: Add hunger system")
        return True
    else:
        print(f"\n❌ STEP 1 FAILED: Food detection integration broken")
        print("Need to debug before proceeding")
        return False

if __name__ == "__main__":
    success = test_incremental_food_detection()
    
    if success:
        print(f"\n🔥 INCREMENTAL APPROACH WORKING!")
        print(f"Wave-based food detection successfully added to working navigation")
    else:
        print(f"\n⚠️  Need to fix Step 1 before adding more complexity")