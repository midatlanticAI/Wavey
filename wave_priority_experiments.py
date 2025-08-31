#!/usr/bin/env python3
"""
Priority Hierarchy Experiments for Wave-Based Navigation
Testing multiple approaches to organize competing directives
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from wave_incremental_survival import WavePosition, WaveFood, generate_maze_with_food

class PriorityMethod(Enum):
    SIMPLE_ADDITION = "simple_addition"           # Current broken approach
    HIERARCHICAL_OVERRIDE = "hierarchical_override" # Hard priority levels
    FREQUENCY_BANDS = "frequency_bands"           # Different frequencies for different priorities
    SUPPRESSION_WAVES = "suppression_waves"      # Use destructive interference
    DYNAMIC_WEIGHTS = "dynamic_weights"          # State-dependent weighting
    RESONANCE_PRIORITY = "resonance_priority"    # Priority through resonance amplification

class ExperimentalNavigator:
    """
    Test different priority organization methods
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], method: PriorityMethod):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start = start
        self.goal = goal
        self.method = method
        
        # Navigation components
        self.goal_wave = WavePosition(goal[0], goal[1], self.width, self.height)
        self.current_pos = start
        self.food_objects = []
        
        # State tracking
        self.path = [start]
        self.visited = set([start])
        self.step_count = 0
        self.max_steps = self.width * self.height * 3
        self.food_eaten = 0
        
        # Experimental state
        self.hunger_level = 0.0  # 0 = satisfied, 1 = starving
        self.hunger_rate = 0.01  # Hunger per step
        self.debug_info = []
    
    def add_food(self, food_positions: List[Tuple[int, int]]):
        """Add food objects"""
        self.food_objects = []
        for pos in food_positions:
            self.food_objects.append(WaveFood(pos[0], pos[1], strength=1.0))
    
    def calculate_base_attractions(self, pos: Tuple[int, int]) -> Dict[str, float]:
        """Calculate raw attraction values for each directive"""
        # Goal attraction (from working system)
        pos_wave = WavePosition(pos[0], pos[1], self.width, self.height)
        
        freq_diff_x = abs(self.goal_wave.frequency_x - pos_wave.frequency_x)
        freq_diff_y = abs(self.goal_wave.frequency_y - pos_wave.frequency_y)
        phase_diff_x = abs(self.goal_wave.phase_x - pos_wave.phase_x)
        phase_diff_y = abs(self.goal_wave.phase_y - pos_wave.phase_y)
        amplitude_product = self.goal_wave.amplitude * pos_wave.amplitude
        
        frequency_factor = 1.0 / (1.0 + 0.05 * (freq_diff_x + freq_diff_y))
        phase_factor = 1.0 / (1.0 + 0.5 * (phase_diff_x + phase_diff_y))
        goal_attraction = frequency_factor * phase_factor * amplitude_product
        
        # Food attraction
        food_attraction = sum(food.calculate_attraction_to_position(pos[0], pos[1]) for food in self.food_objects)
        
        return {
            'goal': goal_attraction,
            'food': food_attraction,
            'hunger': self.hunger_level
        }
    
    # EXPERIMENT 1: Simple Addition (Current broken method)
    def _simple_addition_priority(self, attractions: Dict[str, float]) -> float:
        """Simple weighted addition - known to create loops"""
        return attractions['goal'] * 1.0 + attractions['food'] * 2.0
    
    # EXPERIMENT 2: Hierarchical Override
    def _hierarchical_override_priority(self, attractions: Dict[str, float]) -> float:
        """Hard priority levels with complete override"""
        hunger = attractions['hunger']
        
        # Level 1: Critical hunger - ONLY food matters
        if hunger > 0.8:
            return attractions['food'] * 10.0  # Amplify food, ignore goal
        
        # Level 2: Moderate hunger - Food preferred but not exclusive  
        elif hunger > 0.4:
            return attractions['food'] * 3.0 + attractions['goal'] * 0.5
        
        # Level 3: Low hunger - Goal focused with food bonus
        else:
            return attractions['goal'] * 2.0 + attractions['food'] * 0.8
    
    # EXPERIMENT 3: Frequency Band Separation
    def _frequency_bands_priority(self, attractions: Dict[str, float]) -> float:
        """Different frequency bands for different priorities"""
        # Use hunger level to determine which frequency band dominates
        hunger = attractions['hunger']
        
        # High frequency band (50-100 Hz) = Immediate needs (food)
        high_freq_component = attractions['food'] * math.sin(2 * math.pi * 75 * self.step_count * 0.01)
        
        # Low frequency band (5-15 Hz) = Long-term goals (exit)
        low_freq_component = attractions['goal'] * math.sin(2 * math.pi * 10 * self.step_count * 0.01)
        
        # Hunger modulates frequency band dominance
        high_weight = hunger
        low_weight = 1.0 - hunger
        
        return high_freq_component * high_weight + low_freq_component * low_weight
    
    # EXPERIMENT 4: Suppression Waves
    def _suppression_waves_priority(self, attractions: Dict[str, float]) -> float:
        """Use destructive interference to suppress lower priorities"""
        hunger = attractions['hunger']
        
        # Create base wave for each directive
        goal_wave = attractions['goal'] * math.sin(2 * math.pi * 10 * self.step_count * 0.01)
        food_wave = attractions['food'] * math.sin(2 * math.pi * 25 * self.step_count * 0.01)
        
        # Create suppression wave for goal when hungry
        if hunger > 0.5:
            # Suppression wave with opposite phase to goal wave
            goal_suppression = hunger * math.sin(2 * math.pi * 10 * self.step_count * 0.01 + math.pi)
            suppressed_goal = goal_wave + goal_suppression
        else:
            suppressed_goal = goal_wave
        
        # Create suppression wave for food when not hungry
        if hunger < 0.3:
            food_suppression = (1.0 - hunger) * math.sin(2 * math.pi * 25 * self.step_count * 0.01 + math.pi)
            suppressed_food = food_wave + food_suppression
        else:
            suppressed_food = food_wave
        
        return suppressed_goal + suppressed_food
    
    # EXPERIMENT 5: Dynamic State-Dependent Weighting
    def _dynamic_weights_priority(self, attractions: Dict[str, float]) -> float:
        """Smoothly varying weights based on multiple state factors"""
        hunger = attractions['hunger']
        
        # Distance to goal affects priority
        goal_distance = abs(self.current_pos[0] - self.goal[0]) + abs(self.current_pos[1] - self.goal[1])
        normalized_goal_distance = goal_distance / (self.width + self.height)
        
        # Number of available food affects priority
        food_scarcity = 1.0 / (len(self.food_objects) + 1)
        
        # Dynamic weight calculation
        food_weight = (hunger ** 2) * (1.0 + food_scarcity)  # Exponential hunger response
        goal_weight = (1.0 - hunger) * (1.0 + normalized_goal_distance)  # Higher when far from goal
        
        # Normalize to prevent extreme values
        total_weight = food_weight + goal_weight
        if total_weight > 0:
            food_weight /= total_weight
            goal_weight /= total_weight
        
        return attractions['food'] * food_weight + attractions['goal'] * goal_weight
    
    # EXPERIMENT 6: Resonance-Based Priority
    def _resonance_priority(self, attractions: Dict[str, float]) -> float:
        """Use resonance to amplify dominant priority"""
        hunger = attractions['hunger']
        
        # Base frequencies for each directive
        goal_freq = 12  # Hz
        food_freq = 30  # Hz
        
        # Calculate resonance based on hunger state
        # When hungry, create resonance with food frequency
        # When not hungry, create resonance with goal frequency
        
        resonance_freq = hunger * food_freq + (1.0 - hunger) * goal_freq
        
        # Calculate resonance amplification
        goal_resonance = 1.0 + math.cos(2 * math.pi * (goal_freq - resonance_freq) * 0.1)
        food_resonance = 1.0 + math.cos(2 * math.pi * (food_freq - resonance_freq) * 0.1)
        
        # Apply resonance amplification
        amplified_goal = attractions['goal'] * goal_resonance
        amplified_food = attractions['food'] * food_resonance
        
        return amplified_goal + amplified_food
    
    def calculate_priority_attraction(self, pos: Tuple[int, int]) -> Tuple[float, Dict[str, Any]]:
        """Calculate attraction using selected priority method"""
        base_attractions = self.calculate_base_attractions(pos)
        
        # Select method
        if self.method == PriorityMethod.SIMPLE_ADDITION:
            final_attraction = self._simple_addition_priority(base_attractions)
        elif self.method == PriorityMethod.HIERARCHICAL_OVERRIDE:
            final_attraction = self._hierarchical_override_priority(base_attractions)
        elif self.method == PriorityMethod.FREQUENCY_BANDS:
            final_attraction = self._frequency_bands_priority(base_attractions)
        elif self.method == PriorityMethod.SUPPRESSION_WAVES:
            final_attraction = self._suppression_waves_priority(base_attractions)
        elif self.method == PriorityMethod.DYNAMIC_WEIGHTS:
            final_attraction = self._dynamic_weights_priority(base_attractions)
        elif self.method == PriorityMethod.RESONANCE_PRIORITY:
            final_attraction = self._resonance_priority(base_attractions)
        else:
            final_attraction = base_attractions['goal']
        
        # Debug info
        debug = {
            'base_attractions': base_attractions,
            'final_attraction': final_attraction,
            'method': self.method.value,
            'hunger': self.hunger_level
        }
        
        return final_attraction, debug
    
    def get_valid_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid moves"""
        x, y = pos
        moves = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if self.maze[new_y][new_x] == 0:
                    moves.append((new_x, new_y))
        
        return moves
    
    def select_best_move(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Select move using priority method"""
        valid_moves = self.get_valid_moves(current_pos)
        
        if not valid_moves:
            return None
        
        move_data = []
        for move in valid_moves:
            attraction, debug = self.calculate_priority_attraction(move)
            
            # Penalty for revisiting
            if move in self.visited:
                attraction *= 0.3
            
            move_data.append((move, attraction, debug))
        
        # Sort by attraction
        move_data.sort(key=lambda x: x[1], reverse=True)
        
        # Store debug info for best move
        if move_data:
            best_debug = move_data[0][2]
            self.debug_info.append({
                'step': self.step_count,
                'position': current_pos,
                'method': self.method.value,
                'debug': best_debug
            })
        
        # Select best move with small exploration
        if len(move_data) > 1 and random.random() < 0.1:
            return move_data[1][0]
        else:
            return move_data[0][0]
    
    def update_state(self, pos: Tuple[int, int]):
        """Update agent state"""
        # Increase hunger
        self.hunger_level = min(1.0, self.hunger_level + self.hunger_rate)
        
        # Check food consumption
        for food in self.food_objects[:]:
            if food.x == pos[0] and food.y == pos[1]:
                self.food_objects.remove(food)
                self.food_eaten += 1
                self.hunger_level = max(0.0, self.hunger_level - 0.3)  # Food reduces hunger
                break
    
    def solve_maze_experimental(self) -> Dict[str, Any]:
        """Solve maze using experimental priority method"""
        current_pos = self.current_pos
        
        while current_pos != self.goal and self.step_count < self.max_steps:
            self.step_count += 1
            
            # Update state
            self.update_state(current_pos)
            
            # Select move
            next_move = self.select_best_move(current_pos)
            
            if next_move is None:
                # Backtrack
                if len(self.path) > 1:
                    self.path.pop()
                    current_pos = self.path[-1]
                else:
                    break
            else:
                current_pos = next_move
                self.path.append(current_pos)
                self.visited.add(current_pos)
        
        success = current_pos == self.goal
        
        return {
            'success': success,
            'steps_taken': len(self.path),
            'total_iterations': self.step_count,
            'food_eaten': self.food_eaten,
            'final_hunger': self.hunger_level,
            'method': self.method.value,
            'debug_info': self.debug_info[-10:] if self.debug_info else []  # Last 10 entries
        }

def test_priority_methods():
    """
    Test all priority organization methods
    """
    print("=== WAVE PRIORITY ORGANIZATION EXPERIMENTS ===")
    print("Testing different approaches to manage competing directives")
    print()
    
    # Test parameters
    maze_size = 17
    num_trials = 3
    
    methods_to_test = [
        PriorityMethod.SIMPLE_ADDITION,
        PriorityMethod.HIERARCHICAL_OVERRIDE, 
        PriorityMethod.FREQUENCY_BANDS,
        PriorityMethod.SUPPRESSION_WAVES,
        PriorityMethod.DYNAMIC_WEIGHTS,
        PriorityMethod.RESONANCE_PRIORITY
    ]
    
    results = {}
    
    for method in methods_to_test:
        print(f"Testing {method.value.replace('_', ' ').title()}:")
        method_results = []
        
        for trial in range(num_trials):
            # Generate fresh maze and food
            maze, food_positions = generate_maze_with_food(maze_size, maze_size, num_food=2)
            start = (1, 1)
            goal = (maze_size - 2, maze_size - 2)
            
            # Create navigator with experimental method
            navigator = ExperimentalNavigator(maze, start, goal, method)
            navigator.add_food(food_positions)
            
            # Solve maze
            result = navigator.solve_maze_experimental()
            method_results.append(result)
            
            # Print trial result
            status = "✅" if result['success'] else "❌"
            print(f"  Trial {trial + 1}: {status} {result['steps_taken']} steps, "
                  f"{result['food_eaten']} food, hunger: {result['final_hunger']:.2f}")
        
        # Method summary
        successes = sum(1 for r in method_results if r['success'])
        avg_steps = sum(r['steps_taken'] for r in method_results) / len(method_results)
        avg_food = sum(r['food_eaten'] for r in method_results) / len(method_results)
        success_rate = (successes / len(method_results)) * 100
        
        results[method] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_food': avg_food,
            'results': method_results
        }
        
        print(f"  📊 {success_rate:.0f}% success, {avg_steps:.0f} avg steps, {avg_food:.1f} avg food")
        print()
    
    # Overall analysis
    print("="*70)
    print("PRIORITY ORGANIZATION EXPERIMENT RESULTS:")
    print("="*70)
    
    # Rank methods by success rate
    ranked_methods = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    
    print("Method Rankings:")
    for i, (method, data) in enumerate(ranked_methods, 1):
        method_name = method.value.replace('_', ' ').title()
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{status} {i}. {method_name:20s}: {data['success_rate']:5.0f}% success, "
              f"{data['avg_steps']:5.0f} steps, {data['avg_food']:.1f} food")
    
    # Find best method
    best_method, best_data = ranked_methods[0]
    
    print(f"\n🎯 BEST METHOD: {best_method.value.replace('_', ' ').title()}")
    print(f"Success rate: {best_data['success_rate']:.0f}%")
    
    if best_data['success_rate'] >= 70:
        print("✅ Found effective priority organization method!")
        print("Ready to proceed with this approach")
        return True, best_method
    else:
        print("❌ No method achieved satisfactory performance")
        print("Need alternative approaches")
        return False, None

if __name__ == "__main__":
    success, best_method = test_priority_methods()
    
    if success:
        print(f"\n🔥 PRIORITY ORGANIZATION SOLVED!")
        print(f"Wave-based hierarchy using: {best_method.value}")
    else:
        print(f"\n⚠️  Need more experimental approaches")