#!/usr/bin/env python3
"""
Debug Analysis: Why are agents hitting step limits?
Deep dive into what's actually happening during navigation
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
from wave_priority_experiments import ExperimentalNavigator, PriorityMethod
from wave_incremental_survival import generate_maze_with_food

class NavigationDebugger:
    """
    Detailed analysis of navigation behavior
    """
    
    def __init__(self):
        self.step_history = []
        self.position_frequency = {}
        self.loop_detection = {}
        self.attraction_history = []
    
    def record_step(self, step: int, position: Tuple[int, int], attractions: Dict[str, float], 
                   chosen_move: Tuple[int, int], hunger: float):
        """Record detailed step information"""
        self.step_history.append({
            'step': step,
            'position': position,
            'attractions': attractions.copy(),
            'chosen_move': chosen_move,
            'hunger': hunger
        })
        
        # Track position frequency
        if position not in self.position_frequency:
            self.position_frequency[position] = 0
        self.position_frequency[position] += 1
        
        # Loop detection
        if position in self.loop_detection:
            self.loop_detection[position]['visits'] += 1
            self.loop_detection[position]['last_visit'] = step
        else:
            self.loop_detection[position] = {'visits': 1, 'first_visit': step, 'last_visit': step}
    
    def analyze_behavior(self) -> Dict[str, Any]:
        """Analyze recorded behavior patterns"""
        if not self.step_history:
            return {}
        
        # Position analysis
        most_visited = max(self.position_frequency.items(), key=lambda x: x[1])
        total_unique_positions = len(self.position_frequency)
        
        # Loop analysis
        loops = [(pos, data) for pos, data in self.loop_detection.items() if data['visits'] > 10]
        loops.sort(key=lambda x: x[1]['visits'], reverse=True)
        
        # Movement analysis
        movements = []
        for i in range(1, len(self.step_history)):
            prev_pos = self.step_history[i-1]['position']
            curr_pos = self.step_history[i]['position']
            if prev_pos != curr_pos:
                movements.append((prev_pos, curr_pos))
        
        # Check for back-and-forth patterns
        back_forth_count = 0
        for i in range(2, len(movements)):
            if movements[i-2] == (movements[i][1], movements[i][0]):  # A->B then B->A pattern
                back_forth_count += 1
        
        # Attraction pattern analysis
        attraction_patterns = {
            'goal_dominant': 0,
            'food_dominant': 0,
            'balanced': 0
        }
        
        for step_data in self.step_history:
            goal_att = step_data['attractions'].get('goal', 0)
            food_att = step_data['attractions'].get('food', 0)
            
            if goal_att > food_att * 1.5:
                attraction_patterns['goal_dominant'] += 1
            elif food_att > goal_att * 1.5:
                attraction_patterns['food_dominant'] += 1
            else:
                attraction_patterns['balanced'] += 1
        
        return {
            'total_steps': len(self.step_history),
            'unique_positions': total_unique_positions,
            'most_visited_position': most_visited[0],
            'most_visited_count': most_visited[1],
            'major_loops': loops[:5],  # Top 5 loops
            'back_forth_count': back_forth_count,
            'movement_efficiency': len(movements) / len(self.step_history) if self.step_history else 0,
            'attraction_patterns': attraction_patterns,
            'final_hunger': self.step_history[-1]['hunger'] if self.step_history else 0
        }

class DeepDebugNavigator(ExperimentalNavigator):
    """
    Navigator with detailed debugging capabilities
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], 
                 method: PriorityMethod, debugger: NavigationDebugger):
        super().__init__(maze, start, goal, method)
        self.debugger = debugger
        self.attraction_history = []
    
    def select_best_move_debug(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Enhanced move selection with debug recording"""
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
        
        # Record debugging info
        if move_data:
            best_move = move_data[0][0]
            best_debug = move_data[0][2]
            
            self.debugger.record_step(
                step=self.step_count,
                position=current_pos,
                attractions=best_debug['base_attractions'],
                chosen_move=best_move,
                hunger=self.hunger_level
            )
        
        # Select move
        if len(move_data) > 1 and random.random() < 0.1:
            return move_data[1][0]
        else:
            return move_data[0][0]
    
    def solve_maze_debug(self) -> Dict[str, Any]:
        """Solve maze with detailed debugging"""
        current_pos = self.current_pos
        stuck_count = 0
        last_positions = []
        
        while current_pos != self.goal and self.step_count < self.max_steps:
            self.step_count += 1
            
            # Update state
            self.update_state(current_pos)
            
            # Select move with debug
            next_move = self.select_best_move_debug(current_pos)
            
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
            
            # Stuck detection
            last_positions.append(current_pos)
            if len(last_positions) > 20:
                last_positions.pop(0)
                
            if len(set(last_positions)) <= 3:  # Visiting only 3 positions in last 20 steps
                stuck_count += 1
                if stuck_count > 50:  # Stuck for 50+ steps
                    print(f"    STUCK DETECTED at step {self.step_count}: Position {current_pos}")
                    break
            else:
                stuck_count = 0
            
            # Progress reporting
            if self.step_count % 200 == 0:
                attraction, debug = self.calculate_priority_attraction(current_pos)
                print(f"    Step {self.step_count}: {current_pos}, Hunger: {self.hunger_level:.2f}, "
                      f"Goal: {debug['base_attractions']['goal']:.3f}, "
                      f"Food: {debug['base_attractions']['food']:.3f}")
        
        success = current_pos == self.goal
        
        return {
            'success': success,
            'steps_taken': len(self.path),
            'total_iterations': self.step_count,
            'food_eaten': self.food_eaten,
            'final_hunger': self.hunger_level,
            'method': self.method.value,
            'stuck_detected': stuck_count > 50,
            'debug_analysis': self.debugger.analyze_behavior()
        }

def deep_debug_analysis():
    """
    Deep debug analysis of why agents are failing
    """
    print("=== DEEP DEBUG ANALYSIS ===")
    print("Investigating why agents hit step limits")
    print()
    
    # Test the two best methods with detailed debugging
    methods_to_debug = [
        PriorityMethod.SUPPRESSION_WAVES,
        PriorityMethod.RESONANCE_PRIORITY
    ]
    
    for method in methods_to_debug:
        print(f"Deep debugging: {method.value.replace('_', ' ').title()}")
        print("-" * 50)
        
        # Generate test maze
        maze_size = 17
        maze, food_positions = generate_maze_with_food(maze_size, maze_size, num_food=2)
        start = (1, 1)
        goal = (maze_size - 2, maze_size - 2)
        
        # Create debugger and navigator
        debugger = NavigationDebugger()
        navigator = DeepDebugNavigator(maze, start, goal, method, debugger)
        navigator.add_food(food_positions)
        
        # Run with debugging
        result = navigator.solve_maze_debug()
        
        # Analysis
        analysis = result['debug_analysis']
        
        print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Total steps: {result['total_iterations']}")
        print(f"Food eaten: {result['food_eaten']}")
        print(f"Final hunger: {result['final_hunger']:.2f}")
        print(f"Stuck detected: {result['stuck_detected']}")
        print()
        
        print("BEHAVIOR ANALYSIS:")
        print(f"  Unique positions visited: {analysis['unique_positions']}")
        print(f"  Most visited position: {analysis['most_visited_position']} ({analysis['most_visited_count']} visits)")
        print(f"  Movement efficiency: {analysis['movement_efficiency']:.2f}")
        print(f"  Back-and-forth movements: {analysis['back_forth_count']}")
        print()
        
        print("ATTRACTION PATTERNS:")
        patterns = analysis['attraction_patterns']
        total = sum(patterns.values())
        for pattern, count in patterns.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {pattern.replace('_', ' ').title()}: {count} steps ({percentage:.1f}%)")
        print()
        
        if analysis['major_loops']:
            print("MAJOR LOOPS DETECTED:")
            for i, (pos, data) in enumerate(analysis['major_loops'], 1):
                duration = data['last_visit'] - data['first_visit']
                print(f"  {i}. Position {pos}: {data['visits']} visits over {duration} steps")
        print()
        
        # Specific problem diagnosis
        print("PROBLEM DIAGNOSIS:")
        
        if analysis['most_visited_count'] > 50:
            print(f"  ❌ SEVERE LOOPING: Stuck at {analysis['most_visited_position']} for {analysis['most_visited_count']} visits")
        
        if analysis['back_forth_count'] > 20:
            print(f"  ❌ OSCILLATION: {analysis['back_forth_count']} back-and-forth movements")
        
        if analysis['movement_efficiency'] < 0.5:
            print(f"  ❌ LOW MOBILITY: Only moving {analysis['movement_efficiency']:.1%} of steps")
        
        if patterns['balanced'] > total * 0.7:
            print(f"  ❌ INDECISION: Attractions too balanced {patterns['balanced']}/{total} steps")
        
        if analysis['unique_positions'] < maze_size * 0.3:
            print(f"  ❌ LIMITED EXPLORATION: Only visited {analysis['unique_positions']} positions")
        
        print("="*70)
        print()
    
    # Summary and recommendations
    print("DIAGNOSTIC SUMMARY:")
    print("Based on deep analysis, the main issues appear to be:")
    print("1. Severe position looping - agents get stuck in small areas")
    print("2. Indecisive attraction patterns - no clear behavioral direction")
    print("3. Limited exploration - not discovering new maze areas") 
    print("4. Back-and-forth oscillation - wasted movement")
    print()
    print("RECOMMENDED SOLUTIONS:")
    print("- Add stronger exploration pressure")
    print("- Implement position memory penalties")
    print("- Use clearer winner-take-all decision making")
    print("- Add momentum/direction persistence")

if __name__ == "__main__":
    deep_debug_analysis()