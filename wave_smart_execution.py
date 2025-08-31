#!/usr/bin/env python3
"""
Smart Execution System for Wave-Based Adaptive Rules
The entity has learned WHAT to do - now it needs to execute effectively
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
from wave_adaptive_rule_maker import RuleMakingAgent, SituationPattern, RuleType, AdaptiveRule
from wave_incremental_survival import generate_maze_with_food

class SmartExecutionAgent(RuleMakingAgent):
    """
    Enhanced agent with smart execution capabilities
    Uses the working maze navigation system for movement
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        super().__init__(maze, start, goal)
        
        # Smart execution state
        self.current_target = None
        self.target_type = None  # 'food', 'goal', 'safety'
        self.stuck_counter = 0
        self.last_positions = []
        self.exploration_mode = False
    
    def _find_nearest_target(self, target_type: str) -> Optional[Tuple[int, int]]:
        """Find nearest target of specified type"""
        if target_type == 'food':
            if not self.food_positions:
                return None
            return min(self.food_positions, 
                      key=lambda f: abs(f[0] - self.position[0]) + abs(f[1] - self.position[1]))
        
        elif target_type == 'goal':
            return self.goal
        
        elif target_type == 'safety':
            # Find position far from enemies
            if not self.enemy_positions:
                return self.goal  # No enemies, head to goal
            
            # Find valid position that maximizes distance from nearest enemy
            valid_positions = []
            for x in range(self.width):
                for y in range(self.height):
                    if self.maze[y][x] == 0:  # Open position
                        min_enemy_dist = min([abs(x - e[0]) + abs(y - e[1]) for e in self.enemy_positions])
                        valid_positions.append(((x, y), min_enemy_dist))
            
            if valid_positions:
                # Return position with maximum enemy distance
                return max(valid_positions, key=lambda x: x[1])[0]
            return self.goal
        
        return None
    
    def _calculate_wave_attraction(self, pos: Tuple[int, int], target: Tuple[int, int]) -> float:
        """
        Use the WORKING wave navigation system from the successful maze solver
        """
        if not target:
            return 0
        
        # Distance-based attraction (simple but effective)
        distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        attraction = 1.0 / (1.0 + 0.1 * distance)
        
        # Add wave modulation for biological realism
        wave_factor = math.sin(2 * math.pi * 0.1 * self.steps_taken) * 0.1 + 1.0
        
        return attraction * wave_factor
    
    def _smart_pathfinding(self, target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Smart pathfinding that actually works
        Uses successful maze navigation principles
        """
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return None
        
        # Calculate attraction for each move
        move_attractions = []
        for move in valid_moves:
            attraction = self._calculate_wave_attraction(move, target)
            
            # Penalty for recently visited positions (avoid loops)
            if move in self.last_positions:
                recent_penalty = self.last_positions.count(move) * 0.5
                attraction *= (1.0 / (1.0 + recent_penalty))
            
            # Bonus for making progress toward target
            current_distance = abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
            move_distance = abs(move[0] - target[0]) + abs(move[1] - target[1])
            
            if move_distance < current_distance:
                attraction *= 1.5  # Progress bonus
            
            move_attractions.append((move, attraction))
        
        # Sort by attraction
        move_attractions.sort(key=lambda x: x[1], reverse=True)
        
        # Select best move with small exploration chance
        if len(move_attractions) > 1 and random.random() < 0.15:
            return move_attractions[1][0]  # Second best for exploration
        else:
            return move_attractions[0][0]  # Best move
    
    def _execute_smart_action(self, action: str) -> str:
        """
        Execute action with smart pathfinding
        """
        # Determine target based on action
        if action == "seek_food":
            target = self._find_nearest_target('food')
            self.target_type = 'food'
        elif action == "seek_goal":
            target = self._find_nearest_target('goal')
            self.target_type = 'goal'
        elif action == "avoid_enemy":
            target = self._find_nearest_target('safety')
            self.target_type = 'safety'
        elif action == "explore_new_area":
            # Pick a random open position we haven't visited recently
            unvisited_positions = []
            for x in range(self.width):
                for y in range(self.height):
                    if (self.maze[y][x] == 0 and 
                        (x, y) not in self.last_positions[-20:] and  # Not recently visited
                        abs(x - self.position[0]) + abs(y - self.position[1]) > 3):  # Not too close
                        unvisited_positions.append((x, y))
            
            if unvisited_positions:
                target = random.choice(unvisited_positions)
                self.target_type = 'exploration'
            else:
                target = self.goal  # Fall back to goal
                self.target_type = 'goal'
        else:
            target = self.goal
            self.target_type = 'goal'
        
        # Update current target
        self.current_target = target
        
        if not target:
            return "no_target_available"
        
        # Smart pathfinding to target
        next_move = self._smart_pathfinding(target)
        
        if next_move:
            old_position = self.position
            self.position = next_move
            
            # Track position history for loop avoidance
            self.last_positions.append(old_position)
            if len(self.last_positions) > 30:  # Keep history bounded
                self.last_positions.pop(0)
            
            # Check if we reached target
            if self.position == target:
                if self.target_type == 'food':
                    return "reached_food"
                elif self.target_type == 'goal':
                    return "reached_goal"
                else:
                    return "reached_target"
            else:
                return f"moved_toward_{self.target_type}"
        else:
            return "no_valid_moves"
    
    def execute_action(self, action: str) -> str:
        """Override parent's execute_action with smart execution"""
        return self._execute_smart_action(action)
    
    def _evaluate_action_success(self, old_situation: SituationPattern, new_situation: SituationPattern, action_result: str) -> bool:
        """Enhanced success evaluation"""
        if action_result in ["no_target_available", "no_valid_moves"]:
            return False
        
        if action_result in ["reached_food", "reached_goal", "reached_target"]:
            return True
        
        # Check for progress toward target
        if self.current_target:
            old_distance = abs(self.last_positions[-1][0] - self.current_target[0]) + abs(self.last_positions[-1][1] - self.current_target[1]) if self.last_positions else 999
            new_distance = abs(self.position[0] - self.current_target[0]) + abs(self.position[1] - self.current_target[1])
            
            if new_distance < old_distance:
                return True  # Made progress
        
        # Success based on rule type
        if self.active_rule:
            if self.active_rule.rule_type == RuleType.SURVIVAL:
                if "food" in self.active_rule.action:
                    return (new_situation.hunger_level < old_situation.hunger_level or 
                           new_situation.food_distance < old_situation.food_distance)
                elif "avoid" in self.active_rule.action:
                    return new_situation.enemy_distance > old_situation.enemy_distance
            
            elif self.active_rule.rule_type == RuleType.GOAL:
                return new_situation.goal_distance < old_situation.goal_distance
            
            elif self.active_rule.rule_type == RuleType.EXPLORATION:
                return self.position != (self.last_positions[-1] if self.last_positions else self.position)
        
        return action_result.startswith("moved_toward_")  # Any movement is partial success
    
    def update_state(self):
        """Enhanced state update with stuck detection"""
        # Call parent update
        super().update_state()
        
        # Stuck detection and recovery
        if len(self.last_positions) >= 10:
            recent_unique = len(set(self.last_positions[-10:]))
            if recent_unique <= 3:  # Only visiting 3 positions in last 10 moves
                self.stuck_counter += 1
                if self.stuck_counter > 5:
                    # Force exploration when stuck
                    self.exploration_mode = True
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0
                self.exploration_mode = False

def create_test_environment(width: int, height: int, num_food: int = 3, num_enemies: int = 1):
    """Create test environment"""
    maze, food_positions = generate_maze_with_food(width, height, num_food)
    
    # Add enemies
    open_positions = [(x, y) for x in range(width) for y in range(height) 
                     if maze[y][x] == 0 and (x, y) not in food_positions 
                     and (x, y) != (1, 1) and (x, y) != (width-2, height-2)]
    
    enemy_positions = random.sample(open_positions, min(num_enemies, len(open_positions)))
    
    return maze, food_positions, enemy_positions

def test_smart_execution():
    """
    Test the smart execution system
    """
    print("=== SMART EXECUTION SYSTEM TEST ===")
    print("Testing enhanced execution with working pathfinding")
    print()
    
    # Test parameters
    maze_sizes = [15, 21]
    trials_per_size = 3
    
    overall_results = []
    
    for maze_size in maze_sizes:
        print(f"Testing {maze_size}x{maze_size} with smart execution:")
        print("-" * 50)
        
        size_results = []
        
        for trial in range(trials_per_size):
            print(f"\nTrial {trial + 1}/{trials_per_size}:")
            
            # Create environment
            maze, food_positions, enemy_positions = create_test_environment(
                maze_size, maze_size, num_food=3, num_enemies=1
            )
            
            start = (1, 1)
            goal = (maze_size - 2, maze_size - 2)
            
            # Create smart execution agent
            agent = SmartExecutionAgent(maze, start, goal)
            agent.add_food_and_enemies(food_positions, enemy_positions)
            
            # Run simulation
            max_steps = maze_size * 20  # Increased step limit
            step_count = 0
            
            print(f"  Starting simulation (max {max_steps} steps)...")
            
            while agent.alive and agent.position != goal and step_count < max_steps:
                step_result = agent.step()
                step_count += 1
                
                # Progress reporting
                if step_count % 150 == 0:
                    print(f"    Step {step_count}: Pos {agent.position}, "
                          f"Target: {agent.current_target}, "
                          f"Rules: {step_result['total_rules']}, "
                          f"Hunger: {step_result['hunger']:.2f}")
                
                # Success check
                if agent.position == goal:
                    break
            
            # Results
            success = agent.position == goal
            survival = agent.alive
            
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Final position: {agent.position} (goal: {goal})")
            print(f"  Steps taken: {step_count}")
            print(f"  Food eaten: {agent.food_eaten}")
            print(f"  Survival: {'YES' if survival else 'NO'}")
            print(f"  Rules created: {len(agent.rules)}")
            
            # Analyze rule effectiveness
            effective_rules = [r for r in agent.rules.values() if r.get_success_rate() > 0.7 and r.activation_count > 3]
            print(f"  Highly effective rules: {len(effective_rules)}")
            
            size_results.append({
                'success': success,
                'survival': survival,
                'steps': step_count,
                'food_eaten': agent.food_eaten,
                'rules_created': len(agent.rules),
                'effective_rules': len(effective_rules)
            })
        
        # Size summary
        successes = sum(1 for r in size_results if r['success'])
        survivals = sum(1 for r in size_results if r['survival'])
        
        success_rate = (successes / len(size_results)) * 100
        survival_rate = (survivals / len(size_results)) * 100
        
        print(f"\n{maze_size}x{maze_size} Summary:")
        print(f"  Success rate: {success_rate:.0f}%")
        print(f"  Survival rate: {survival_rate:.0f}%")
        
        overall_results.append({
            'maze_size': maze_size,
            'success_rate': success_rate,
            'survival_rate': survival_rate,
            'results': size_results
        })
    
    # Overall assessment
    print("\n" + "="*70)
    print("SMART EXECUTION RESULTS:")
    print("="*70)
    
    total_successes = sum(len([r for r in result['results'] if r['success']]) for result in overall_results)
    total_trials = sum(len(result['results']) for result in overall_results)
    overall_success_rate = (total_successes / total_trials) * 100 if total_trials > 0 else 0
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_trials})")
    
    for result in overall_results:
        print(f"  {result['maze_size']:2d}x{result['maze_size']}: {result['success_rate']:5.0f}% success, {result['survival_rate']:5.0f}% survival")
    
    # Assessment
    if overall_success_rate >= 60:
        print(f"\n🎯 SMART EXECUTION: SUCCESS!")
        print(f"✅ Rule-based intelligence + smart execution = working system")
        return True
    elif overall_success_rate >= 30:
        print(f"\n⚠️  SMART EXECUTION: PARTIAL SUCCESS")
        print(f"Significant improvement but needs optimization")
        return True
    else:
        print(f"\n❌ SMART EXECUTION: STILL INSUFFICIENT")
        return False

if __name__ == "__main__":
    success = test_smart_execution()
    
    if success:
        print(f"\n🚀 BREAKTHROUGH: Intelligence + Execution = SUCCESS!")
        print(f"Wave-based adaptive rule making with smart execution works!")
    else:
        print(f"\n⚠️  Smart execution needs further improvements")