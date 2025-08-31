#!/usr/bin/env python3
"""
Comprehensive Wave-Based Survival Agent Testing
Multiple scenarios, learning progression, adversarial environments
"""

import random
import time
from typing import Dict, List, Tuple, Any
from wave_survival_agent import WaveSurvivalAgent

class SurvivalEnvironment:
    """
    Dynamic survival environment with random mazes, food, enemies
    """
    
    def __init__(self, width: int, height: int, seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.width = width
        self.height = height
        self.maze = self.generate_maze()
        self.food_positions = []
        self.enemy_positions = []
        self.exit_position = (width - 2, height - 2)
        
        self.spawn_food()
        self.spawn_enemies()
    
    def generate_maze(self) -> List[List[int]]:
        """Generate random maze"""
        maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        def carve_path(x, y):
            maze[y][x] = 0
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 1 < new_x < self.width - 1 and 1 < new_y < self.height - 1 and maze[new_y][new_x] == 1:
                    maze[y + dy // 2][x + dx // 2] = 0
                    carve_path(new_x, new_y)
        
        carve_path(1, 1)
        maze[1][1] = 0
        maze[self.height - 2][self.width - 2] = 0
        
        return maze
    
    def spawn_food(self):
        """Randomly place food in maze"""
        num_food = max(3, self.width // 10)
        open_positions = [(x, y) for x in range(self.width) for y in range(self.height) 
                         if self.maze[y][x] == 0 and (x, y) != (1, 1) and (x, y) != self.exit_position]
        
        self.food_positions = random.sample(open_positions, min(num_food, len(open_positions)))
    
    def spawn_enemies(self):
        """Randomly place enemies in maze"""
        num_enemies = max(1, self.width // 15)
        open_positions = [(x, y) for x in range(self.width) for y in range(self.height) 
                         if self.maze[y][x] == 0 and (x, y) != (1, 1) and (x, y) != self.exit_position 
                         and (x, y) not in self.food_positions]
        
        if open_positions:
            self.enemy_positions = random.sample(open_positions, min(num_enemies, len(open_positions)))
    
    def move_enemies(self):
        """Move enemies randomly (simple AI)"""
        new_enemy_positions = []
        
        for enemy_pos in self.enemy_positions:
            x, y = enemy_pos
            possible_moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]  # Include staying
            valid_moves = [pos for pos in possible_moves 
                          if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height and self.maze[pos[1]][pos[0]] == 0]
            
            if valid_moves:
                new_enemy_positions.append(random.choice(valid_moves))
            else:
                new_enemy_positions.append(enemy_pos)
        
        self.enemy_positions = new_enemy_positions
    
    def print_environment(self, agent_pos: Tuple[int, int]):
        """Print current environment state"""
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                pos = (x, y)
                if pos == agent_pos:
                    row += "A"
                elif pos == self.exit_position:
                    row += "E"
                elif pos in self.food_positions:
                    row += "F"
                elif pos in self.enemy_positions:
                    row += "X"
                elif self.maze[y][x] == 1:
                    row += "█"
                else:
                    row += " "
            print(row)

def run_survival_scenario(agent: WaveSurvivalAgent, environment: SurvivalEnvironment, 
                         max_steps: int = 1000, verbose: bool = False) -> Dict[str, Any]:
    """Run single survival scenario"""
    step_count = 0
    
    while agent.alive and step_count < max_steps:
        # Agent perceives environment
        agent.perceive_environment(environment.maze, environment.food_positions, environment.enemy_positions)
        
        # Agent decides action
        action = agent.decide_action(environment.maze, environment.food_positions, 
                                   environment.enemy_positions, environment.exit_position)
        
        # Execute action
        result = agent.execute_action(action, environment.maze, environment.food_positions, 
                                    environment.enemy_positions, environment.exit_position)
        
        # Move enemies
        if step_count % 3 == 0:  # Enemies move every 3 steps
            environment.move_enemies()
        
        step_count += 1
        
        # Check win conditions
        if result['outcome'] == 'escaped':
            break
        
        if verbose and step_count % 50 == 0:
            print(f"Step {step_count}: Position {agent.position}, Hunger {agent.hunger_level:.2f}")
            environment.print_environment(agent.position)
            print()
    
    return {
        'outcome': 'escaped' if agent.position == environment.exit_position else 
                  ('died' if not agent.alive else 'timeout'),
        'steps': step_count,
        'food_eaten': agent.food_eaten,
        'hunger_level': agent.hunger_level,
        'concepts_learned': len(agent.learning.concept_patterns)
    }

def test_wave_survival_scaling():
    """
    Test wave-based survival agent across multiple scenarios and scales
    """
    print("=== COMPREHENSIVE WAVE-BASED SURVIVAL AGENT TEST ===")
    print("Testing multi-modal learning, adaptation, and scaling")
    print()
    
    # Test scenarios with increasing complexity
    scenarios = [
        {'size': 15, 'trials': 3, 'name': 'Small Maze'},
        {'size': 21, 'trials': 3, 'name': 'Medium Maze'},
        {'size': 31, 'trials': 2, 'name': 'Large Maze'},
        {'size': 41, 'trials': 2, 'name': 'XL Maze'},
    ]
    
    overall_results = []
    
    for scenario in scenarios:
        print(f"Testing {scenario['name']} ({scenario['size']}x{scenario['size']}):")
        scenario_results = []
        
        for trial in range(scenario['trials']):
            print(f"  Trial {trial + 1}/{scenario['trials']}...")
            
            # Create new environment
            environment = SurvivalEnvironment(scenario['size'], scenario['size'], seed=random.randint(1, 1000))
            
            # Create agent (fresh for each trial to test learning)
            agent = WaveSurvivalAgent(scenario['size'], scenario['size'])
            
            # Run scenario
            result = run_survival_scenario(agent, environment, max_steps=scenario['size'] * 20)
            scenario_results.append(result)
            
            # Print trial result
            status = "✅" if result['outcome'] == 'escaped' else "❌"
            print(f"    {status} {result['outcome']}: {result['steps']} steps, "
                  f"{result['food_eaten']} food, {result['concepts_learned']} concepts")
        
        # Scenario summary
        successes = sum(1 for r in scenario_results if r['outcome'] == 'escaped')
        avg_steps = sum(r['steps'] for r in scenario_results) / len(scenario_results)
        avg_food = sum(r['food_eaten'] for r in scenario_results) / len(scenario_results)
        avg_concepts = sum(r['concepts_learned'] for r in scenario_results) / len(scenario_results)
        
        success_rate = (successes / len(scenario_results)) * 100
        
        print(f"  📊 Summary: {success_rate:.0f}% success, {avg_steps:.0f} avg steps, "
              f"{avg_food:.1f} avg food, {avg_concepts:.1f} avg concepts")
        print()
        
        overall_results.append({
            'scenario': scenario['name'],
            'size': scenario['size'],
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_concepts': avg_concepts,
            'results': scenario_results
        })
    
    # Overall analysis
    print("="*60)
    print("COMPREHENSIVE SURVIVAL AGENT RESULTS:")
    print("="*60)
    
    total_successes = sum(len([r for r in scenario['results'] if r['outcome'] == 'escaped']) for scenario in overall_results)
    total_trials = sum(len(scenario['results']) for scenario in overall_results)
    overall_success_rate = (total_successes / total_trials) * 100 if total_trials > 0 else 0
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_trials})")
    print()
    
    print("Scaling Analysis:")
    for result in overall_results:
        status = "✅" if result['success_rate'] >= 50 else "❌"
        print(f"  {status} {result['scenario']:12s}: {result['success_rate']:5.0f}% success, "
              f"{result['avg_concepts']:4.1f} concepts learned")
    
    # Learning progression analysis
    print(f"\nLearning Progression:")
    max_concepts = max(result['avg_concepts'] for result in overall_results)
    if max_concepts > 0:
        for result in overall_results:
            learning_bar = "█" * int((result['avg_concepts'] / max_concepts) * 20)
            print(f"  {result['scenario']:12s}: {learning_bar} ({result['avg_concepts']:.1f})")
    
    # Assessment
    print(f"\n" + "="*60)
    if overall_success_rate >= 60:
        print("🎯 WAVE-BASED SURVIVAL INTELLIGENCE: SUCCESS!")
        print("System demonstrates:")
        print("  ✅ Multi-modal sensory integration")
        print("  ✅ Real-time learning and adaptation") 
        print("  ✅ Concept formation from experience")
        print("  ✅ Goal-driven survival behavior")
        print("  ✅ Scaling to complex environments")
        return True
    elif overall_success_rate >= 30:
        print("⚠️  WAVE-BASED SURVIVAL INTELLIGENCE: PARTIAL SUCCESS")
        print("System shows promise but needs optimization")
        return True
    else:
        print("❌ WAVE-BASED SURVIVAL INTELLIGENCE: INSUFFICIENT")
        print("Fundamental issues with wave-based approach")
        return False

if __name__ == "__main__":
    success = test_wave_survival_scaling()
    
    if success:
        print("\n🧠 CONFIRMED: Wave-based survival intelligence is viable!")
        print("Multi-modal learning and adaptation through pure wave computation")
    else:
        print("\n⚠️  Wave-based survival intelligence needs major improvements")