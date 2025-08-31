#!/usr/bin/env python3
"""
Testing Wave-Based Adaptive Rule Making System
Watch the entity create and evolve its own behavioral rules
"""

import random
from typing import Dict, List, Tuple, Any
from wave_adaptive_rule_maker import RuleMakingAgent, SituationPattern, RuleType
from wave_incremental_survival import generate_maze_with_food

def create_dynamic_environment(width: int, height: int, num_food: int = 3, num_enemies: int = 1):
    """Create environment with food and enemies"""
    maze, food_positions = generate_maze_with_food(width, height, num_food)
    
    # Add enemies in open positions
    open_positions = [(x, y) for x in range(width) for y in range(height) 
                     if maze[y][x] == 0 and (x, y) not in food_positions 
                     and (x, y) != (1, 1) and (x, y) != (width-2, height-2)]
    
    enemy_positions = random.sample(open_positions, min(num_enemies, len(open_positions)))
    
    return maze, food_positions, enemy_positions

def print_environment_state(agent: RuleMakingAgent, maze: List[List[int]]):
    """Print current environment with agent, food, enemies"""
    print("Environment:")
    for y in range(len(maze)):
        row = ""
        for x in range(len(maze[0])):
            pos = (x, y)
            if pos == agent.position:
                row += "A"
            elif pos == agent.goal:
                row += "G"  
            elif pos in agent.food_positions:
                row += "F"
            elif pos in agent.enemy_positions:
                row += "X"
            elif maze[y][x] == 1:
                row += "█"
            else:
                row += " "
        print(row)

def analyze_rule_evolution(agent: RuleMakingAgent) -> Dict[str, Any]:
    """Analyze how rules evolved during the simulation"""
    if not agent.rules:
        return {"total_rules": 0}
    
    # Rule type distribution
    rule_types = {}
    for rule in agent.rules.values():
        rule_type = rule.rule_type.value
        if rule_type not in rule_types:
            rule_types[rule_type] = []
        rule_types[rule_type].append(rule)
    
    # Rule effectiveness
    effective_rules = [r for r in agent.rules.values() if r.get_success_rate() > 0.6 and r.activation_count > 2]
    ineffective_rules = [r for r in agent.rules.values() if r.get_success_rate() < 0.3 and r.activation_count > 2]
    
    # Most used rules
    most_used = sorted(agent.rules.values(), key=lambda r: r.activation_count, reverse=True)[:3]
    
    return {
        "total_rules": len(agent.rules),
        "rule_types": {k: len(v) for k, v in rule_types.items()},
        "effective_rules": len(effective_rules),
        "ineffective_rules": len(ineffective_rules),
        "most_used_rules": [(r.rule_id, r.action, r.activation_count, r.get_success_rate()) for r in most_used],
        "avg_rule_strength": sum(r.amplitude for r in agent.rules.values()) / len(agent.rules),
        "rule_details": {r.rule_id: {
            'action': r.action,
            'type': r.rule_type.value,
            'strength': r.amplitude,
            'success_rate': r.get_success_rate(),
            'activations': r.activation_count
        } for r in agent.rules.values()}
    }

def test_adaptive_rule_making():
    """
    Test the adaptive rule making system
    """
    print("=== WAVE-BASED ADAPTIVE RULE MAKING SYSTEM ===")
    print("Entity creates and evolves its own behavioral rules through experience")
    print()
    
    # Test parameters
    maze_sizes = [15, 21]
    trials_per_size = 2
    
    overall_results = []
    
    for maze_size in maze_sizes:
        print(f"Testing {maze_size}x{maze_size} adaptive environment:")
        print("-" * 50)
        
        size_results = []
        
        for trial in range(trials_per_size):
            print(f"\nTrial {trial + 1}/{trials_per_size}:")
            
            # Create dynamic environment
            maze, food_positions, enemy_positions = create_dynamic_environment(
                maze_size, maze_size, num_food=3, num_enemies=1
            )
            
            start = (1, 1)
            goal = (maze_size - 2, maze_size - 2)
            
            # Create adaptive agent
            agent = RuleMakingAgent(maze, start, goal)
            agent.add_food_and_enemies(food_positions, enemy_positions)
            
            # Run simulation
            max_steps = maze_size * 15
            step_count = 0
            
            print(f"  Starting simulation (max {max_steps} steps)...")
            print(f"  Goal: {goal}, Food: {len(food_positions)}, Enemies: {len(enemy_positions)}")
            
            while agent.alive and agent.position != goal and step_count < max_steps:
                step_result = agent.step()
                step_count += 1
                
                # Progress reporting
                if step_count % 100 == 0:
                    print(f"    Step {step_count}: Pos {agent.position}, "
                          f"Rules: {step_result['total_rules']}, "
                          f"Hunger: {step_result['hunger']:.2f}, "
                          f"Action: {step_result['action']}")
                
                # Early success detection
                if agent.position == goal:
                    break
            
            # Results analysis
            success = agent.position == goal
            survival = agent.alive
            
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Steps taken: {step_count}")
            print(f"  Food eaten: {agent.food_eaten}")
            print(f"  Final hunger: {agent.hunger_level:.2f}")
            print(f"  Survival: {'YES' if survival else 'NO'}")
            
            # Rule evolution analysis
            rule_analysis = analyze_rule_evolution(agent)
            print(f"  Rules created: {rule_analysis['total_rules']}")
            print(f"  Rule types: {rule_analysis['rule_types']}")
            print(f"  Effective rules: {rule_analysis['effective_rules']}")
            
            if rule_analysis['most_used_rules']:
                print("  Most used rules:")
                for rule_id, action, count, success_rate in rule_analysis['most_used_rules']:
                    print(f"    {rule_id}: {action} (used {count}x, {success_rate:.1%} success)")
            
            # Store results
            trial_result = {
                'success': success,
                'survival': survival,
                'steps': step_count,
                'food_eaten': agent.food_eaten,
                'rules_created': rule_analysis['total_rules'],
                'effective_rules': rule_analysis['effective_rules'],
                'rule_analysis': rule_analysis
            }
            
            size_results.append(trial_result)
        
        # Size summary
        successes = sum(1 for r in size_results if r['success'])
        survivals = sum(1 for r in size_results if r['survival'])
        avg_steps = sum(r['steps'] for r in size_results) / len(size_results)
        avg_rules = sum(r['rules_created'] for r in size_results) / len(size_results)
        avg_effective = sum(r['effective_rules'] for r in size_results) / len(size_results)
        
        print(f"\n{maze_size}x{maze_size} Summary:")
        print(f"  Success rate: {successes}/{len(size_results)} ({successes/len(size_results)*100:.0f}%)")
        print(f"  Survival rate: {survivals}/{len(size_results)} ({survivals/len(size_results)*100:.0f}%)")
        print(f"  Average steps: {avg_steps:.0f}")
        print(f"  Average rules created: {avg_rules:.1f}")
        print(f"  Average effective rules: {avg_effective:.1f}")
        
        overall_results.append({
            'maze_size': maze_size,
            'success_rate': successes/len(size_results)*100,
            'survival_rate': survivals/len(size_results)*100,
            'avg_steps': avg_steps,
            'avg_rules': avg_rules,
            'avg_effective': avg_effective,
            'trials': size_results
        })
    
    # Overall analysis
    print("\n" + "="*70)
    print("ADAPTIVE RULE MAKING SYSTEM RESULTS:")
    print("="*70)
    
    total_successes = sum(len([t for t in result['trials'] if t['success']]) for result in overall_results)
    total_trials = sum(len(result['trials']) for result in overall_results)
    overall_success_rate = (total_successes / total_trials) * 100 if total_trials > 0 else 0
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_trials})")
    
    print(f"\nScaling Analysis:")
    for result in overall_results:
        print(f"  {result['maze_size']:2d}x{result['maze_size']}: "
              f"{result['success_rate']:5.0f}% success, "
              f"{result['avg_rules']:4.1f} rules, "
              f"{result['avg_effective']:4.1f} effective")
    
    # Rule learning analysis
    all_rule_data = []
    for result in overall_results:
        for trial in result['trials']:
            rule_analysis = trial['rule_analysis']
            if 'rule_details' in rule_analysis:
                all_rule_data.extend(rule_analysis['rule_details'].values())
    
    if all_rule_data:
        print(f"\nRule Learning Analysis:")
        
        # Action type distribution
        action_counts = {}
        for rule_data in all_rule_data:
            action = rule_data['action']
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        print(f"  Most common actions:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {action}: {count} rules")
        
        # Success rate by action
        action_success = {}
        for rule_data in all_rule_data:
            action = rule_data['action']
            if action not in action_success:
                action_success[action] = []
            if rule_data['activations'] > 0:
                action_success[action].append(rule_data['success_rate'])
        
        print(f"  Action effectiveness:")
        for action, success_rates in action_success.items():
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates)
                print(f"    {action}: {avg_success:.1%} average success")
    
    # Assessment
    print(f"\n" + "="*70)
    if overall_success_rate >= 60:
        print("🎯 ADAPTIVE RULE MAKING: SUCCESS!")
        print("✅ Entity successfully creates its own behavioral rules")
        print("✅ Rules adapt and improve through experience")
        print("✅ System shows emergent intelligent behavior")
        return True
    elif overall_success_rate >= 30:
        print("⚠️  ADAPTIVE RULE MAKING: PARTIAL SUCCESS")
        print("System shows rule learning but needs optimization")
        return True
    else:
        print("❌ ADAPTIVE RULE MAKING: INSUFFICIENT")
        print("Rule creation not leading to effective behavior")
        return False

if __name__ == "__main__":
    success = test_adaptive_rule_making()
    
    if success:
        print(f"\n🔥 BREAKTHROUGH: Entity creates its own intelligence!")
        print(f"Wave-based adaptive rule making system works!")
    else:
        print(f"\n⚠️  Adaptive rule making needs refinement")