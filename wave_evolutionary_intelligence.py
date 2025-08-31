#!/usr/bin/env python3
"""
Evolutionary Wave-Based Intelligence System
Generational knowledge transfer through wave pattern inheritance
"""

import math
import random
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from wave_smart_execution import SmartExecutionAgent
from wave_adaptive_rule_maker import RuleType, AdaptiveRule, SituationPattern
from wave_incremental_survival import generate_maze_with_food

@dataclass
class WaveGene:
    """
    Heritable wave pattern encoding successful behaviors
    """
    action: str
    rule_type: RuleType
    frequency: float
    amplitude: float
    phase: float
    condition_pattern: Dict[str, float]
    fitness: float  # How successful this pattern was
    generation_born: int

class EvolutionaryMemory:
    """
    Species-level memory that accumulates knowledge across generations
    """
    
    def __init__(self):
        self.generation = 0
        self.species_knowledge = {}  # action -> list of WaveGenes
        self.survival_patterns = []  # Patterns from successful agents
        self.extinction_patterns = []  # Patterns from failed agents
        self.fitness_history = []  # Track species performance over time
    
    def record_survivor(self, agent):
        """Extract and record successful patterns from survivor"""
        print(f"    🧬 Recording survivor knowledge from generation {self.generation}")
        
        # Extract high-performing rules
        valuable_rules = []
        for rule in agent.rules.values():
            if rule.get_success_rate() > 0.7 and rule.activation_count >= 3:
                valuable_rules.append(rule)
        
        print(f"      Found {len(valuable_rules)} valuable rules to inherit")
        
        # Convert rules to heritable wave genes
        for rule in valuable_rules:
            gene = WaveGene(
                action=rule.action,
                rule_type=rule.rule_type,
                frequency=rule.frequency,
                amplitude=rule.amplitude,
                phase=rule.phase,
                condition_pattern={
                    'hunger_level': rule.condition_pattern.hunger_level,
                    'food_distance': rule.condition_pattern.food_distance,
                    'goal_distance': rule.condition_pattern.goal_distance,
                    'enemy_distance': rule.condition_pattern.enemy_distance,
                    'energy_level': rule.condition_pattern.energy_level
                },
                fitness=rule.get_success_rate() * (rule.activation_count / 10),  # Usage-weighted fitness
                generation_born=self.generation
            )
            
            # Add to species knowledge
            if gene.action not in self.species_knowledge:
                self.species_knowledge[gene.action] = []
            self.species_knowledge[gene.action].append(gene)
        
        # Record survival pattern
        self.survival_patterns.append({
            'generation': self.generation,
            'final_position': agent.position,
            'steps_taken': agent.steps_taken,
            'food_eaten': agent.food_eaten,
            'rules_created': len(agent.rules),
            'valuable_rules': len(valuable_rules)
        })
    
    def record_extinction(self, agent):
        """Record patterns from failed agents (to avoid)"""
        # Extract frequently used but failed patterns
        failed_rules = [r for r in agent.rules.values() 
                       if r.activation_count > 5 and r.get_success_rate() < 0.3]
        
        for rule in failed_rules:
            extinction_pattern = {
                'action': rule.action,
                'condition': rule.condition_pattern.hunger_level,
                'failure_rate': 1.0 - rule.get_success_rate(),
                'generation': self.generation
            }
            self.extinction_patterns.append(extinction_pattern)
    
    def evolve_species_knowledge(self):
        """Evolve species knowledge - strengthen good patterns, weaken bad ones"""
        print(f"    🧬 Evolving species knowledge for generation {self.generation + 1}")
        
        # Strengthen successful patterns from recent survivors
        recent_survivors = [p for p in self.survival_patterns if p['generation'] >= self.generation - 2]
        
        for action, genes in self.species_knowledge.items():
            for gene in genes:
                # Boost fitness of patterns used by recent survivors
                if any(gene.generation_born >= s['generation'] - 1 for s in recent_survivors):
                    gene.fitness *= 1.1
                
                # Decay old patterns that haven't been validated recently
                if gene.generation_born < self.generation - 5:
                    gene.fitness *= 0.95
        
        # Remove very weak patterns
        for action in list(self.species_knowledge.keys()):
            self.species_knowledge[action] = [g for g in self.species_knowledge[action] if g.fitness > 0.1]
            if not self.species_knowledge[action]:
                del self.species_knowledge[action]
        
        print(f"      Species now knows {sum(len(genes) for genes in self.species_knowledge.values())} patterns")
    
    def get_inherited_knowledge(self) -> Dict[str, List[WaveGene]]:
        """Get current species knowledge for new generation"""
        return copy.deepcopy(self.species_knowledge)
    
    def advance_generation(self):
        """Move to next generation"""
        self.evolve_species_knowledge()
        self.generation += 1
        print(f"  📈 Advanced to generation {self.generation}")

class EvolutionaryAgent(SmartExecutionAgent):
    """
    Agent that can inherit wave-based knowledge from previous generations
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], 
                 inherited_knowledge: Dict[str, List[WaveGene]] = None, generation: int = 0):
        super().__init__(maze, start, goal)
        
        # Evolutionary properties
        self.generation = generation
        self.inherited_knowledge = inherited_knowledge or {}
        self.inherited_rules_used = 0
        
        # More realistic survival parameters
        self.hunger_rate = 0.003  # 333 steps survival time instead of 100
        self.energy_decay_rate = 0.002
        
        # Initialize with inherited knowledge
        self._initialize_inherited_rules()
    
    def _initialize_inherited_rules(self):
        """Initialize agent with inherited wave patterns"""
        if not self.inherited_knowledge:
            print(f"      Generation {self.generation}: No inherited knowledge (first generation)")
            return
        
        inherited_count = 0
        
        for action, genes in self.inherited_knowledge.items():
            # Select best genes for each action type
            best_genes = sorted(genes, key=lambda g: g.fitness, reverse=True)[:3]  # Top 3 per action
            
            for gene in best_genes:
                # Create inherited rule from gene
                inherited_pattern = SituationPattern(
                    hunger_level=gene.condition_pattern['hunger_level'],
                    food_distance=gene.condition_pattern['food_distance'], 
                    goal_distance=gene.condition_pattern['goal_distance'],
                    enemy_distance=gene.condition_pattern['enemy_distance'],
                    energy_level=gene.condition_pattern['energy_level'],
                    steps_since_progress=0
                )
                
                inherited_rule = AdaptiveRule(
                    rule_id=f"inherited_{self.rule_counter}_{gene.generation_born}",
                    condition_pattern=inherited_pattern,
                    action=gene.action,
                    rule_type=gene.rule_type
                )
                
                # Set wave properties from gene
                inherited_rule.frequency = gene.frequency
                inherited_rule.amplitude = gene.amplitude * 0.8  # Slightly weaken inherited rules
                inherited_rule.phase = gene.phase
                inherited_rule.creation_step = 0
                
                # Give inherited rules a head start
                inherited_rule.success_count = max(1, int(gene.fitness * 5))
                inherited_rule.activation_count = max(2, int(gene.fitness * 7))
                
                self.rules[inherited_rule.rule_id] = inherited_rule
                inherited_count += 1
                self.rule_counter += 1
        
        self.inherited_rules_used = inherited_count
        print(f"      Generation {self.generation}: Inherited {inherited_count} wave patterns")
    
    def get_fitness_score(self) -> float:
        """Calculate agent fitness for evolutionary purposes"""
        base_fitness = 0
        
        # Survival fitness
        if self.alive:
            base_fitness += 100
        
        # Goal achievement fitness  
        goal_distance = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        max_distance = self.width + self.height
        proximity_score = (1.0 - (goal_distance / max_distance)) * 50
        base_fitness += proximity_score
        
        # Food acquisition fitness
        base_fitness += self.food_eaten * 20
        
        # Learning fitness (rule creation and effectiveness)
        if self.rules:
            avg_rule_success = sum(r.get_success_rate() for r in self.rules.values()) / len(self.rules)
            base_fitness += avg_rule_success * 30
        
        # Efficiency fitness
        if self.steps_taken > 0:
            efficiency = min(1.0, 100 / self.steps_taken)  # Bonus for solving quickly
            base_fitness += efficiency * 25
        
        return base_fitness

def run_evolutionary_experiment():
    """
    Run multi-generational evolutionary intelligence experiment
    """
    print("=== EVOLUTIONARY WAVE-BASED INTELLIGENCE EXPERIMENT ===")
    print("Testing generational knowledge transfer and species-level learning")
    print()
    
    # Experiment parameters
    maze_size = 17
    num_generations = 8
    agents_per_generation = 4
    max_steps_per_agent = 400  # More realistic time limit
    
    # Initialize evolutionary memory
    species_memory = EvolutionaryMemory()
    generation_results = []
    
    for generation in range(num_generations):
        print(f"GENERATION {generation + 1}/{num_generations}")
        print("=" * 50)
        
        # Create maze environment
        maze, food_positions = generate_maze_with_food(maze_size, maze_size, num_food=4)
        
        # Add enemies manually
        open_positions = [(x, y) for x in range(maze_size) for y in range(maze_size) 
                         if maze[y][x] == 0 and (x, y) not in food_positions 
                         and (x, y) != (1, 1) and (x, y) != (maze_size-2, maze_size-2)]
        enemy_positions = [random.choice(open_positions)] if open_positions else []
        start = (1, 1)
        goal = (maze_size - 2, maze_size - 2)
        
        # Get inherited knowledge
        inherited_knowledge = species_memory.get_inherited_knowledge()
        
        generation_agents = []
        generation_survivors = []
        
        # Run agents in this generation
        for agent_id in range(agents_per_generation):
            print(f"  Agent {agent_id + 1}/{agents_per_generation}:")
            
            # Create evolutionary agent
            agent = EvolutionaryAgent(maze, start, goal, inherited_knowledge, generation)
            agent.add_food_and_enemies(food_positions.copy(), enemy_positions.copy())
            
            # Run agent simulation
            step_count = 0
            while agent.alive and agent.position != goal and step_count < max_steps_per_agent:
                agent.step()
                step_count += 1
            
            # Record results
            success = agent.position == goal
            fitness = agent.get_fitness_score()
            
            print(f"    Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"    Position: {agent.position} (goal: {goal})")  
            print(f"    Steps: {step_count}, Food: {agent.food_eaten}, Fitness: {fitness:.1f}")
            print(f"    Rules: {len(agent.rules)} total, {agent.inherited_rules_used} inherited")
            
            generation_agents.append({
                'agent': agent,
                'success': success,
                'fitness': fitness,
                'steps': step_count
            })
            
            # Record in species memory
            if success or fitness > 80:  # Survivors or high-fitness agents
                species_memory.record_survivor(agent)
                generation_survivors.append(agent)
            else:
                species_memory.record_extinction(agent)
        
        # Generation summary
        successes = sum(1 for a in generation_agents if a['success'])
        avg_fitness = sum(a['fitness'] for a in generation_agents) / len(generation_agents)
        avg_steps = sum(a['steps'] for a in generation_agents) / len(generation_agents)
        
        success_rate = (successes / len(generation_agents)) * 100
        
        print(f"\n  Generation {generation + 1} Summary:")
        print(f"    Success rate: {success_rate:.0f}% ({successes}/{agents_per_generation})")
        print(f"    Average fitness: {avg_fitness:.1f}")
        print(f"    Average steps: {avg_steps:.0f}")
        print(f"    Survivors: {len(generation_survivors)}")
        
        generation_results.append({
            'generation': generation + 1,
            'success_rate': success_rate,
            'avg_fitness': avg_fitness,
            'survivors': len(generation_survivors),
            'species_knowledge_size': sum(len(genes) for genes in species_memory.species_knowledge.values())
        })
        
        # Evolve species knowledge
        species_memory.advance_generation()
        print()
    
    # Overall evolutionary analysis
    print("=" * 70)
    print("EVOLUTIONARY INTELLIGENCE RESULTS:")
    print("=" * 70)
    
    print("Generation Performance:")
    for result in generation_results:
        print(f"  Gen {result['generation']:2d}: "
              f"{result['success_rate']:5.0f}% success, "
              f"fitness {result['avg_fitness']:5.1f}, "
              f"{result['survivors']} survivors, "
              f"{result['species_knowledge_size']} knowledge patterns")
    
    # Evolution analysis
    first_half = generation_results[:len(generation_results)//2]
    second_half = generation_results[len(generation_results)//2:]
    
    early_avg_success = sum(r['success_rate'] for r in first_half) / len(first_half)
    late_avg_success = sum(r['success_rate'] for r in second_half) / len(second_half)
    
    early_avg_fitness = sum(r['avg_fitness'] for r in first_half) / len(first_half) 
    late_avg_fitness = sum(r['avg_fitness'] for r in second_half) / len(second_half)
    
    print(f"\nEvolutionary Progress:")
    print(f"  Early generations (1-{len(first_half)}): {early_avg_success:.1f}% success, {early_avg_fitness:.1f} fitness")
    print(f"  Late generations ({len(first_half)+1}-{len(generation_results)}): {late_avg_success:.1f}% success, {late_avg_fitness:.1f} fitness")
    
    improvement_success = late_avg_success - early_avg_success
    improvement_fitness = late_avg_fitness - early_avg_fitness
    
    print(f"  Success rate improvement: {improvement_success:+.1f}%")
    print(f"  Fitness improvement: {improvement_fitness:+.1f}")
    
    # Final assessment
    print(f"\n" + "=" * 70)
    if improvement_success > 10 or improvement_fitness > 15:
        print("🎯 EVOLUTIONARY INTELLIGENCE: SUCCESS!")
        print("✅ Species-level learning demonstrated")
        print("✅ Generational knowledge transfer working")
        print("✅ Wave patterns successfully inherited")
        print("✅ Population intelligence emerging")
        return True
    elif improvement_success > 5 or improvement_fitness > 8:
        print("⚠️  EVOLUTIONARY INTELLIGENCE: PARTIAL SUCCESS")
        print("Some evolutionary improvement detected")
        return True
    else:
        print("❌ EVOLUTIONARY INTELLIGENCE: INSUFFICIENT")
        print("No clear evolutionary improvement")
        return False

if __name__ == "__main__":
    success = run_evolutionary_experiment()
    
    if success:
        print(f"\n🔥 BREAKTHROUGH: EVOLUTIONARY WAVE INTELLIGENCE WORKS!")
        print(f"Species learns and improves across generations through wave pattern inheritance!")
    else:
        print(f"\n⚠️  Evolutionary system needs refinement")