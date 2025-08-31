#!/usr/bin/env python3
"""
WAVE-BASED ADAPTIVE RULE MAKER
The entity creates and evolves its own behavioral rules through experience
No hardcoded behaviors - pure emergent rule formation
"""

import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RuleType(Enum):
    SURVIVAL = "survival"
    RESOURCE = "resource" 
    GOAL = "goal"
    EXPLORATION = "exploration"
    SOCIAL = "social"  # For future enemy interactions

@dataclass
class SituationPattern:
    """Environmental pattern that triggers rules"""
    hunger_level: float
    food_distance: float
    goal_distance: float
    enemy_distance: float
    energy_level: float
    steps_since_progress: int

class AdaptiveRule:
    """
    A rule that the entity creates and modifies based on experience
    """
    
    def __init__(self, rule_id: str, condition_pattern: SituationPattern, action: str, rule_type: RuleType):
        self.rule_id = rule_id
        self.condition_pattern = condition_pattern
        self.action = action
        self.rule_type = rule_type
        
        # Wave properties
        self.frequency = self._calculate_rule_frequency()
        self.amplitude = 0.5  # Start with medium strength
        self.phase = random.uniform(0, 2 * math.pi)
        
        # Experience tracking
        self.activation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.creation_step = 0
        
        # Adaptation parameters
        self.success_amplifier = 1.2
        self.failure_dampener = 0.8
        self.min_amplitude = 0.1
        self.max_amplitude = 2.0
    
    def _calculate_rule_frequency(self) -> float:
        """Encode rule pattern as frequency"""
        # Different rule types get different base frequencies
        base_frequencies = {
            RuleType.SURVIVAL: 60,    # High priority
            RuleType.RESOURCE: 40,    # Medium priority  
            RuleType.GOAL: 20,        # Lower priority
            RuleType.EXPLORATION: 10, # Lowest priority
            RuleType.SOCIAL: 30
        }
        
        base_freq = base_frequencies[self.rule_type]
        
        # Modulate frequency based on condition pattern
        pattern_modifier = (self.condition_pattern.hunger_level * 10 + 
                           self.condition_pattern.energy_level * 5 +
                           self.condition_pattern.steps_since_progress * 0.1)
        
        return base_freq + pattern_modifier
    
    def matches_situation(self, current_situation: SituationPattern, tolerance: float = 0.2) -> float:
        """Check how well this rule matches current situation (0-1 score)"""
        matches = 0
        total_checks = 0
        
        # Check each condition with tolerance
        conditions = [
            ('hunger_level', self.condition_pattern.hunger_level, current_situation.hunger_level),
            ('food_distance', self.condition_pattern.food_distance, current_situation.food_distance),
            ('goal_distance', self.condition_pattern.goal_distance, current_situation.goal_distance),
            ('enemy_distance', self.condition_pattern.enemy_distance, current_situation.enemy_distance),
            ('energy_level', self.condition_pattern.energy_level, current_situation.energy_level),
        ]
        
        for name, expected, actual in conditions:
            total_checks += 1
            
            if expected < 0:  # Wildcard condition
                matches += 1
            else:
                diff = abs(expected - actual)
                if diff <= tolerance:
                    matches += 1
                elif diff <= tolerance * 2:
                    matches += 0.5  # Partial match
        
        return matches / total_checks if total_checks > 0 else 0
    
    def calculate_activation_strength(self, situation_match: float, time_step: float) -> float:
        """Calculate how strongly this rule should activate"""
        # Base activation from situation match and rule strength
        base_strength = situation_match * self.amplitude
        
        # Wave modulation based on frequency and time
        wave_modulation = math.sin(2 * math.pi * self.frequency * time_step * 0.01 + self.phase)
        wave_factor = (wave_modulation + 1) / 2  # Normalize to 0-1
        
        return base_strength * wave_factor
    
    def apply_outcome(self, success: bool, step: int):
        """Adapt rule based on outcome"""
        self.activation_count += 1
        
        if success:
            self.success_count += 1
            self.amplitude = min(self.amplitude * self.success_amplifier, self.max_amplitude)
        else:
            self.failure_count += 1
            self.amplitude = max(self.amplitude * self.failure_dampener, self.min_amplitude)
        
        # Rules that haven't been used recently decay slightly
        steps_since_creation = step - self.creation_step
        if self.activation_count == 0 and steps_since_creation > 100:
            self.amplitude *= 0.99
    
    def get_success_rate(self) -> float:
        """Calculate success rate of this rule"""
        if self.activation_count == 0:
            return 0.5  # Unknown
        return self.success_count / self.activation_count
    
    def __str__(self) -> str:
        success_rate = self.get_success_rate()
        return f"Rule[{self.rule_id}]: {self.action} | Strength: {self.amplitude:.2f} | Success: {success_rate:.1%}"

class RuleMakingAgent:
    """
    Agent that creates, applies, and evolves its own behavioral rules
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start = start
        self.goal = goal
        
        # Agent state
        self.position = start
        self.hunger_level = 0.0
        self.energy_level = 1.0
        self.steps_taken = 0
        self.steps_since_progress = 0
        self.food_eaten = 0
        self.alive = True
        
        # Environment
        self.food_positions = []
        self.enemy_positions = []
        
        # Rule system
        self.rules = {}  # rule_id -> AdaptiveRule
        self.rule_counter = 0
        self.active_rule = None
        self.rule_creation_threshold = 0.3  # Create new rule if no good match
        
        # Experience tracking
        self.path = [start]
        self.last_position = start
        self.experience_history = []
    
    def add_food_and_enemies(self, food_positions: List[Tuple[int, int]], enemy_positions: List[Tuple[int, int]] = None):
        """Add environmental elements"""
        self.food_positions = food_positions.copy()
        self.enemy_positions = enemy_positions.copy() if enemy_positions else []
    
    def get_current_situation(self) -> SituationPattern:
        """Analyze current environmental situation"""
        # Calculate distances
        goal_distance = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        
        if self.food_positions:
            food_distances = [abs(self.position[0] - f[0]) + abs(self.position[1] - f[1]) for f in self.food_positions]
            food_distance = min(food_distances)
        else:
            food_distance = 999  # No food available
        
        if self.enemy_positions:
            enemy_distances = [abs(self.position[0] - e[0]) + abs(self.position[1] - e[1]) for e in self.enemy_positions]
            enemy_distance = min(enemy_distances)
        else:
            enemy_distance = 999  # No enemies
        
        return SituationPattern(
            hunger_level=self.hunger_level,
            food_distance=food_distance,
            goal_distance=goal_distance,
            enemy_distance=enemy_distance,
            energy_level=self.energy_level,
            steps_since_progress=self.steps_since_progress
        )
    
    def find_best_rule_match(self, situation: SituationPattern) -> Tuple[Optional[AdaptiveRule], float]:
        """Find the rule that best matches current situation"""
        if not self.rules:
            return None, 0
        
        best_rule = None
        best_activation = 0
        
        time_step = self.steps_taken * 0.1
        
        for rule in self.rules.values():
            situation_match = rule.matches_situation(situation)
            activation_strength = rule.calculate_activation_strength(situation_match, time_step)
            
            if activation_strength > best_activation:
                best_activation = activation_strength
                best_rule = rule
        
        return best_rule, best_activation
    
    def create_new_rule(self, situation: SituationPattern) -> AdaptiveRule:
        """Create a new rule based on current situation"""
        self.rule_counter += 1
        rule_id = f"rule_{self.rule_counter}"
        
        # Determine rule type and action based on situation
        if situation.hunger_level > 0.6:
            rule_type = RuleType.SURVIVAL
            action = "seek_food"
        elif situation.enemy_distance < 5:
            rule_type = RuleType.SURVIVAL  
            action = "avoid_enemy"
        elif situation.goal_distance < 10:
            rule_type = RuleType.GOAL
            action = "seek_goal"
        elif situation.steps_since_progress > 20:
            rule_type = RuleType.EXPLORATION
            action = "explore_new_area"
        else:
            rule_type = RuleType.GOAL
            action = "seek_goal"
        
        # Create rule with some tolerance in conditions
        tolerant_pattern = SituationPattern(
            hunger_level=situation.hunger_level,
            food_distance=situation.food_distance,
            goal_distance=situation.goal_distance,
            enemy_distance=situation.enemy_distance,
            energy_level=situation.energy_level,
            steps_since_progress=situation.steps_since_progress
        )
        
        new_rule = AdaptiveRule(rule_id, tolerant_pattern, action, rule_type)
        new_rule.creation_step = self.steps_taken
        
        self.rules[rule_id] = new_rule
        
        print(f"    🧠 Created new rule: {action} (hunger: {situation.hunger_level:.2f}, goal_dist: {situation.goal_distance})")
        
        return new_rule
    
    def execute_action(self, action: str) -> str:
        """Execute the chosen action"""
        if action == "seek_food":
            return self._seek_food()
        elif action == "seek_goal":
            return self._seek_goal()
        elif action == "avoid_enemy":
            return self._avoid_enemy()
        elif action == "explore_new_area":
            return self._explore()
        else:
            return self._seek_goal()  # Default
    
    def _get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get valid adjacent positions"""
        x, y = self.position
        moves = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                self.maze[new_y][new_x] == 0):
                moves.append((new_x, new_y))
        
        return moves
    
    def _seek_food(self) -> str:
        """Move toward nearest food"""
        if not self.food_positions:
            return "no_food_available"
        
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return "trapped"
        
        # Find move that gets closer to nearest food
        nearest_food = min(self.food_positions, 
                          key=lambda f: abs(f[0] - self.position[0]) + abs(f[1] - self.position[1]))
        
        best_move = min(valid_moves,
                       key=lambda m: abs(m[0] - nearest_food[0]) + abs(m[1] - nearest_food[1]))
        
        self.position = best_move
        return "moved_toward_food"
    
    def _seek_goal(self) -> str:
        """Move toward goal"""
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return "trapped"
        
        # Find move that gets closer to goal
        best_move = min(valid_moves,
                       key=lambda m: abs(m[0] - self.goal[0]) + abs(m[1] - self.goal[1]))
        
        self.position = best_move
        return "moved_toward_goal"
    
    def _avoid_enemy(self) -> str:
        """Move away from nearest enemy"""
        if not self.enemy_positions:
            return self._seek_goal()  # No enemies, default to goal
        
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return "trapped"
        
        # Find move that gets farther from nearest enemy
        nearest_enemy = min(self.enemy_positions,
                           key=lambda e: abs(e[0] - self.position[0]) + abs(e[1] - self.position[1]))
        
        best_move = max(valid_moves,
                       key=lambda m: abs(m[0] - nearest_enemy[0]) + abs(m[1] - nearest_enemy[1]))
        
        self.position = best_move
        return "moved_away_from_enemy"
    
    def _explore(self) -> str:
        """Explore new areas"""
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return "trapped"
        
        # Prefer moves to positions we haven't visited
        unvisited_moves = [m for m in valid_moves if m not in self.path]
        
        if unvisited_moves:
            self.position = random.choice(unvisited_moves)
            return "explored_new_area"
        else:
            # All moves visited, pick randomly
            self.position = random.choice(valid_moves)
            return "explored_revisited_area"
    
    def update_state(self):
        """Update agent internal state"""
        # Update hunger and energy
        self.hunger_level = min(1.0, self.hunger_level + 0.01)
        self.energy_level = max(0.0, self.energy_level - 0.005)
        
        # Check food consumption
        if self.position in self.food_positions:
            self.food_positions.remove(self.position)
            self.food_eaten += 1
            self.hunger_level = max(0.0, self.hunger_level - 0.3)
            self.energy_level = min(1.0, self.energy_level + 0.2)
        
        # Update progress tracking
        if self.position != self.last_position:
            # Check if we made progress toward goal
            old_distance = abs(self.last_position[0] - self.goal[0]) + abs(self.last_position[1] - self.goal[1])
            new_distance = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
            
            if new_distance < old_distance:
                self.steps_since_progress = 0  # Made progress
            else:
                self.steps_since_progress += 1
            
            self.last_position = self.position
        else:
            self.steps_since_progress += 1
        
        # Add to path
        self.path.append(self.position)
        
        # Check survival
        if self.hunger_level >= 1.0:
            self.alive = False
    
    def step(self) -> Dict[str, Any]:
        """Single step of adaptive behavior"""
        if not self.alive:
            return {'alive': False}
        
        self.steps_taken += 1
        
        # Analyze current situation
        situation = self.get_current_situation()
        
        # Find best rule match
        best_rule, activation_strength = self.find_best_rule_match(situation)
        
        # Create new rule if no good match
        if activation_strength < self.rule_creation_threshold:
            best_rule = self.create_new_rule(situation)
            activation_strength = 0.5
        
        # Execute rule action
        self.active_rule = best_rule
        action_result = self.execute_action(best_rule.action)
        
        # Evaluate success
        old_situation = situation
        self.update_state()
        new_situation = self.get_current_situation()
        
        # Determine if action was successful
        success = self._evaluate_action_success(old_situation, new_situation, action_result)
        
        # Apply learning to rule
        best_rule.apply_outcome(success, self.steps_taken)
        
        # Record experience
        self.experience_history.append({
            'step': self.steps_taken,
            'situation': old_situation,
            'rule_used': best_rule.rule_id,
            'action': best_rule.action,
            'result': action_result,
            'success': success,
            'position': self.position
        })
        
        return {
            'alive': self.alive,
            'position': self.position,
            'hunger': self.hunger_level,
            'energy': self.energy_level,
            'active_rule': best_rule.rule_id,
            'action': best_rule.action,
            'success': success,
            'total_rules': len(self.rules)
        }
    
    def _evaluate_action_success(self, old_situation: SituationPattern, new_situation: SituationPattern, action_result: str) -> bool:
        """Evaluate if the action was successful"""
        if action_result in ["trapped", "no_food_available"]:
            return False
        
        # Success criteria based on action type
        if self.active_rule.rule_type == RuleType.SURVIVAL:
            # Survival success: reduced hunger, avoided danger
            if "food" in self.active_rule.action:
                return new_situation.hunger_level < old_situation.hunger_level or new_situation.food_distance < old_situation.food_distance
            elif "avoid" in self.active_rule.action:
                return new_situation.enemy_distance > old_situation.enemy_distance
        
        elif self.active_rule.rule_type == RuleType.GOAL:
            # Goal success: made progress toward goal
            return new_situation.goal_distance < old_situation.goal_distance
        
        elif self.active_rule.rule_type == RuleType.EXPLORATION:
            # Exploration success: moved to new position or reduced stagnation
            return self.position != self.last_position or new_situation.steps_since_progress < old_situation.steps_since_progress
        
        return True  # Default to success

# Test functions in next file due to length...