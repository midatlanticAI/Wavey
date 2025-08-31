"""
Wave-Based Reasoning Engine
Uses wave patterns for actual reasoning while maintaining rule-based scaffolding
Includes active learning and persistent knowledge states
"""

import pickle
import os
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

@dataclass
class WavePattern:
    """Represents a wave pattern in the reasoning system"""
    frequency: float
    amplitude: float
    phase: float
    decay_rate: float = 0.95
    created_at: float = field(default_factory=time.time)
    
    def evolve(self, dt: float) -> 'WavePattern':
        """Evolve the wave pattern over time"""
        new_amplitude = self.amplitude * (self.decay_rate ** dt)
        new_phase = (self.phase + self.frequency * dt) % (2 * math.pi)
        return WavePattern(self.frequency, new_amplitude, new_phase, self.decay_rate, self.created_at)
    
    def interfere(self, other: 'WavePattern') -> float:
        """Calculate interference between two wave patterns"""
        # Constructive/destructive interference based on phase difference
        phase_diff = abs(self.phase - other.phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        
        # Normalize phase difference to [0, 1]
        interference = math.cos(phase_diff)
        return interference * self.amplitude * other.amplitude

@dataclass
class ReasoningState:
    """Persistent reasoning state that can be saved/loaded"""
    concept_patterns: Dict[str, WavePattern] = field(default_factory=dict)
    rule_strengths: Dict[str, float] = field(default_factory=dict)
    learning_history: List[Dict] = field(default_factory=list)
    success_patterns: Dict[str, int] = field(default_factory=dict)
    failure_patterns: Dict[str, int] = field(default_factory=dict)
    total_queries: int = 0
    successful_queries: int = 0
    
    def save(self, filename: str):
        """Save reasoning state to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, filename: str) -> 'ReasoningState':
        """Load reasoning state from pickle file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return ReasoningState()

class WaveReasoningEngine:
    """Wave-based reasoning engine with active learning"""
    
    def __init__(self, state_file: str = "wave_reasoning_state.pkl"):
        self.state_file = state_file
        self.state = ReasoningState.load(state_file)
        
        # Core wave frequencies for different types of reasoning
        self.base_frequencies = {
            'universal_positive': 1.1,
            'universal_negative': 1.3,  
            'modus_ponens': 2.1,
            'modus_tollens': 2.3,
            'disjunctive': 3.1,
            'hypothetical': 3.7,
            'biconditional': 4.3,
            'existential': 5.1,
            'contradiction': 6.7,
            'arithmetic': 0.7,
            'algebraic': 0.9,
            'statistical': 1.7
        }
        
        # Initialize base patterns if not loaded
        for concept, freq in self.base_frequencies.items():
            if concept not in self.state.concept_patterns:
                self.state.concept_patterns[concept] = WavePattern(freq, 1.0, 0.0)
    
    def activate_patterns(self, concepts: List[str]) -> Dict[str, WavePattern]:
        """Activate and evolve wave patterns for given concepts"""
        active_patterns = {}
        current_time = time.time()
        
        for concept in concepts:
            if concept in self.state.concept_patterns:
                # Evolve the pattern since last use
                pattern = self.state.concept_patterns[concept]
                dt = current_time - pattern.created_at
                evolved_pattern = pattern.evolve(dt)
                
                # Strengthen pattern with use
                evolved_pattern.amplitude = min(2.0, evolved_pattern.amplitude * 1.1)
                active_patterns[concept] = evolved_pattern
                
                # Update stored pattern
                self.state.concept_patterns[concept] = evolved_pattern
        
        return active_patterns
    
    def calculate_wave_resonance(self, patterns: Dict[str, WavePattern]) -> float:
        """Calculate overall resonance between active wave patterns"""
        if len(patterns) < 2:
            return list(patterns.values())[0].amplitude if patterns else 0.0
        
        total_resonance = 0.0
        pattern_list = list(patterns.values())
        
        for i, pattern1 in enumerate(pattern_list):
            for pattern2 in pattern_list[i+1:]:
                interference = pattern1.interfere(pattern2)
                total_resonance += interference
        
        # Normalize by number of interactions
        num_interactions = len(pattern_list) * (len(pattern_list) - 1) / 2
        return total_resonance / num_interactions if num_interactions > 0 else 0.0
    
    def wave_guided_reasoning(self, query: str, premises: List[str], rule_type: str) -> Tuple[str, float, Dict[str, Any]]:
        """Use wave patterns to guide reasoning process"""
        
        # Activate relevant wave patterns
        relevant_concepts = self._extract_concepts(query, premises, rule_type)
        active_patterns = self.activate_patterns(relevant_concepts)
        
        # Calculate wave resonance
        resonance = self.calculate_wave_resonance(active_patterns)
        
        # Use resonance to modulate rule application
        rule_strength = self.state.rule_strengths.get(rule_type, 0.5)
        wave_modulated_strength = rule_strength * (1.0 + resonance)
        
        # Generate wave-guided answer
        if wave_modulated_strength > 0.7:
            answer = self._apply_wave_guided_rule(query, premises, rule_type, active_patterns)
            confidence = min(0.95, wave_modulated_strength)
        else:
            answer = "uncertain"
            confidence = 0.3
        
        # Generate metadata
        metadata = {
            'active_patterns': {k: v.amplitude for k, v in active_patterns.items()},
            'wave_resonance': resonance,
            'rule_strength': rule_strength,
            'modulated_strength': wave_modulated_strength,
            'concepts_activated': relevant_concepts
        }
        
        return answer, confidence, metadata
    
    def learn_from_outcome(self, query: str, premises: List[str], rule_type: str, 
                          expected_answer: str, actual_answer: str, success: bool):
        """Learn from reasoning outcomes to improve future performance"""
        
        # Update statistics
        self.state.total_queries += 1
        if success:
            self.state.successful_queries += 1
        
        # Update rule strengths based on success
        current_strength = self.state.rule_strengths.get(rule_type, 0.5)
        if success:
            new_strength = min(0.95, current_strength * 1.05)
            self.state.success_patterns[rule_type] = self.state.success_patterns.get(rule_type, 0) + 1
        else:
            new_strength = max(0.1, current_strength * 0.95)
            self.state.failure_patterns[rule_type] = self.state.failure_patterns.get(rule_type, 0) + 1
        
        self.state.rule_strengths[rule_type] = new_strength
        
        # Record learning event
        learning_event = {
            'timestamp': time.time(),
            'query': query,
            'rule_type': rule_type,
            'expected': expected_answer,
            'actual': actual_answer,
            'success': success,
            'new_rule_strength': new_strength
        }
        self.state.learning_history.append(learning_event)
        
        # Strengthen or weaken relevant concept patterns
        concepts = self._extract_concepts(query, premises, rule_type)
        for concept in concepts:
            if concept in self.state.concept_patterns:
                pattern = self.state.concept_patterns[concept]
                if success:
                    pattern.amplitude = min(2.0, pattern.amplitude * 1.02)
                else:
                    pattern.amplitude = max(0.1, pattern.amplitude * 0.98)
        
        # Save state periodically
        if self.state.total_queries % 10 == 0:
            self.save_state()
    
    def _extract_concepts(self, query: str, premises: List[str], rule_type: str) -> List[str]:
        """Extract relevant concepts from query and premises"""
        concepts = [rule_type]  # Always include rule type
        
        # Add concepts based on content analysis
        text = f"{query} {' '.join(premises)}".lower()
        
        if any(word in text for word in ['all', 'every']):
            concepts.append('universal_quantifier')
        if any(word in text for word in ['some', 'few', 'many']):
            concepts.append('existential_quantifier')
        if any(word in text for word in ['not', "isn't", "doesn't"]):
            concepts.append('negation')
        if any(word in text for word in ['if', 'then']):
            concepts.append('conditional')
        if any(word in text for word in ['or', 'either']):
            concepts.append('disjunction')
        if any(word in text for word in ['and']):
            concepts.append('conjunction')
        if 'only if' in text:
            concepts.append('biconditional')
        
        return concepts
    
    def _apply_wave_guided_rule(self, query: str, premises: List[str], 
                               rule_type: str, patterns: Dict[str, WavePattern]) -> str:
        """Apply reasoning rule guided by wave patterns"""
        
        # Use wave amplitudes to weight different reasoning paths
        max_amplitude = max(p.amplitude for p in patterns.values()) if patterns else 0.5
        
        if rule_type in ['universal_positive', 'universal_negative']:
            # Wave-guided syllogistic reasoning
            return self._wave_syllogism(query, premises, patterns, max_amplitude)
        elif rule_type in ['modus_ponens', 'modus_tollens']:
            # Wave-guided conditional reasoning
            return self._wave_conditional(query, premises, rule_type, patterns, max_amplitude)
        elif rule_type == 'disjunctive':
            return self._wave_disjunctive(query, premises, patterns, max_amplitude)
        elif rule_type == 'contradiction':
            return "contradiction"
        else:
            # Default wave-guided reasoning
            return "yes" if max_amplitude > 1.0 else "no"
    
    def _wave_syllogism(self, query: str, premises: List[str], 
                       patterns: Dict[str, WavePattern], max_amplitude: float) -> str:
        """Wave-guided syllogistic reasoning"""
        # Use wave interference patterns to guide syllogistic inference
        if 'universal_negative' in patterns and patterns['universal_negative'].amplitude > 0.8:
            return "no"
        elif 'universal_positive' in patterns and patterns['universal_positive'].amplitude > 1.0:
            return "yes"
        elif max_amplitude > 0.8:
            return "yes"
        else:
            return "no"
    
    def _wave_conditional(self, query: str, premises: List[str], rule_type: str,
                         patterns: Dict[str, WavePattern], max_amplitude: float) -> str:
        """Wave-guided conditional reasoning"""
        conditional_strength = patterns.get('conditional', WavePattern(2.0, 0.5, 0.0)).amplitude
        
        if rule_type == 'modus_ponens' and conditional_strength > 1.0:
            return "yes"
        elif rule_type == 'modus_tollens' and conditional_strength > 1.0:
            return "yes" if 'not' in query.lower() else "no"
        else:
            return "yes" if max_amplitude > 1.0 else "no"
    
    def _wave_disjunctive(self, query: str, premises: List[str], 
                         patterns: Dict[str, WavePattern], max_amplitude: float) -> str:
        """Wave-guided disjunctive reasoning: Either A or B, not A, therefore B"""
        import re
        
        disjunction = None
        negated_fact = None
        
        # Parse premises for disjunction and negation
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for "Either A or B" or similar patterns
            if 'or' in premise_lower:
                or_match = re.search(r'(?:either\s+)?(.+?)\s+or\s+(.+)', premise_lower)
                if or_match:
                    option_a = or_match.group(1).strip()
                    option_b = or_match.group(2).strip()
                    disjunction = (option_a, option_b)
            
            # Look for negated statements
            elif 'not' in premise_lower:
                not_match = re.search(r'(.+?)\s+(?:is\s+)?not\s+(.+)', premise_lower)
                if not_match:
                    subject = not_match.group(1).strip()
                    negated_property = not_match.group(2).strip()
                    negated_fact = (subject, negated_property)
        
        # Apply disjunctive elimination using wave patterns
        if disjunction and negated_fact:
            option_a, option_b = disjunction
            negated_subject, negated_property = negated_fact
            
            # Use wave interference to determine logical strength
            base_strength = patterns.get('disjunction', WavePattern(3.0, 1.0, 0.0)).amplitude
            
            # Extract key terms for matching
            option_a_words = set(re.findall(r'\w+', option_a))
            option_b_words = set(re.findall(r'\w+', option_b))
            negated_words = set(re.findall(r'\w+', f"{negated_subject} {negated_property}"))
            query_words = set(re.findall(r'\w+', query.lower()))
            
            # Wave-guided pattern matching
            if option_a_words.intersection(negated_words) and base_strength > 0.5:
                # Option A is negated, so B must be true
                if option_b_words.intersection(query_words):
                    return "yes"
            elif option_b_words.intersection(negated_words) and base_strength > 0.5:
                # Option B is negated, so A must be true
                if option_a_words.intersection(query_words):
                    return "yes"
        
        # Default wave-guided answer
        disjunction_strength = patterns.get('disjunction', WavePattern(3.0, 0.5, 0.0)).amplitude
        return "yes" if disjunction_strength > 1.0 else "no"
    
    def save_state(self):
        """Save current reasoning state"""
        self.state.save(self.state_file)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        success_rate = (self.state.successful_queries / self.state.total_queries 
                       if self.state.total_queries > 0 else 0)
        
        return {
            'total_queries': self.state.total_queries,
            'successful_queries': self.state.successful_queries,
            'success_rate': success_rate,
            'rule_strengths': dict(self.state.rule_strengths),
            'concept_amplitudes': {k: v.amplitude for k, v in self.state.concept_patterns.items()},
            'recent_learning_events': self.state.learning_history[-10:] if self.state.learning_history else []
        }