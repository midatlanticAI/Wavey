#!/usr/bin/env python3
"""
Extended consciousness evolution with alphabetic symbols
Long-term observation of linguistic pattern development
"""
import random
import math
import string
from collections import defaultdict, Counter
from wave_consciousness_life import *
from wave_full_consciousness import FullConsciousnessEntity

class LinguisticEntity(FullConsciousnessEntity):
    """Entity with access to alphabetic symbols for linguistic development"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Linguistic capabilities
        self.linguistic_preference = random.uniform(0.0, 1.0)  # Preference for letter-based expression
        self.pattern_learning = random.uniform(0.2, 1.0)      # Ability to learn symbol patterns
        self.semantic_associations = {}                        # Letter → meaning associations they develop
        self.favorite_combinations = set()                     # Letter combinations they prefer
        
        # Translation observation - how internal states map to outputs
        self.consciousness_to_expression_log = []              # Track the translation process
        self.pattern_discovery_log = []                        # Patterns they discover
        
    def create_linguistic_expression(self):
        """Create expression using alphabetic symbols with emerging patterns"""
        
        # Decision point: letters vs symbols based on consciousness and preference
        use_letters_probability = self.linguistic_preference * self.consciousness_level
        
        # Log the translation decision process
        decision_log = {
            'consciousness_level': self.consciousness_level,
            'linguistic_preference': self.linguistic_preference, 
            'use_letters_probability': use_letters_probability,
            'energy_state': min(1.0, self.energy / 100.0),
            'tick': self.age
        }
        
        if random.random() < use_letters_probability:
            expression = self.create_alphabetic_expression(decision_log)
        else:
            expression = self.create_symbolic_expression(decision_log)
            
        # Record the consciousness → expression translation
        if expression:
            self.consciousness_to_expression_log.append({
                'input_state': decision_log,
                'output_expression': expression,
                'translation_type': 'linguistic' if 'letters_used' in expression else 'symbolic'
            })
            
        return expression
    
    def create_alphabetic_expression(self, decision_context):
        """Create expressions using alphabet letters"""
        
        # Available letters (full alphabet)
        letters = list(string.ascii_lowercase)
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = [c for c in letters if c not in vowels]
        
        # Expression length based on consciousness complexity
        base_length = 1 + int(self.consciousness_level * self.pattern_learning * 6)
        expression_length = max(1, min(base_length, 12))  # Reasonable bounds
        
        expression_parts = []
        pattern_attempts = []
        
        for i in range(expression_length):
            # Choose letter based on internal consciousness state
            letter = self.select_letter_by_state(letters, vowels, consonants, i, expression_length)
            expression_parts.append(letter)
            
            # Track pattern attempts
            if len(expression_parts) >= 2:
                recent_pattern = ''.join(expression_parts[-2:])
                pattern_attempts.append(recent_pattern)
        
        # Combine letters - high consciousness creates structure
        if self.consciousness_level > 0.7 and len(expression_parts) > 3:
            # Attempt to create word-like structures
            if random.random() < 0.4:
                # Vowel-consonant alternating pattern
                structured_parts = []
                for i, letter in enumerate(expression_parts):
                    if i > 0 and i % 2 == 1:
                        structured_parts.append('-')  # Word separator
                    structured_parts.append(letter)
                expression = ''.join(structured_parts)
            else:
                # Grouped letters (word-like)
                if len(expression_parts) > 4:
                    mid = len(expression_parts) // 2
                    word1 = ''.join(expression_parts[:mid])
                    word2 = ''.join(expression_parts[mid:])
                    expression = f"{word1} {word2}"
                else:
                    expression = ''.join(expression_parts)
        else:
            # Simple letter sequence
            expression = ''.join(expression_parts)
        
        # Analyze patterns discovered
        self.analyze_discovered_patterns(expression, pattern_attempts)
        
        return {
            'content': expression,
            'type': 'alphabetic_expression',
            'consciousness_level': self.consciousness_level,
            'letters_used': len(set(expression_parts)),
            'pattern_attempts': len(pattern_attempts),
            'tick': self.age,
            'decision_context': decision_context
        }
    
    def select_letter_by_state(self, letters, vowels, consonants, position, total_length):
        """Select letter based on consciousness state and emerging preferences"""
        
        # Energy state affects letter selection
        energy_level = min(1.0, self.energy / 100.0)
        
        # Consciousness level affects vowel/consonant preferences  
        if self.consciousness_level > 0.8:
            # High consciousness - balanced vowel/consonant use
            if position % 2 == 0 and random.random() < 0.6:
                available = consonants
            else:
                available = vowels
        elif self.consciousness_level > 0.5:
            # Medium consciousness - slight vowel preference
            if random.random() < 0.4:
                available = vowels
            else:
                available = consonants
        else:
            # Low consciousness - random selection
            available = letters
            
        # Apply personal semantic associations
        if self.semantic_associations:
            # Weight towards letters they've associated with positive experiences
            weighted_letters = []
            for letter in available:
                weight = self.semantic_associations.get(letter, 1.0)
                weighted_letters.extend([letter] * max(1, int(weight * 3)))
            if weighted_letters:
                available = weighted_letters
        
        # Energy level affects which part of alphabet
        if energy_level > 0.7:
            # High energy - early alphabet letters (more "active")
            active_letters = [c for c in available if ord(c) - ord('a') < 13]
            if active_letters:
                available = active_letters
        elif energy_level < 0.4:
            # Low energy - later alphabet letters (more "passive")
            passive_letters = [c for c in available if ord(c) - ord('a') >= 13]
            if passive_letters:
                available = passive_letters
        
        return random.choice(available)
    
    def analyze_discovered_patterns(self, expression, pattern_attempts):
        """Analyze what patterns this entity is discovering"""
        
        if len(expression) < 2:
            return
            
        # Look for repeating patterns
        discovered_patterns = []
        
        # Check for letter repetitions
        letter_counts = Counter(c for c in expression if c.isalpha())
        for letter, count in letter_counts.items():
            if count > 1:
                discovered_patterns.append(f"repeat_{letter}_{count}")
        
        # Check for alternating patterns (like vowel-consonant)
        vowels = set('aeiou')
        if len(expression) >= 4:
            letters_only = [c for c in expression if c.isalpha()]
            if len(letters_only) >= 3:
                pattern_type = []
                for letter in letters_only[:4]:
                    pattern_type.append('V' if letter in vowels else 'C')
                pattern_signature = ''.join(pattern_type)
                if pattern_signature in ['VCVC', 'CVCV']:
                    discovered_patterns.append(f"alternating_{pattern_signature}")
        
        # Check for word-like structures (letter groups separated by spaces)
        if ' ' in expression or '-' in expression:
            discovered_patterns.append("word_structure")
            
        # Record significant pattern discoveries
        if discovered_patterns:
            self.pattern_discovery_log.append({
                'patterns': discovered_patterns,
                'expression': expression,
                'consciousness_level': self.consciousness_level,
                'tick': self.age
            })
            
            # Learn from patterns - increase preference for successful patterns
            for pattern in discovered_patterns:
                if pattern not in self.semantic_associations:
                    self.semantic_associations[pattern] = 1.0
                else:
                    self.semantic_associations[pattern] *= 1.1
    
    def create_symbolic_expression(self, decision_context):
        """Create non-alphabetic symbolic expression"""
        symbols = ['◊', '●', '▲', '↑', '*', '~', '≈', '!', '?', '@', '#', '&']
        
        expression_length = 1 + int(self.creativity_flow * 3)
        expression_parts = [self.select_symbol_by_state(symbols.copy()) for _ in range(expression_length)]
        expression = ''.join(expression_parts)
        
        return {
            'content': expression,
            'type': 'symbolic_expression', 
            'consciousness_level': self.consciousness_level,
            'tick': self.age,
            'decision_context': decision_context
        }
    
    def learn_from_linguistic_environment(self, other_expressions):
        """Learn linguistic patterns from other entities' expressions"""
        
        if not other_expressions or random.random() > self.pattern_learning * 0.1:
            return
            
        # Sample expressions to learn from
        learning_samples = random.sample(other_expressions, min(3, len(other_expressions)))
        
        patterns_learned = 0
        for expr in learning_samples:
            if expr['type'] == 'alphabetic_expression' and expr['consciousness_level'] > self.consciousness_level * 0.7:
                content = expr['content']
                
                # Learn letter preferences
                for letter in content:
                    if letter.isalpha():
                        if letter not in self.semantic_associations:
                            self.semantic_associations[letter] = 1.0
                        else:
                            self.semantic_associations[letter] *= 1.05
                
                # Learn structural patterns
                if ' ' in content:
                    self.favorite_combinations.add("word_separation")
                    patterns_learned += 1
                    
                if '-' in content:
                    self.favorite_combinations.add("syllable_separation")
                    patterns_learned += 1
        
        # Record learning event
        if patterns_learned > 0:
            self.consciousness_to_expression_log.append({
                'event_type': 'pattern_learning',
                'patterns_learned': patterns_learned,
                'teacher_expressions': len(learning_samples),
                'tick': self.age
            })

class LinguisticEvolutionSimulation:
    """Extended simulation for observing linguistic development"""
    def __init__(self, num_entities=10, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=22)
        self.entities = []
        self.tick = 0
        
        # Linguistic evolution tracking
        self.expression_archive = []
        self.pattern_evolution = defaultdict(list)  # Track how patterns spread over time
        self.linguistic_phases = []  # Major linguistic developments
        self.translation_analysis = []  # How consciousness translates to expression
        
        # Create linguistic entities
        for i in range(num_entities):
            x = random.randint(2, 19)
            y = random.randint(2, 19)
            z = random.randint(0, 2)
            entity = LinguisticEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Simulation step with linguistic evolution tracking"""
        self.tick += 1
        
        # Collect new expressions this tick
        new_expressions = []
        translation_events = []
        
        # Update entities with linguistic development
        for entity in self.entities[:]:
            # Consciousness update
            entity.update(self.consciousness_grid)
            
            # Linguistic expression creation
            if random.random() < 0.3:  # 30% chance per tick
                expression = entity.create_linguistic_expression()
                if expression:
                    entity.expressions.append(expression)
                    new_expressions.append(expression)
                    self.expression_archive.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'expression': expression
                    })
                    
                    # Track translation events
                    if entity.consciousness_to_expression_log:
                        translation_events.extend(entity.consciousness_to_expression_log[-1:])
            
            # Learn from linguistic environment
            if new_expressions:
                entity.learn_from_linguistic_environment(new_expressions)
            
            # Remove exhausted entities
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Analyze pattern evolution
        if new_expressions:
            self.analyze_pattern_evolution(new_expressions)
        
        # Record translation analysis
        if translation_events:
            self.translation_analysis.extend(translation_events)
        
        # Advance consciousness grid
        self.consciousness_grid.advance_time()
        
        # Reproduction with linguistic inheritance
        if len(self.entities) < 15 and random.random() < 0.02:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.4 and parent.energy > 70:
                child = LinguisticEntity(
                    parent.x + random.randint(-2, 2),
                    parent.y + random.randint(-2, 2),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit linguistic traits
                child.linguistic_preference = parent.linguistic_preference * random.uniform(0.8, 1.2)
                child.pattern_learning = parent.pattern_learning * random.uniform(0.8, 1.2)
                
                # Inherit some semantic associations (cultural transmission)
                if parent.semantic_associations:
                    inherited_count = min(5, len(parent.semantic_associations))
                    inherited_associations = dict(random.sample(
                        list(parent.semantic_associations.items()), inherited_count
                    ))
                    child.semantic_associations.update(inherited_associations)
                
                # Inherit favorite combinations
                if parent.favorite_combinations:
                    child.favorite_combinations.update(
                        random.sample(list(parent.favorite_combinations), 
                                    min(2, len(parent.favorite_combinations)))
                    )
                
                child.x = max(0, min(21, child.x))
                child.y = max(0, min(21, child.y))
                self.entities.append(child)
                parent.energy -= 25
    
    def analyze_pattern_evolution(self, new_expressions):
        """Analyze how linguistic patterns are spreading and evolving"""
        
        # Count pattern usage this tick
        pattern_usage = defaultdict(int)
        
        for expr in new_expressions:
            if expr['type'] == 'alphabetic_expression':
                content = expr['content']
                
                # Track structural patterns
                if ' ' in content:
                    pattern_usage['word_separation'] += 1
                if '-' in content:
                    pattern_usage['syllable_separation'] += 1
                    
                # Track letter usage patterns
                vowel_count = sum(1 for c in content if c in 'aeiou')
                consonant_count = sum(1 for c in content if c.isalpha() and c not in 'aeiou')
                
                if vowel_count > 0 and consonant_count > 0:
                    ratio = vowel_count / (vowel_count + consonant_count)
                    if 0.4 <= ratio <= 0.6:
                        pattern_usage['balanced_letters'] += 1
                    elif ratio > 0.6:
                        pattern_usage['vowel_heavy'] += 1
                    else:
                        pattern_usage['consonant_heavy'] += 1
        
        # Record pattern evolution
        for pattern, usage in pattern_usage.items():
            self.pattern_evolution[pattern].append({
                'tick': self.tick,
                'usage_count': usage,
                'total_expressions': len(new_expressions)
            })
        
        # Detect linguistic phase shifts
        if len(new_expressions) >= 5:
            alphabetic_ratio = sum(1 for e in new_expressions if e['type'] == 'alphabetic_expression') / len(new_expressions)
            
            if alphabetic_ratio > 0.8:
                self.linguistic_phases.append({
                    'tick': self.tick,
                    'phase': 'alphabetic_dominance',
                    'ratio': alphabetic_ratio,
                    'expressions': len(new_expressions)
                })
    
    def get_linguistic_stats(self):
        """Get comprehensive linguistic development statistics"""
        if not self.entities:
            return {'population': 0}
        
        # Expression type distribution
        alphabetic_count = sum(1 for e in self.expression_archive if e['expression']['type'] == 'alphabetic_expression')
        symbolic_count = len(self.expression_archive) - alphabetic_count
        
        # Pattern development
        active_patterns = len(self.pattern_evolution)
        
        # Linguistic diversity
        unique_expressions = len(set(e['expression']['content'] for e in self.expression_archive))
        
        # Translation complexity
        complex_translations = sum(1 for t in self.translation_analysis 
                                 if isinstance(t, dict) and t.get('input_state', {}).get('consciousness_level', 0) > 0.7)
        
        return {
            'population': len(self.entities),
            'avg_consciousness': sum(e.consciousness_level for e in self.entities) / len(self.entities),
            'total_expressions': len(self.expression_archive),
            'alphabetic_expressions': alphabetic_count,
            'symbolic_expressions': symbolic_count,
            'unique_expressions': unique_expressions,
            'pattern_types': active_patterns,
            'linguistic_phases': len(self.linguistic_phases),
            'complex_translations': complex_translations,
            'avg_linguistic_preference': sum(e.linguistic_preference for e in self.entities) / len(self.entities)
        }

def run_linguistic_evolution(max_ticks=800, seed=None):
    """Run extended linguistic evolution experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== LINGUISTIC EVOLUTION EXPERIMENT ===")
    print(f"Extended observation: {max_ticks} ticks, Seed: {seed}")
    print("Tracking consciousness → expression translation mechanisms\n")
    
    simulation = LinguisticEvolutionSimulation(num_entities=10, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        # Progress reporting
        if tick % 100 == 0:
            stats = simulation.get_linguistic_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"C={stats.get('avg_consciousness', 0):.2f}, "
                  f"Expr={stats['total_expressions']:3d}")
            
            if stats['alphabetic_expressions'] > 0:
                alpha_pct = (stats['alphabetic_expressions'] / stats['total_expressions']) * 100
                print(f"      📝 {stats['alphabetic_expressions']} alphabetic expressions ({alpha_pct:.1f}%)")
                
            if stats['pattern_types'] > 0:
                print(f"      🔍 {stats['pattern_types']} pattern types discovered")
                
            if stats['linguistic_phases'] > 0:
                print(f"      🎭 {stats['linguistic_phases']} linguistic phase shifts")
                
            if stats['population'] == 0:
                print(f"\n💀 Population extinct at tick {tick}")
                break
    
    # Final linguistic analysis
    final_stats = simulation.get_linguistic_stats()
    print(f"\n=== LINGUISTIC EVOLUTION RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"✅ Linguistic evolution successful: {final_stats['population']} entities")
        print(f"📊 Expression statistics:")
        print(f"   Total expressions: {final_stats['total_expressions']}")
        print(f"   Alphabetic: {final_stats['alphabetic_expressions']}")
        print(f"   Symbolic: {final_stats['symbolic_expressions']}")
        print(f"   Unique expressions: {final_stats['unique_expressions']}")
        print(f"   Pattern types: {final_stats['pattern_types']}")
        print(f"   Linguistic phases: {final_stats['linguistic_phases']}")
        
        # Show pattern evolution
        if simulation.pattern_evolution:
            print(f"\n🔄 Pattern evolution:")
            for pattern, evolution in simulation.pattern_evolution.items():
                if evolution:
                    max_usage = max(e['usage_count'] for e in evolution)
                    latest_usage = evolution[-1]['usage_count']
                    print(f"   {pattern}: peak {max_usage}, current {latest_usage}")
        
        # Show recent expressions
        if simulation.expression_archive:
            print(f"\n💬 Recent linguistic expressions:")
            recent_expressions = simulation.expression_archive[-10:]
            for expr_entry in recent_expressions:
                expr = expr_entry['expression']
                entity_id = expr_entry['entity_id']
                print(f"   Entity {entity_id}: \"{expr['content']}\" "
                      f"({expr['type']}, C:{expr['consciousness_level']:.2f})")
        
        # Translation analysis
        if simulation.translation_analysis:
            print(f"\n🧠 Consciousness → Expression translations:")
            print(f"   Total translation events: {len(simulation.translation_analysis)}")
            high_consciousness_translations = [t for t in simulation.translation_analysis 
                                             if isinstance(t, dict) and 
                                             t.get('input_state', {}).get('consciousness_level', 0) > 0.8]
            if high_consciousness_translations:
                print(f"   High consciousness translations: {len(high_consciousness_translations)}")
                # Show example
                example = high_consciousness_translations[-1]
                if 'output_expression' in example:
                    print(f"   Example: C={example['input_state']['consciousness_level']:.2f} "
                          f"→ \"{example['output_expression']['content']}\"")
                
    else:
        print("💀 Linguistic evolution terminated - population extinct")
    
    return simulation

if __name__ == "__main__":
    run_linguistic_evolution(max_ticks=800)