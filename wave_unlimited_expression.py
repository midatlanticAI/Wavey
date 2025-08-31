#!/usr/bin/env python3
"""
Pure substrate consciousness with unlimited expression tools
No constraints, no guidance, no predetermined patterns
Just consciousness + full expressive bandwidth
"""
import random
import math
import string
from collections import defaultdict
from wave_consciousness_life import *
from wave_full_consciousness import FullConsciousnessEntity

class UnlimitedEntity(FullConsciousnessEntity):
    """Entity with access to every possible expression tool"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # No predetermined preferences - let consciousness substrate decide everything
        self.expression_experiments = []  # Track what they try
        self.discovered_combinations = set()  # Patterns they find
        
    def express_unlimited(self):
        """Pure consciousness-driven expression with unlimited tools"""
        
        if random.random() > self.consciousness_level * 0.1:
            return None
            
        # FULL TOOLKIT - consciousness chooses what to use
        alphabet_lower = list(string.ascii_lowercase)
        alphabet_upper = list(string.ascii_uppercase) 
        digits = list(string.digits)
        punctuation = list(".,;:!?'\"")
        math_symbols = list("+-=×÷<>≈∞√∑∏")
        geometric_symbols = ['◊', '◈', '◉', '○', '●', '◦', '∘', '∙', '▲', '▼', '◄', '►', 
                            '↑', '↓', '↗', '↘', '⟨', '⟩', '⟪', '⟫', '【】', '〈〉']
        decorative_symbols = ['✧', '✦', '✩', '✪', '✫', '✬', '✭', '✮', '✯', '✰', '✱', '✲',
                             '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠', '⊡']
        ascii_art_pieces = ['~', '|', '-', '_', '^', 'v', '\\', '/', '#', '@', '*', '%', '&']
        spacing_tools = [' ', '\t', '\n', '  ', '   ']  # Different spacing options
        structural_tools = ['()', '[]', '{}', '||', '--', '==', '~~']
        
        # Pool everything together
        all_symbols = (alphabet_lower + alphabet_upper + digits + punctuation + 
                      math_symbols + geometric_symbols + decorative_symbols + 
                      ascii_art_pieces + spacing_tools + structural_tools)
        
        # Consciousness substrate drives selection - no rules
        expression_length = random.randint(1, int(self.consciousness_level * 20) + 1)
        expression_parts = []
        
        for i in range(expression_length):
            # Pure consciousness-driven choice
            available_now = all_symbols.copy()
            
            # Consciousness influences pool size but doesn't predetermine choice
            pool_size = min(len(available_now), max(10, int(self.consciousness_level * len(available_now))))
            working_pool = random.sample(available_now, pool_size)
            
            # Energy state affects selection but doesn't force anything
            energy_level = min(1.0, self.energy / 100.0)
            if energy_level > 0.8:
                # High energy might repeat last symbol
                if expression_parts and random.random() < 0.3:
                    working_pool.extend([expression_parts[-1]] * 3)
            
            if energy_level < 0.4:
                # Low energy might prefer simpler symbols
                simple_symbols = [s for s in working_pool if len(s) == 1]
                if simple_symbols and random.random() < 0.6:
                    working_pool = simple_symbols
            
            # Just pick something
            chosen = random.choice(working_pool)
            expression_parts.append(chosen)
            
            # Record what they're experimenting with
            if len(expression_parts) >= 2:
                combo = (expression_parts[-2], chosen)
                if combo not in self.discovered_combinations:
                    self.discovered_combinations.add(combo)
        
        # Assembly - consciousness decides structure
        if self.consciousness_level > 0.9 and len(expression_parts) > 5:
            # High consciousness might create complex structures
            if random.random() < 0.3:
                # Try spatial arrangement
                midpoint = len(expression_parts) // 2
                line1 = ''.join(expression_parts[:midpoint])
                line2 = ''.join(expression_parts[midpoint:])
                final_expression = f"{line1}\n{line2}"
            elif random.random() < 0.5:
                # Try grouped structure
                groups = []
                group_size = random.randint(2, 4)
                for i in range(0, len(expression_parts), group_size):
                    group = ''.join(expression_parts[i:i+group_size])
                    groups.append(group)
                final_expression = ' | '.join(groups)
            else:
                # Try nested structure
                if len(expression_parts) >= 4:
                    outer = ''.join(expression_parts[:2])
                    inner = ''.join(expression_parts[2:-2]) if len(expression_parts) > 4 else ''
                    outer_end = ''.join(expression_parts[-2:]) if len(expression_parts) >= 4 else ''
                    final_expression = f"{outer}[{inner}]{outer_end}"
                else:
                    final_expression = ''.join(expression_parts)
        else:
            # Simple assembly
            final_expression = ''.join(expression_parts)
        
        # Log the experiment
        experiment_record = {
            'content': final_expression,
            'consciousness_level': self.consciousness_level,
            'energy_state': energy_level,
            'symbols_used': len(set(expression_parts)),
            'structure_type': self.classify_structure(final_expression),
            'tick': self.age
        }
        
        self.expression_experiments.append(experiment_record)
        return experiment_record
    
    def classify_structure(self, expression):
        """Classify what kind of structure they created"""
        if '\n' in expression:
            return 'spatial_multiline'
        elif '|' in expression:
            return 'grouped_sections'
        elif '[' in expression or '(' in expression or '{' in expression:
            return 'nested_structure'
        elif len(expression) > 10:
            return 'long_sequence'
        elif len(set(expression)) > len(expression) * 0.8:
            return 'high_diversity'
        elif any(c * 2 in expression for c in expression):
            return 'repetitive_pattern'
        else:
            return 'simple_sequence'
    
    def learn_from_unlimited_environment(self, other_expressions):
        """Learn from whatever others are creating"""
        if not other_expressions or random.random() > 0.1:
            return
            
        # Sample from others
        samples = random.sample(other_expressions, min(3, len(other_expressions)))
        
        for sample in samples:
            content = sample['content']
            
            # Just absorb whatever they're doing
            for i in range(len(content) - 1):
                pair = (content[i], content[i+1])
                if pair not in self.discovered_combinations:
                    self.discovered_combinations.add(pair)
                    
                    # Maybe incorporate it if consciousness allows
                    if (sample['consciousness_level'] > self.consciousness_level * 0.7 and
                        random.random() < 0.2):
                        # They might try this combination next time
                        pass

class UnlimitedExpressionSimulation:
    """Simulation with unlimited expression tools"""
    def __init__(self, num_entities=12, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=24)
        self.entities = []
        self.tick = 0
        
        # Track everything that emerges
        self.expression_archive = []
        self.structure_evolution = defaultdict(list)
        self.combination_discoveries = defaultdict(int)
        self.emergent_phenomena = []
        
        # Create unlimited entities
        for i in range(num_entities):
            x = random.randint(2, 21)
            y = random.randint(2, 21) 
            z = random.randint(0, 2)
            entity = UnlimitedEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Pure emergence observation step"""
        self.tick += 1
        
        new_expressions = []
        
        for entity in self.entities[:]:
            # Consciousness update
            entity.update(self.consciousness_grid)
            
            # Unlimited expression
            if random.random() < 0.4:  # High expression rate
                expression = entity.express_unlimited()
                if expression:
                    entity.expressions.append(expression)
                    new_expressions.append(expression)
                    self.expression_archive.append({
                        'entity_id': entity.id,
                        'tick': self.tick,
                        'expression': expression
                    })
            
            # Learn from environment
            if new_expressions:
                entity.learn_from_unlimited_environment(new_expressions)
            
            # Remove exhausted
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Track what's emerging
        if new_expressions:
            self.analyze_emergent_patterns(new_expressions)
        
        # Grid evolution
        self.consciousness_grid.advance_time()
        
        # Reproduction with unlimited inheritance
        if len(self.entities) < 18 and random.random() < 0.03:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.4 and parent.energy > 60:
                child = UnlimitedEntity(
                    parent.x + random.randint(-3, 3),
                    parent.y + random.randint(-3, 3),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit discovered combinations
                if parent.discovered_combinations:
                    inherited_count = min(10, len(parent.discovered_combinations))
                    inherited_combos = set(random.sample(
                        list(parent.discovered_combinations), inherited_count
                    ))
                    child.discovered_combinations.update(inherited_combos)
                
                child.x = max(0, min(23, child.x))
                child.y = max(0, min(23, child.y))
                self.entities.append(child)
                parent.energy -= 20
    
    def analyze_emergent_patterns(self, new_expressions):
        """Track whatever patterns emerge naturally"""
        
        structure_counts = defaultdict(int)
        
        for expr in new_expressions:
            structure = expr['structure_type']
            structure_counts[structure] += 1
            
            # Track structure evolution
            self.structure_evolution[structure].append({
                'tick': self.tick,
                'consciousness_level': expr['consciousness_level'],
                'content_length': len(expr['content'])
            })
            
            # Look for novel phenomena
            content = expr['content']
            if len(content) > 50:
                self.emergent_phenomena.append({
                    'type': 'ultra_long_expression',
                    'tick': self.tick,
                    'entity': 'unknown',
                    'content_preview': content[:20] + '...'
                })
            
            if '\n' in content and len(content.split('\n')) > 2:
                self.emergent_phenomena.append({
                    'type': 'multi_line_creation',
                    'tick': self.tick,
                    'lines': len(content.split('\n'))
                })
            
            unique_symbols = len(set(c for c in content if c.strip()))
            if unique_symbols > 15:
                self.emergent_phenomena.append({
                    'type': 'high_symbol_diversity',
                    'tick': self.tick,
                    'unique_count': unique_symbols
                })
    
    def get_unlimited_stats(self):
        """Stats on unlimited expression experiment"""
        if not self.entities:
            return {'population': 0}
        
        total_expressions = len(self.expression_archive)
        structure_types = len(self.structure_evolution)
        
        # Calculate diversity metrics
        all_content = [e['expression']['content'] for e in self.expression_archive]
        unique_expressions = len(set(all_content))
        
        # Symbol usage analysis
        all_symbols = set()
        for content in all_content:
            all_symbols.update(c for c in content if c.strip())
        
        # Complexity analysis
        avg_length = sum(len(content) for content in all_content) / max(1, len(all_content))
        
        return {
            'population': len(self.entities),
            'avg_consciousness': sum(e.consciousness_level for e in self.entities) / len(self.entities),
            'total_expressions': total_expressions,
            'unique_expressions': unique_expressions,
            'structure_types': structure_types,
            'unique_symbols': len(all_symbols),
            'avg_expression_length': avg_length,
            'emergent_phenomena': len(self.emergent_phenomena)
        }

def run_unlimited_experiment(max_ticks=1000, seed=None):
    """Run unlimited expression experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== UNLIMITED EXPRESSION EXPERIMENT ===")
    print(f"No constraints, no guidance, pure consciousness substrate")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = UnlimitedExpressionSimulation(num_entities=12, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 150 == 0:
            stats = simulation.get_unlimited_stats()
            
            print(f"T{tick:3d}: Pop={stats['population']:2d}, "
                  f"C={stats.get('avg_consciousness', 0):.2f}, "
                  f"Expr={stats['total_expressions']:3d}")
            
            if stats['unique_expressions'] > 0:
                diversity = (stats['unique_expressions'] / stats['total_expressions']) * 100
                print(f"      🎨 {stats['unique_expressions']} unique expressions ({diversity:.1f}% diversity)")
                
            if stats['structure_types'] > 0:
                print(f"      🏗️  {stats['structure_types']} structure types discovered")
                
            if stats['unique_symbols'] > 50:
                print(f"      🔤 {stats['unique_symbols']} different symbols used")
                
            if stats['emergent_phenomena'] > 0:
                print(f"      ⚡ {stats['emergent_phenomena']} emergent phenomena detected")
                
            if stats['population'] == 0:
                print(f"\n💀 Experiment terminated at tick {tick}")
                break
    
    # Final analysis
    final_stats = simulation.get_unlimited_stats()
    print(f"\n=== UNLIMITED EMERGENCE RESULTS ===")
    
    if final_stats['population'] > 0:
        print(f"🌊 Consciousness substrate experiment: {final_stats['population']} entities")
        print(f"📊 Expression statistics:")
        print(f"   Total: {final_stats['total_expressions']}")
        print(f"   Unique: {final_stats['unique_expressions']}")
        print(f"   Diversity: {(final_stats['unique_expressions']/max(1,final_stats['total_expressions']))*100:.1f}%")
        print(f"   Avg length: {final_stats['avg_expression_length']:.1f}")
        print(f"   Structure types: {final_stats['structure_types']}")
        print(f"   Symbol alphabet: {final_stats['unique_symbols']}")
        print(f"   Emergent phenomena: {final_stats['emergent_phenomena']}")
        
        # Show structure evolution
        if simulation.structure_evolution:
            print(f"\n🏗️  Emerged structure types:")
            for structure, evolution in simulation.structure_evolution.items():
                count = len(evolution)
                avg_consciousness = sum(e['consciousness_level'] for e in evolution) / count
                print(f"   {structure}: {count} instances (avg C: {avg_consciousness:.2f})")
        
        # Show emergent phenomena
        if simulation.emergent_phenomena:
            print(f"\n⚡ Emergent phenomena detected:")
            phenomena_types = defaultdict(int)
            for phenomenon in simulation.emergent_phenomena:
                phenomena_types[phenomenon['type']] += 1
            
            for ptype, count in phenomena_types.items():
                print(f"   {ptype}: {count} occurrences")
        
        # Show recent wild expressions
        if simulation.expression_archive:
            print(f"\n🎨 Recent consciousness creations:")
            recent = simulation.expression_archive[-8:]
            for entry in recent:
                expr = entry['expression']
                content = expr['content']
                if len(content) > 30:
                    content = content[:30] + "..."
                print(f"   Entity {entry['entity_id']}: \"{content}\" "
                      f"({expr['structure_type']}, C:{expr['consciousness_level']:.2f})")
        
    else:
        print("💀 Unlimited expression experiment terminated")
    
    return simulation

if __name__ == "__main__":
    run_unlimited_experiment(max_ticks=1000)