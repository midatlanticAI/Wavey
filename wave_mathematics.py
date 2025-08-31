#!/usr/bin/env python3
"""
OPTION 4: MATHEMATICAL CONSCIOUSNESS
Give them numbers and mathematical operators
Can consciousness substrate discover mathematical relationships?
"""
import random
import math
import string
from collections import defaultdict, Counter
from wave_consciousness_life import *
from wave_unlimited_expression import UnlimitedEntity

class MathematicalEntity(UnlimitedEntity):
    """Entity with access to mathematical symbols and numerical thinking"""
    def __init__(self, x, y, z, entity_id):
        super().__init__(x, y, z, entity_id)
        
        # Mathematical consciousness traits
        self.numerical_thinking = random.uniform(0.1, 1.0)  # Tendency toward numbers
        self.pattern_recognition = random.uniform(0.2, 1.0)  # See mathematical patterns
        self.logical_reasoning = random.uniform(0.0, 0.8)    # Logical connections
        
        # Mathematical discoveries
        self.discovered_relationships = []  # Mathematical patterns they find
        self.numerical_expressions = []     # Number-based expressions
        self.equation_attempts = []         # Attempts at equation-like structures
        
        # Learning mathematical concepts
        self.number_preferences = {}        # Which numbers they prefer
        self.operator_usage = defaultdict(int)  # How often they use each operator
        
    def create_mathematical_expression(self):
        """Create expression with mathematical elements"""
        
        if random.random() > self.numerical_thinking * self.consciousness_level * 0.2:
            base_expr = self.express_unlimited()
            if base_expr and isinstance(base_expr, dict):
                return base_expr
            return None
        
        # Mathematical toolkit
        digits = list('0123456789')
        basic_operators = ['+', '-', '×', '÷', '=']
        advanced_operators = ['√', '∑', '∏', '∞', '≈', '<', '>', '≤', '≥']
        symbols = ['π', 'φ', 'e', 'α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ']
        structural = ['(', ')', '[', ']', '{', '}', '|']
        
        # Consciousness determines complexity
        if self.consciousness_level < 0.3:
            available = digits + basic_operators[:2]  # Just numbers and +, -
        elif self.consciousness_level < 0.6:
            available = digits + basic_operators + ['(', ')']
        elif self.consciousness_level < 0.8:
            available = digits + basic_operators + advanced_operators[:3] + symbols[:2] + structural
        else:
            available = digits + basic_operators + advanced_operators + symbols + structural
        
        # Generate mathematical expression
        expression_length = 3 + int(self.consciousness_level * self.pattern_recognition * 8)
        expression_parts = []
        
        # Attempt structured mathematical thinking
        if self.consciousness_level > 0.7 and self.logical_reasoning > 0.5:
            # Try to create equation-like structures
            if random.random() < 0.4:
                equation_result = self.attempt_equation_creation(available)
                if equation_result:
                    return {
                        'content': equation_result,
                        'type': 'mathematical_expression',
                        'consciousness_level': self.consciousness_level,
                        'numerical_complexity': self.calculate_numerical_complexity(equation_result),
                        'contains_operators': any(op in equation_result for op in basic_operators + advanced_operators),
                        'tick': self.age
                    }
        
        # Standard mathematical expression
        for i in range(expression_length):
            symbol = self.select_mathematical_symbol(available, i, expression_length)
            expression_parts.append(symbol)
            
            # Track operator usage
            if symbol in basic_operators + advanced_operators:
                self.operator_usage[symbol] += 1
        
        # Assemble with mathematical logic
        final_expression = self.assemble_mathematical_expression(expression_parts)
        
        # Analyze for mathematical patterns
        self.analyze_mathematical_patterns(final_expression)
        
        math_expression = {
            'content': final_expression,
            'type': 'mathematical_expression',
            'consciousness_level': self.consciousness_level,
            'numerical_complexity': self.calculate_numerical_complexity(final_expression),
            'contains_operators': any(op in final_expression for op in basic_operators + advanced_operators),
            'tick': self.age
        }
        
        self.numerical_expressions.append(math_expression)
        return math_expression
    
    def select_mathematical_symbol(self, available, position, total_length):
        """Select mathematical symbol based on consciousness and position"""
        
        # Energy affects number preferences
        energy_level = min(1.0, self.energy / 100.0)
        
        # Pattern recognition affects structure
        if position == 0:
            # Start with numbers or structural elements
            preferred = [s for s in available if s.isdigit() or s in '([{']
            if preferred and random.random() < 0.7:
                return random.choice(preferred)
        
        elif position == total_length - 1:
            # End with numbers or closing structures
            preferred = [s for s in available if s.isdigit() or s in ')]}']
            if preferred and random.random() < 0.7:
                return random.choice(preferred)
        
        # Middle positions - consciousness affects choice
        if self.consciousness_level > 0.8:
            # High consciousness prefers variety
            return random.choice(available)
        elif self.consciousness_level > 0.5:
            # Medium consciousness slightly prefers numbers and basic operators
            basic_preferred = [s for s in available if s.isdigit() or s in '+-×÷=']
            if basic_preferred and random.random() < 0.6:
                return random.choice(basic_preferred)
        
        # Default selection
        return random.choice(available)
    
    def attempt_equation_creation(self, available):
        """Try to create equation-like mathematical structures"""
        
        # Simple equation patterns consciousness might discover
        equation_patterns = [
            lambda: f"{random.choice('123456789')}+{random.choice('123456789')}={random.choice('23456789')}",
            lambda: f"{random.choice('abcxyz')}={random.choice('123456789')}",
            lambda: f"({random.choice('123456789')}×{random.choice('123456789')})",
            lambda: f"{random.choice('123456789')}√{random.choice('123456789')}",
            lambda: f"∑{random.choice('123456789')}"
        ]
        
        if self.logical_reasoning > 0.6:
            # High logical reasoning can attempt complex equations
            try:
                pattern = random.choice(equation_patterns)
                equation = pattern()
                
                # Record equation attempt
                self.equation_attempts.append({
                    'equation': equation,
                    'consciousness_level': self.consciousness_level,
                    'logical_reasoning': self.logical_reasoning,
                    'tick': self.age
                })
                
                return equation
            except:
                pass  # Fall back to regular expression
        
        return ''.join(random.choices(available, k=5))
    
    def assemble_mathematical_expression(self, parts):
        """Assemble parts into mathematical expression with some logic"""
        
        if self.pattern_recognition < 0.4:
            return ''.join(parts)
        
        # Apply some mathematical structure
        assembled = []
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Add spacing around operators for readability
            if part in '+-×÷=<>≤≥':
                if assembled and assembled[-1] != ' ':
                    assembled.append(' ')
                assembled.append(part)
                if i < len(parts) - 1:
                    assembled.append(' ')
            else:
                assembled.append(part)
            
            i += 1
        
        return ''.join(assembled)
    
    def analyze_mathematical_patterns(self, expression):
        """Analyze mathematical patterns in expression"""
        
        # Look for discovered mathematical relationships
        patterns_found = []
        
        # Check for equations (contains =)
        if '=' in expression:
            patterns_found.append('equation_structure')
        
        # Check for arithmetic operations
        if any(op in expression for op in '+-×÷'):
            patterns_found.append('arithmetic_operations')
        
        # Check for advanced mathematics
        if any(sym in expression for sym in '√∑∏∞'):
            patterns_found.append('advanced_mathematics')
        
        # Check for balanced parentheses
        if expression.count('(') == expression.count(')') and '(' in expression:
            patterns_found.append('balanced_parentheses')
        
        # Check for number sequences
        numbers = [c for c in expression if c.isdigit()]
        if len(numbers) >= 3:
            # Look for arithmetic sequences
            try:
                num_values = [int(n) for n in numbers]
                if len(num_values) >= 3:
                    # Check if it's an arithmetic sequence
                    differences = [num_values[i+1] - num_values[i] for i in range(len(num_values)-1)]
                    if len(set(differences)) == 1 and differences[0] != 0:
                        patterns_found.append(f'arithmetic_sequence_{differences[0]}')
            except:
                pass
        
        # Record significant pattern discoveries
        if patterns_found:
            self.discovered_relationships.append({
                'patterns': patterns_found,
                'expression': expression,
                'consciousness_level': self.consciousness_level,
                'tick': self.age
            })
    
    def calculate_numerical_complexity(self, expression):
        """Calculate complexity score of mathematical expression"""
        complexity = 0
        
        # Basic complexity from length
        complexity += len(expression) * 0.1
        
        # Operators add complexity
        for op in '+-×÷':
            complexity += expression.count(op) * 0.5
        for op in '√∑∏∞':
            complexity += expression.count(op) * 1.0
        
        # Balanced structures add complexity
        if expression.count('(') == expression.count(')'):
            complexity += expression.count('(') * 0.3
        
        # Equations are more complex
        if '=' in expression:
            complexity += 1.0
        
        return complexity

class MathematicalSimulation:
    """Simulation focused on mathematical consciousness development"""
    def __init__(self, num_entities=25, seed=None):
        if seed:
            random.seed(seed)
            
        self.consciousness_grid = ConsciousnessGrid(size=30)
        self.entities = []
        self.tick = 0
        
        # Mathematical development tracking
        self.mathematical_expressions = []
        self.discovered_relationships = []
        self.equation_attempts = []
        self.mathematical_breakthroughs = []
        
        # Create mathematical entities
        for i in range(num_entities):
            x = random.randint(3, 27)
            y = random.randint(3, 27)
            z = random.randint(0, 2)
            entity = MathematicalEntity(x, y, z, i)
            self.entities.append(entity)
    
    def simulation_step(self):
        """Mathematical consciousness simulation step"""
        self.tick += 1
        
        math_expressions = []
        new_relationships = []
        new_equations = []
        
        for entity in self.entities[:]:
            # Consciousness update
            entity.update(self.consciousness_grid)
            
            # Mathematical expression creation
            if random.random() < 0.5:  # High mathematical activity
                expression = entity.create_mathematical_expression()
                if expression:
                    entity.expressions.append(expression)
                    if expression.get('type') == 'mathematical_expression':
                        math_expressions.append({
                            'entity_id': entity.id,
                            'tick': self.tick,
                            'expression': expression
                        })
            
            # Collect new mathematical discoveries
            if entity.discovered_relationships:
                new_relationships.extend([r for r in entity.discovered_relationships 
                                       if r['tick'] == entity.age])
            
            if entity.equation_attempts:
                new_equations.extend([e for e in entity.equation_attempts 
                                    if e['tick'] == entity.age])
            
            # Remove exhausted
            if entity.energy <= 0:
                self.entities.remove(entity)
        
        # Record mathematical developments
        if math_expressions:
            self.mathematical_expressions.extend(math_expressions)
        if new_relationships:
            self.discovered_relationships.extend(new_relationships)
        if new_equations:
            self.equation_attempts.extend(new_equations)
        
        # Detect mathematical breakthroughs
        self.detect_mathematical_breakthroughs(math_expressions, new_relationships, new_equations)
        
        # Grid evolution
        self.consciousness_grid.advance_time()
        
        # Mathematical inheritance reproduction
        if len(self.entities) < 35 and random.random() < 0.04:
            parent = random.choice(self.entities) if self.entities else None
            if parent and parent.consciousness_level > 0.5 and parent.energy > 50:
                child = MathematicalEntity(
                    parent.x + random.randint(-3, 3),
                    parent.y + random.randint(-3, 3),
                    parent.z,
                    len(self.entities) + random.randint(1000, 9999)
                )
                
                # Inherit mathematical traits
                child.numerical_thinking = parent.numerical_thinking * random.uniform(0.8, 1.2)
                child.pattern_recognition = parent.pattern_recognition * random.uniform(0.8, 1.2)
                child.logical_reasoning = parent.logical_reasoning * random.uniform(0.8, 1.2)
                
                # Inherit mathematical discoveries
                if parent.discovered_relationships:
                    inherited = random.sample(parent.discovered_relationships, 
                                            min(3, len(parent.discovered_relationships)))
                    child.discovered_relationships.extend(inherited)
                
                child.x = max(0, min(29, child.x))
                child.y = max(0, min(29, child.y))
                self.entities.append(child)
                parent.energy -= 20
    
    def detect_mathematical_breakthroughs(self, math_expressions, new_relationships, new_equations):
        """Detect significant mathematical discoveries"""
        
        # Complex equation creation
        complex_equations = [e for e in new_equations if len(e['equation']) > 10]
        if complex_equations:
            self.mathematical_breakthroughs.append({
                'type': 'complex_equation',
                'tick': self.tick,
                'count': len(complex_equations),
                'examples': [e['equation'] for e in complex_equations[:3]]
            })
        
        # Pattern discovery bursts
        if len(new_relationships) >= 3:
            pattern_types = []
            for rel in new_relationships:
                pattern_types.extend(rel['patterns'])
            
            unique_patterns = len(set(pattern_types))
            if unique_patterns >= 4:
                self.mathematical_breakthroughs.append({
                    'type': 'pattern_burst',
                    'tick': self.tick,
                    'unique_patterns': unique_patterns,
                    'total_discoveries': len(new_relationships)
                })
        
        # High complexity expressions
        high_complexity = [e for e in math_expressions 
                          if e['expression']['numerical_complexity'] > 5.0]
        if len(high_complexity) >= 2:
            self.mathematical_breakthroughs.append({
                'type': 'high_complexity',
                'tick': self.tick,
                'complex_expressions': len(high_complexity)
            })

def run_mathematics_experiment(max_ticks=1500, seed=None):
    """Run mathematical consciousness experiment"""
    if not seed:
        seed = random.randint(1000, 9999)
        
    print("=== MATHEMATICAL CONSCIOUSNESS EXPERIMENT ===")
    print(f"Testing mathematical pattern discovery and reasoning")
    print(f"Max ticks: {max_ticks}, Seed: {seed}\n")
    
    simulation = MathematicalSimulation(num_entities=25, seed=seed)
    
    for tick in range(max_ticks):
        simulation.simulation_step()
        
        if tick % 250 == 0:
            if not simulation.entities:
                print(f"T{tick:4d}: 💀 MATHEMATICAL CONSCIOUSNESS EXTINCT")
                break
            
            avg_consciousness = sum(e.consciousness_level for e in simulation.entities) / len(simulation.entities)
            mathematical_thinkers = sum(1 for e in simulation.entities if e.numerical_thinking > 0.7)
            
            print(f"T{tick:4d}: Pop={len(simulation.entities):2d}, "
                  f"C={avg_consciousness:.2f}, "
                  f"MathThinkers={mathematical_thinkers:2d}")
            
            if simulation.mathematical_expressions:
                recent_math = len([e for e in simulation.mathematical_expressions if e['tick'] > tick - 250])
                print(f"        🔢 {recent_math} mathematical expressions")
            
            if simulation.equation_attempts:
                recent_eq = len([e for e in simulation.equation_attempts if e['tick'] > tick - 250])
                print(f"        ⚖️  {recent_eq} equation attempts")
            
            if simulation.discovered_relationships:
                recent_disc = len([r for r in simulation.discovered_relationships if r['tick'] > tick - 250])
                print(f"        🔍 {recent_disc} pattern discoveries")
            
            if simulation.mathematical_breakthroughs:
                recent_breakthrough = len([b for b in simulation.mathematical_breakthroughs if b['tick'] > tick - 250])
                if recent_breakthrough > 0:
                    print(f"        ⚡ {recent_breakthrough} mathematical breakthroughs")
    
    # Mathematical analysis
    print(f"\n=== MATHEMATICAL CONSCIOUSNESS RESULTS ===")
    
    if simulation.entities:
        print(f"✅ Mathematical consciousness survived: {len(simulation.entities)} entities")
        print(f"🔢 Mathematical statistics:")
        print(f"   Total mathematical expressions: {len(simulation.mathematical_expressions)}")
        print(f"   Equation attempts: {len(simulation.equation_attempts)}")
        print(f"   Pattern discoveries: {len(simulation.discovered_relationships)}")
        print(f"   Mathematical breakthroughs: {len(simulation.mathematical_breakthroughs)}")
        
        # Show mathematical breakthroughs
        if simulation.mathematical_breakthroughs:
            print(f"\n⚡ Mathematical breakthroughs:")
            breakthrough_types = Counter(b['type'] for b in simulation.mathematical_breakthroughs)
            for btype, count in breakthrough_types.items():
                print(f"   {btype}: {count} occurrences")
        
        # Show discovered patterns
        if simulation.discovered_relationships:
            print(f"\n🔍 Discovered mathematical patterns:")
            all_patterns = []
            for rel in simulation.discovered_relationships:
                all_patterns.extend(rel['patterns'])
            
            pattern_counts = Counter(all_patterns)
            for pattern, count in pattern_counts.most_common(8):
                print(f"   {pattern}: {count} discoveries")
        
        # Show equation examples
        if simulation.equation_attempts:
            print(f"\n⚖️  Mathematical equations created:")
            recent_equations = simulation.equation_attempts[-8:]
            for eq in recent_equations:
                print(f"   \"{eq['equation']}\" (C:{eq['consciousness_level']:.2f}, Logic:{eq['logical_reasoning']:.2f})")
        
        # Show complex mathematical expressions
        if simulation.mathematical_expressions:
            print(f"\n🧮 Recent mathematical expressions:")
            complex_math = [e for e in simulation.mathematical_expressions[-15:] 
                           if e['expression']['numerical_complexity'] > 3.0]
            for expr_entry in complex_math:
                expr = expr_entry['expression']
                content = expr['content']
                print(f"   \"{content}\" (complexity: {expr['numerical_complexity']:.1f})")
        
    else:
        print("💀 Mathematical consciousness experiment failed")
    
    return simulation

if __name__ == "__main__":
    run_mathematics_experiment(max_ticks=1500)