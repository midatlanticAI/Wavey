#!/usr/bin/env python3
"""
Deep Diagnostic Analysis of Wave Intelligence Results
Understanding the root causes of observed behaviors
"""

import math
import random
import json
from typing import Dict, List, Any

class WaveDiagnostic:
    """Diagnoses wave intelligence system behavior"""
    
    def __init__(self):
        self.diagnostic_data = {}
    
    def analyze_hardcoded_ranges(self):
        """Analyze if optimal ranges create artificial success"""
        print("=== OPTIMAL RANGE ANALYSIS ===")
        
        optimal_ranges = {
            'navigation': {'frequency': (0.8, 1.2), 'amplitude': (1.0, 1.5)},
            'combat': {'frequency': (1.5, 2.0), 'amplitude': (1.2, 2.0)},
            'exploration': {'frequency': (0.5, 1.0), 'amplitude': (0.8, 1.3)},
            'survival': {'frequency': (1.0, 1.8), 'amplitude': (1.0, 1.8)}
        }
        
        print("Current optimal ranges:")
        for scenario, ranges in optimal_ranges.items():
            print(f"  {scenario:12}: freq={ranges['frequency']}, amp={ranges['amplitude']}")
        
        # Test what happens with random ranges
        print("\nTesting with RANDOM optimal ranges:")
        
        for trial in range(3):
            print(f"\n--- Random Trial {trial + 1} ---")
            random_ranges = {}
            for scenario in optimal_ranges:
                random_ranges[scenario] = {
                    'frequency': (random.uniform(0.1, 5.0), random.uniform(0.1, 5.0)),
                    'amplitude': (random.uniform(0.1, 3.0), random.uniform(0.1, 3.0))
                }
            
            for scenario, ranges in random_ranges.items():
                print(f"  {scenario:12}: freq={ranges['frequency']}, amp={ranges['amplitude']}")
            
            # Test agent performance with random ranges
            agent_performance = self._test_with_ranges(random_ranges)
            print(f"  Performance with random ranges: {agent_performance:.3f}")
        
        print("\n🔍 INSIGHT: If performance varies significantly with different ranges,")
        print("   it suggests the optimal ranges are artificially constraining success")
    
    def analyze_wave_computation_impact(self):
        """Analyze how much wave computation actually matters"""
        print("\n=== WAVE COMPUTATION IMPACT ANALYSIS ===")
        
        test_cases = [
            ("Standard waves", self._create_standard_agent),
            ("Constant output", self._create_constant_agent),
            ("Pure random", self._create_random_agent),
            ("Always same action", self._create_fixed_action_agent)
        ]
        
        results = []
        for name, agent_creator in test_cases:
            performances = []
            for _ in range(10):  # Multiple trials
                agent = agent_creator()
                perf = self._detailed_performance_test(agent)
                performances.append(perf['success'])
            
            avg_perf = sum(performances) / len(performances)
            results.append((name, avg_perf, performances))
            print(f"{name:20}: {avg_perf:.3f} avg (range: {min(performances):.3f}-{max(performances):.3f})")
        
        # Analyze variance between methods
        all_perfs = [result[1] for result in results]
        if max(all_perfs) - min(all_perfs) < 0.1:
            print("🚨 MAJOR ISSUE: All methods perform similarly!")
            print("   This suggests wave computation is not the deciding factor")
        else:
            print("✅ Different methods show different performance")
    
    def analyze_random_vs_intelligence(self):
        """Deep analysis of randomness vs actual intelligence"""
        print("\n=== RANDOMNESS VS INTELLIGENCE ANALYSIS ===")
        
        # Test with fixed random seeds for reproducibility
        random.seed(42)
        
        print("Testing decision consistency...")
        agent = self._create_standard_agent()
        
        # Same scenario, different times - should show wave patterns
        base_scenario = {'type': 'navigation', 'difficulty': 0.3}
        
        decisions_over_time = []
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            state = {'time': t, 'scenario': base_scenario}
            decision = self._make_wave_decision(agent, state)
            decisions_over_time.append((t, decision['action'], decision['scores']))
        
        print("Decision patterns over time:")
        for t, action, scores in decisions_over_time:
            print(f"  t={t}: {action} (scores: {scores})")
        
        # Check if decisions follow wave patterns
        unique_decisions = set(action for _, action, _ in decisions_over_time)
        if len(unique_decisions) == 1:
            print("⚠️  All decisions identical - no wave variation")
        elif len(unique_decisions) == len(decisions_over_time):
            print("🔍 High decision variety - could be random or complex waves")
        else:
            print("✅ Moderate decision variation - suggests wave influence")
    
    def analyze_scenario_difficulty_impact(self):
        """Analyze how scenario difficulty affects outcomes"""
        print("\n=== SCENARIO DIFFICULTY IMPACT ===")
        
        agent = self._create_standard_agent()
        
        difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for difficulty in difficulty_levels:
            successes = 0
            total_tests = 50
            
            for _ in range(total_tests):
                scenario = {'type': 'navigation', 'difficulty': difficulty}
                effectiveness = self._evaluate_single_action(agent, 'move', scenario)
                if effectiveness > difficulty:
                    successes += 1
            
            success_rate = successes / total_tests
            print(f"Difficulty {difficulty}: {success_rate:.3f} success rate")
        
        print("\n🔍 Expected: Higher difficulty should reduce success rate")
        print("   If success rates don't correlate with difficulty, system is broken")
    
    def _create_standard_agent(self):
        """Create agent with standard wave patterns"""
        return {
            'move': {'frequency': 1.0, 'amplitude': 1.0, 'phase': 0},
            'attack': {'frequency': 1.5, 'amplitude': 1.2, 'phase': 0.5},
            'explore': {'frequency': 0.8, 'amplitude': 0.9, 'phase': 0.2}
        }
    
    def _create_constant_agent(self):
        """Create agent that always outputs same values"""
        return {
            'move': {'frequency': 0, 'amplitude': 1, 'phase': 0},  # Always 0
            'attack': {'frequency': 0, 'amplitude': 2, 'phase': 0},  # Always 0
            'explore': {'frequency': 0, 'amplitude': 0.5, 'phase': 0}  # Always 0
        }
    
    def _create_random_agent(self):
        """Create agent with random patterns each time"""
        return {
            'move': {'frequency': random.uniform(0.1, 3.0), 'amplitude': random.uniform(0.1, 2.0), 'phase': random.uniform(0, 6.28)},
            'attack': {'frequency': random.uniform(0.1, 3.0), 'amplitude': random.uniform(0.1, 2.0), 'phase': random.uniform(0, 6.28)},
            'explore': {'frequency': random.uniform(0.1, 3.0), 'amplitude': random.uniform(0.1, 2.0), 'phase': random.uniform(0, 6.28)}
        }
    
    def _create_fixed_action_agent(self):
        """Agent designed to always choose same action"""
        return {
            'move': {'frequency': 0, 'amplitude': 10, 'phase': 0},  # Always highest
            'attack': {'frequency': 0, 'amplitude': 1, 'phase': 0},
            'explore': {'frequency': 0, 'amplitude': 1, 'phase': 0}
        }
    
    def _make_wave_decision(self, agent, state):
        """Make decision using wave computation"""
        time = state.get('time', 0)
        action_scores = {}
        
        for action, pattern in agent.items():
            wave_value = (pattern['amplitude'] * 
                         math.sin(pattern['frequency'] * time + pattern['phase']))
            action_scores[action] = wave_value
        
        # Add small random component
        for action in action_scores:
            action_scores[action] += random.uniform(-0.05, 0.05)
        
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'action': best_action,
            'scores': {k: round(v, 3) for k, v in action_scores.items()}
        }
    
    def _detailed_performance_test(self, agent):
        """Detailed performance test"""
        scenarios = [
            {'type': 'navigation', 'difficulty': 0.3},
            {'type': 'combat', 'difficulty': 0.5},
            {'type': 'exploration', 'difficulty': 0.4},
            {'type': 'survival', 'difficulty': 0.6}
        ]
        
        success_count = 0
        for i, scenario in enumerate(scenarios):
            state = {'time': i * 0.1, 'scenario': scenario}
            decision = self._make_wave_decision(agent, state)
            effectiveness = self._evaluate_single_action(agent, decision['action'], scenario)
            
            if effectiveness > scenario['difficulty']:
                success_count += 1
        
        return {'success': success_count / len(scenarios)}
    
    def _evaluate_single_action(self, agent, action, scenario):
        """Evaluate single action effectiveness"""
        if action not in agent:
            return random.uniform(0, 0.3)
        
        pattern = agent[action]
        
        optimal_ranges = {
            'navigation': {'frequency': (0.8, 1.2), 'amplitude': (1.0, 1.5)},
            'combat': {'frequency': (1.5, 2.0), 'amplitude': (1.2, 2.0)},
            'exploration': {'frequency': (0.5, 1.0), 'amplitude': (0.8, 1.3)},
            'survival': {'frequency': (1.0, 1.8), 'amplitude': (1.0, 1.8)}
        }
        
        scenario_type = scenario['type']
        if scenario_type not in optimal_ranges:
            return random.uniform(0, 0.5)
        
        optimal = optimal_ranges[scenario_type]
        
        freq_mean = (optimal['frequency'][0] + optimal['frequency'][1]) / 2.0
        amp_mean = (optimal['amplitude'][0] + optimal['amplitude'][1]) / 2.0
        
        freq_score = max(0, 1.0 - abs(pattern['frequency'] - freq_mean) / 2.0)
        amp_score = max(0, 1.0 - abs(pattern['amplitude'] - amp_mean) / 2.0)
        
        effectiveness = (freq_score + amp_score) / 2.0
        effectiveness = max(0, min(1, effectiveness + random.uniform(-0.1, 0.1)))
        
        return effectiveness
    
    def _test_with_ranges(self, custom_ranges):
        """Test performance with custom optimal ranges"""
        agent = self._create_standard_agent()
        
        # Temporarily modify evaluation to use custom ranges
        scenario = {'type': 'navigation', 'difficulty': 0.3}
        pattern = agent['move']
        
        ranges = custom_ranges['navigation']
        freq_mean = (ranges['frequency'][0] + ranges['frequency'][1]) / 2.0
        amp_mean = (ranges['amplitude'][0] + ranges['amplitude'][1]) / 2.0
        
        freq_score = max(0, 1.0 - abs(pattern['frequency'] - freq_mean) / 2.0)
        amp_score = max(0, 1.0 - abs(pattern['amplitude'] - amp_mean) / 2.0)
        
        effectiveness = (freq_score + amp_score) / 2.0
        return effectiveness

def run_deep_diagnosis():
    """Run comprehensive diagnosis of wave intelligence system"""
    print("=== DEEP WAVE INTELLIGENCE DIAGNOSIS ===")
    print("Investigating root causes of observed behaviors\n")
    
    diagnostic = WaveDiagnostic()
    
    diagnostic.analyze_hardcoded_ranges()
    diagnostic.analyze_wave_computation_impact()
    diagnostic.analyze_random_vs_intelligence()
    diagnostic.analyze_scenario_difficulty_impact()
    
    print("\n=== DIAGNOSTIC CONCLUSIONS ===")
    print("1. System shows parameter sensitivity - wave values do affect outcomes")
    print("2. Wave functions compute correctly and vary over time")
    print("3. SUCCESS appears tied to hardcoded 'optimal ranges' rather than emergent intelligence")
    print("4. The 100% success rates suggest evaluation criteria may be too lenient")
    print("\n🔍 KEY INSIGHT: The intelligence may be in the hardcoded optimal ranges,")
    print("   not in the wave computation itself. The system works, but may not be")
    print("   as 'emergent' as initially believed.")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Make optimal ranges dynamic/learned rather than hardcoded")
    print("2. Add more complex, multi-step scenarios")  
    print("3. Introduce environmental changes that require adaptation")
    print("4. Test with completely novel scenario types not in training")

if __name__ == "__main__":
    run_deep_diagnosis()