#!/usr/bin/env python3
"""
Wave-Based Arithmetic Core - First Principles Implementation
Ground zero approach with falsifiability criteria
Pure Python implementation - no external libraries
"""

import math
from typing import Dict, Any, List, Tuple
import time

class WaveNumber:
    """
    Represents a number as a wave with amplitude, frequency, and phase
    
    Encoding scheme:
    - Frequency: represents the number value (value * base_freq)
    - Amplitude: normalized to 1.0
    - Phase: initial phase offset (0.0 for simplicity)
    """
    
    def __init__(self, value: int, duration: float = 1.0, sample_rate: int = 100):
        """
        Create wave representation of a number
        
        Args:
            value: The integer to represent (1-10 for initial testing)
            duration: Time duration of wave in seconds  
            sample_rate: Samples per second
        """
        self.value = value
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        
        # Wave parameters
        self.base_freq = 10  # Hz - each number is this * value
        self.frequency = value * self.base_freq
        self.amplitude = 1.0  # Normalized
        self.phase = 0.0     # No initial phase offset
        
        # Generate time points and wave values manually
        self.time_points = []
        self.wave_values = []
        
        for i in range(self.samples):
            t = i / self.sample_rate
            self.time_points.append(t)
            
            # Calculate wave value: amplitude * sin(2π * frequency * time + phase)
            wave_val = self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phase)
            self.wave_values.append(wave_val)
    
    def get_wave_signature(self) -> Dict[str, float]:
        """Return wave parameters for verification"""
        # Calculate peak amplitude manually
        abs_values = [abs(val) for val in self.wave_values]
        peak_amplitude = max(abs_values)
        
        # Calculate RMS manually
        sum_squares = sum(val**2 for val in self.wave_values)
        mean_squares = sum_squares / len(self.wave_values)
        rms = math.sqrt(mean_squares)
        
        return {
            'value': self.value,
            'amplitude': self.amplitude,
            'frequency': self.frequency,
            'phase': self.phase,
            'duration': self.duration,
            'peak_amplitude': peak_amplitude,
            'rms': rms
        }
    
    def decode_from_wave(self) -> int:
        """
        Attempt to decode the number from wave properties
        This tests if our encoding is recoverable
        """
        # Use frequency to decode
        decoded_value = round(self.frequency / self.base_freq)
        return decoded_value
    
    def print_wave_sample(self, num_points: int = 10):
        """Print a sample of wave values for visualization"""
        print(f"Wave sample for value {self.value} (frequency {self.frequency} Hz):")
        step = max(1, len(self.wave_values) // num_points)
        for i in range(0, len(self.wave_values), step):
            t = self.time_points[i]
            val = self.wave_values[i]
            print(f"  t={t:.3f}s: {val:.6f}")
        print()

def test_wave_encoding():
    """
    Criterion 1 Test: Can we encode integers 1-3 as distinct, reproducible waves?
    """
    print("=== CRITERION 1: WAVE NUMBER ENCODING TEST ===")
    print("Testing if integers 1-3 can be encoded as distinct wave signatures")
    print()
    
    # Create wave representations
    numbers = [1, 2, 3]
    waves = {}
    signatures = {}
    
    for num in numbers:
        waves[num] = WaveNumber(num)
        signatures[num] = waves[num].get_wave_signature()
        
        print(f"Number {num}:")
        print(f"  Frequency: {signatures[num]['frequency']} Hz")
        print(f"  Peak Amplitude: {signatures[num]['peak_amplitude']:.6f}")
        print(f"  RMS: {signatures[num]['rms']:.6f}")
        print()
    
    # Test 1: Are signatures distinct?
    print("DISTINCTNESS TEST:")
    frequencies = [signatures[num]['frequency'] for num in numbers]
    if len(set(frequencies)) == len(frequencies):
        print("✅ All wave signatures are distinct")
    else:
        print("❌ Wave signatures are not distinct")
        return False
    
    # Test 2: Are they reproducible?
    print("\nREPRODUCIBILITY TEST:")
    reproducible = True
    for num in numbers:
        wave2 = WaveNumber(num)  # Create second instance
        sig2 = wave2.get_wave_signature()
        
        if abs(signatures[num]['frequency'] - sig2['frequency']) > 0.001:
            print(f"❌ Number {num} not reproducible")
            reproducible = False
        else:
            print(f"✅ Number {num} reproducible")
    
    # Test 3: Can we decode back to original number?
    print("\nDECODABILITY TEST:")
    decodable = True
    for num in numbers:
        decoded = waves[num].decode_from_wave()
        if decoded == num:
            print(f"✅ Number {num} correctly decoded as {decoded}")
        else:
            print(f"❌ Number {num} incorrectly decoded as {decoded}")
            decodable = False
    
    # Final verdict
    print("\n" + "="*50)
    print("CRITERION 1 RESULTS:")
    
    if len(set(frequencies)) == len(frequencies) and reproducible and decodable:
        print("✅ CRITERION 1 PASSED: Wave encoding is valid")
        return True
    else:
        print("❌ CRITERION 1 FAILED: Wave encoding is invalid")
        return False

def wave_interference_addition(wave1: WaveNumber, wave2: WaveNumber) -> List[float]:
    """
    Perform addition through wave interference
    Add the wave values point by point (superposition principle)
    """
    if len(wave1.wave_values) != len(wave2.wave_values):
        raise ValueError("Waves must have same number of samples")
    
    # Superposition: add corresponding wave values
    interference_pattern = []
    for i in range(len(wave1.wave_values)):
        combined_value = wave1.wave_values[i] + wave2.wave_values[i]
        interference_pattern.append(combined_value)
    
    return interference_pattern

def decode_interference_result(interference_pattern: List[float], time_points: List[float], base_freq: float = 10) -> int:
    """
    Attempt to decode the result from interference pattern
    This is the critical test - can we extract the sum?
    """
    # Method 1: Frequency domain analysis (simplified)
    # Look for dominant frequency in the interference pattern
    
    # Sample at different frequencies to find the strongest
    max_correlation = 0
    best_frequency = 0
    
    # Test frequencies corresponding to numbers 1-10
    for test_value in range(1, 11):
        test_freq = test_value * base_freq
        correlation = 0
        
        # Calculate correlation with test frequency
        for i, t in enumerate(time_points):
            expected_val = math.sin(2 * math.pi * test_freq * t)
            actual_val = interference_pattern[i]
            correlation += expected_val * actual_val
        
        correlation = abs(correlation)
        if correlation > max_correlation:
            max_correlation = correlation
            best_frequency = test_freq
    
    # Decode frequency back to number
    decoded_result = round(best_frequency / base_freq)
    return decoded_result, max_correlation

def test_wave_addition():
    """
    Criterion 2 Test: Does wave interference perform addition?
    Test: 2 + 1 = 3
    """
    print("=== CRITERION 2: WAVE INTERFERENCE ADDITION TEST ===")
    print("Testing if wave interference can perform addition: 2 + 1 = 3")
    print()
    
    # Create waves for 2 and 1
    wave_2 = WaveNumber(2)
    wave_1 = WaveNumber(1)
    
    print("Input waves:")
    print(f"  Wave for 2: frequency = {wave_2.frequency} Hz")
    print(f"  Wave for 1: frequency = {wave_1.frequency} Hz")
    print(f"  Expected result: 3 (frequency = 30 Hz)")
    print()
    
    # Perform wave interference
    print("Performing wave interference (superposition)...")
    interference_result = wave_interference_addition(wave_2, wave_1)
    
    # Try to decode the result
    decoded_value, correlation = decode_interference_result(
        interference_result, wave_2.time_points, wave_2.base_freq
    )
    
    print(f"Decoded result: {decoded_value}")
    print(f"Correlation strength: {correlation:.2f}")
    print()
    
    # Test if result is correct
    expected_result = 3
    if decoded_value == expected_result:
        print("✅ ADDITION SUCCESS: 2 + 1 = 3 via wave interference")
        return True
    else:
        print(f"❌ ADDITION FAILED: Expected {expected_result}, got {decoded_value}")
        
        # Analyze why it failed
        print("\nFAILURE ANALYSIS:")
        print("Wave interference pattern sample:")
        for i in range(0, min(20, len(interference_result)), 2):
            t = wave_2.time_points[i]
            val = interference_result[i]
            print(f"  t={t:.3f}s: {val:.6f}")
        
        return False

if __name__ == "__main__":
    # First test wave encoding
    encoding_success = test_wave_encoding()
    
    if encoding_success:
        print("\n" + "="*70)
        # Now test wave arithmetic
        arithmetic_success = test_wave_addition()
        
        if arithmetic_success:
            print("\n🎯 CRITERION 2 PASSED: Wave arithmetic works!")
            print("Ready to proceed to efficiency testing...")
        else:
            print("\n❌ CRITERION 2 FAILED: Wave arithmetic doesn't work")
            print("Need to revise approach or abandon wave computation hypothesis")
    else:
        print("\n❌ Cannot proceed - wave encoding failed")