#!/usr/bin/env python3
"""
Time-Dependent Vine Copula Demonstration

This script demonstrates the complete pipeline for time-dependent vine copula analysis.

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def run_quick_demo():
    """Run a quick demonstration with smaller parameters"""
    
    print("🚀 QUICK TIME-DEPENDENT VINE COPULA DEMO")
    print("=" * 60)
    
    # Import modules
    from time_dependent_flows import TimeDependentVineCopula
    from time_dependent_data_generator import TimeDependentDataGenerator
    
    # Configuration
    config = {'dim': 3, 'n_time_steps': 30, 'n_samples_per_time': 80, 'epochs': 50}
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Generate data
    print(f"\n📊 Generating synthetic data...")
    generator = TimeDependentDataGenerator(dim=config['dim'], random_seed=42)
    
    data, times, metadata = generator.generate_sinusoidal_correlation_data(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        base_correlation=0.4,
        amplitude=0.3,
        frequency=1.5
    )
    
    print(f"  ✅ Generated data shape: {data.shape}")
    
    # Initialize model
    print(f"\n🔮 Initializing time-dependent vine copula...")
    model = TimeDependentVineCopula(
        dim=config['dim'],
        vine_type='c-vine',
        optimization_method='tau',
        n_time_steps=config['n_time_steps']
    )
    
    model.initialize_vine_structure()
    model.initialize_flows(hidden_dim=16)
    
    print(f"  ✅ Initialized {len(model.flow_models)} flow models")
    
    # Fit model
    print(f"\n🎯 Training model...")
    try:
        model.fit(
            data, times,
            learning_rate=5e-3,
            num_epochs=config['epochs'],
            patience=15
        )
        
        # Analyze results
        predictions = model.predict_bandwidth_evolution()
        print(f"  ✅ Training completed!")
        print(f"  ✅ Generated predictions for {len(predictions)} edges")
        
        # Create visualizations
        generator.visualize_generated_data('sinusoidal', save_plots=True)
        
        print(f"\nResults saved to:")
        print(f"  Data: {generator.results_dir}")
        print(f"  Model: {model.results_dir}")
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")

def run_comprehensive_demo():
    """Run comprehensive analysis"""
    
    print("🔬 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    from comprehensive_time_dependent_analysis import ComprehensiveTimeDependentAnalysis
    
    # Initialize analyzer
    analyzer = ComprehensiveTimeDependentAnalysis(dim=3, random_seed=42)
    
    # Run analysis
    analyzer.run_complete_analysis(
        n_time_steps=80,
        n_samples_per_time=120,
        test_scenarios=['piecewise', 'sinusoidal']
    )
    
    print(f"✅ Comprehensive analysis complete!")
    print(f"📊 Results saved to: {analyzer.results_dir}")

def main():
    """Main demonstration function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Time-Dependent Vine Copula Demo")
    parser.add_argument('--quick', action='store_true', help='Run quick demo')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive analysis')
    
    args = parser.parse_args()
    
    if not (args.quick or args.comprehensive):
        args.quick = True
    
    print("🌟 TIME-DEPENDENT VINE COPULA WITH NORMALIZING FLOWS")
    print("=" * 80)
    
    if args.quick:
        run_quick_demo()
    
    if args.comprehensive:
        run_comprehensive_demo()
    
    print(f"\n🎉 Demo complete!")

if __name__ == "__main__":
    main() 