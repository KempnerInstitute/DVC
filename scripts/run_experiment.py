#!/usr/bin/env python3
"""
Command-line interface for running DVC experiments.

This script provides a simple way to run experiments using YAML configuration files.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path so we can import dvc_package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.experiments.experiment_framework import ExperimentRunner


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run DVC experiments from YAML configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run probability analysis experiment
  python scripts/run_experiment.py drafts/configs/probability_analysis.yaml

  # Run entropy analysis with debug logging
  python scripts/run_experiment.py drafts/configs/entropy_analysis.yaml --log-level DEBUG

  # Run time-dependent experiment
  python scripts/run_experiment.py drafts/configs/time_dependent.yaml

  # Create example configuration files
  python scripts/run_experiment.py --create-examples

  # List available example configurations
  python scripts/run_experiment.py --list-examples
        """
    )
    
    parser.add_argument(
        'config', 
        nargs='?',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--create-examples',
        action='store_true',
        help='Create example configuration files in configs/'
    )
    
    parser.add_argument(
        '--list-examples',
        action='store_true',
        help='List available example configuration files'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Override output directory from config file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger('DVC.main')
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Handle special commands
    if args.create_examples:
        runner.create_example_configs("configs")
        print("Example configuration files created in 'configs/' directory")
        print("\nAvailable examples:")
        print("  - probability_analysis.yaml: Probability distribution analysis")
        print("  - entropy_analysis.yaml: Entropy and information estimation")
        print("  - time_dependent.yaml: Time-dependent vine copula modeling")
        print("  - comprehensive_comparison.yaml: Comprehensive comparison study")
        print("\nUsage: python scripts/run_experiment.py <config_file>.yaml")
        return
    
    if args.list_examples:
        examples_dir = Path("configs")
        if examples_dir.exists():
            yaml_files = list(examples_dir.glob("*.yaml"))
            if yaml_files:
                print("Available configuration files:")
                for yaml_file in yaml_files:
                    print(f"  - {yaml_file}")
            else:
                print("No configuration files found.")
                print("Run with --create-examples to create them.")
        else:
            print("Configs directory not found.")
            print("Run with --create-examples to create example configuration files.")
        return
    
    # Check if config file is provided
    if not args.config:
        parser.print_help()
        print("\nError: Configuration file is required.")
        print("Use --create-examples to create example files, or --list-examples to see available examples.")
        sys.exit(1)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Run experiment
    try:
        logger.info(f"Starting experiment with config: {config_path}")
        print(f"Running experiment: {config_path}")
        print("=" * 60)
        
        results = runner.run_from_config(str(config_path))
        
        # Print summary
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print(f"Results saved to: {results.get('config', {}).get('output_dir', 'results')}")
        print(f"Experiment type: {results.get('experiment_type', 'unknown')}")
        print(f"Timestamp: {results.get('timestamp', 'unknown')}")
        
        # Print specific results based on experiment type
        experiment_type = results.get('experiment_type')
        
        if experiment_type == 'probability_analysis':
            data_info = results.get('data_info', {})
            print(f"Data: {data_info.get('n_samples', 'N/A')} samples, {data_info.get('n_variables', 'N/A')} variables")
            
            vine_analysis = results.get('vine_analysis', {})
            print(f"Vine types analyzed: {list(vine_analysis.keys())}")
            
        elif experiment_type == 'entropy_analysis':
            entropy_analysis = results.get('entropy_analysis', {})
            print(f"Datasets analyzed: {list(entropy_analysis.keys())}")
            
            method_comparison = results.get('method_comparison', {})
            print(f"Entropy methods: {list(method_comparison.keys())}")
            
        elif experiment_type == 'time_dependent':
            data_info = results.get('data_info', {})
            print(f"Time steps: {data_info.get('n_time_steps', 'N/A')}")
            print(f"Time range: {data_info.get('time_range', 'N/A')}")
            
            model_results = results.get('model_results', {})
            if model_results.get('model_created', False):
                print(f"Model: {model_results.get('n_edges', 'N/A')} edges, hidden_dim={model_results.get('hidden_dim', 'N/A')}")
        
        print("\nCheck the results directory for detailed outputs and plots.")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\nExperiment failed: {e}")
        print("Check experiment.log for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
