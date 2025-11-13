"""
Script to run both experiments
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from experiments.experiment1_news_context import NewsContextExperiment
from experiments.experiment2_node_types import NodeTypesExperiment


def main():
    parser = argparse.ArgumentParser(description='Run ML experiments')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv',
                       help='Path to data CSV')
    parser.add_argument('--exp', type=str, choices=['1', '2', 'both'], default='both',
                       help='Which experiment to run')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.exp in ['1', 'both']:
        print("=" * 60)
        print("Running Experiment 1: News Context")
        print("=" * 60)
        exp1 = NewsContextExperiment(
            data_path=args.data,
            output_dir=f"{args.output_dir}/exp1"
        )
        exp1.setup()
        results1 = exp1.run()
        exp1.save_results("exp1_results.json")
        print(f"\nExperiment 1 complete! Results saved to {args.output_dir}/exp1/")
    
    if args.exp in ['2', 'both']:
        print("\n" + "=" * 60)
        print("Running Experiment 2: Node Types")
        print("=" * 60)
        exp2 = NodeTypesExperiment(
            data_path=args.data,
            output_dir=f"{args.output_dir}/exp2"
        )
        exp2.setup()
        results2 = exp2.run()
        exp2.save_results("exp2_results.json")
        print(f"\nExperiment 2 complete! Results saved to {args.output_dir}/exp2/")
    
    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

