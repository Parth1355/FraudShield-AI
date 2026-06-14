"""
Complete Fraud Detection Pipeline
===================================
End-to-end pipeline that generates data, trains the model, and optionally runs the web app.

Usage:
    python pipeline.py                          # Run full pipeline
    python pipeline.py --skip-data-gen          # Skip data generation
    python pipeline.py --no-app                 # Don't launch Streamlit app
    python pipeline.py --data-rows 50000        # Generate with custom row count
    python pipeline.py --skip-app                # Alias for --no-app
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline modules from src
from generate_sample_data import generate
from train_model import (
    load_data,
    clean_training_data,
    engineer_training_features,
    train,
    save_artifacts,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_dataset_exists(dataset_path: str = "fraud_dataset.csv") -> bool:
    """Check if the dataset already exists."""
    if os.path.exists(dataset_path):
        return True
    if os.path.exists("fraud_dataset.csv.csv"):
        return True
    return os.path.exists(os.path.join("data", dataset_path))


def step_1_generate_data(num_rows: int = 100_000, force: bool = False) -> str:
    """
    Step 1: Generate synthetic data if it doesn't exist.
    
    Args:
        num_rows: Number of rows to generate
        force: Force regeneration even if data exists
    
    Returns:
        Path to the generated dataset
    """
    print_section("STEP 1: GENERATE SYNTHETIC DATA")
    
    dataset_path = os.path.join("data", "fraud_dataset.csv")
    
    if not force and check_dataset_exists(dataset_path):
        print(f"[INFO] Dataset already exists at {dataset_path}")
        print("[INFO] Skipping data generation. Use --force-data to regenerate.")
        return dataset_path
    
    print(f"[INFO] Generating {num_rows:,} synthetic transactions...")
    df = generate(n_rows=num_rows)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the generated data
    df.to_csv(dataset_path, index=False)
    print(f"[SUCCESS] Data saved to {dataset_path}")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud rate: {df['isFraud'].mean():.4%}")
    
    return dataset_path


def step_2_load_data(dataset_path: str):
    """
    Step 2: Load the dataset.
    
    Args:
        dataset_path: Path to the CSV file
    
    Returns:
        Loaded DataFrame
    """
    print_section("STEP 2: LOAD DATASET")
    
    try:
        df = load_data(dataset_path)
        print(f"[SUCCESS] Dataset loaded successfully")
        print(f"[INFO] Dataset shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nTroubleshooting tips:")
        print("  1. Ensure fraud_dataset.csv exists in the project root")
        print("  2. Run with --force-data to generate synthetic data")
        print("  3. Check the data/ directory")
        sys.exit(1)


def step_3_clean_data(df):
    """
    Step 3: Clean the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print_section("STEP 3: CLEAN DATA")
    
    df_clean = clean_training_data(df)
    print(f"[SUCCESS] Data cleaning complete")
    print(f"[INFO] Fraud rate: {df_clean['isFraud'].mean():.4%}")
    
    return df_clean


def step_4_engineer_features(df):
    """
    Step 4: Engineer features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Feature-engineered DataFrame
    """
    print_section("STEP 4: ENGINEER FEATURES")
    
    df_features = engineer_training_features(df)
    print(f"[SUCCESS] Feature engineering complete")
    
    return df_features


def step_5_train_model(df):
    """
    Step 5: Train the fraud detection model.
    
    Args:
        df: Feature-engineered DataFrame
    
    Returns:
        Tuple of (model, preprocessor, metrics)
    """
    print_section("STEP 5: TRAIN MODEL")
    
    model, preprocessor, metrics = train(df)
    print(f"[SUCCESS] Model training complete")
    print(f"\n[METRICS SUMMARY]")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    
    return model, preprocessor, metrics


def step_6_save_artifacts(model, preprocessor, metrics, out_dir: str = "."):
    """
    Step 6: Save model artifacts.
    
    Args:
        model: Trained model
        preprocessor: Feature preprocessor
        metrics: Performance metrics
        out_dir: Output directory for artifacts
    """
    print_section("STEP 6: SAVE ARTIFACTS")
    
    save_artifacts(model, preprocessor, metrics, out_dir=out_dir)
    print(f"[SUCCESS] All artifacts saved to {out_dir}")
    

def step_7_launch_app():
    """
    Step 7: Launch the Streamlit web application.
    """
    print_section("STEP 7: LAUNCH WEB APPLICATION")
    
    print("[INFO] Starting Streamlit app...")
    print("[INFO] The app will open at http://localhost:8501")
    print("[INFO] Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            ["streamlit", "run", "src/app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to launch Streamlit app: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")
    except FileNotFoundError:
        print("[ERROR] Streamlit is not installed or not in PATH")
        print("\nInstall it with:")
        print("  pip install streamlit")


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Complete Fraud Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                         # Run full pipeline with app
  python pipeline.py --no-app                # Train model without launching app
  python pipeline.py --skip-data-gen         # Skip data generation
  python pipeline.py --force-data            # Regenerate data even if it exists
  python pipeline.py --data-rows 50000       # Generate 50k rows of data
        """
    )
    
    parser.add_argument(
        "--data",
        default="data/fraud_dataset.csv",
        help="Path to dataset (default: data/fraud_dataset.csv)"
    )
    parser.add_argument(
        "--data-rows",
        type=int,
        default=100_000,
        help="Number of rows to generate (default: 100,000)"
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation step"
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Force regenerate data even if it exists"
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Don't launch Streamlit app after training"
    )
    parser.add_argument(
        "--skip-app",
        action="store_true",
        help="Alias for --no-app"
    )
    parser.add_argument(
        "--out",
        default="models",
        help="Output directory for model artifacts (default: models)"
    )
    
    args = parser.parse_args()
    
    # Resolve skip-app flag
    skip_app = args.no_app or args.skip_app
    
    try:
        print("\n" + "=" * 70)
        print("  FRAUD DETECTION PIPELINE - COMPLETE WORKFLOW")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Dataset: {args.data}")
        print(f"  Data rows: {args.data_rows:,}")
        print(f"  Skip data generation: {args.skip_data_gen}")
        print(f"  Force data regeneration: {args.force_data}")
        print(f"  Launch app: {not skip_app}")
        print(f"  Output directory: {args.out}")
        
        # Step 1: Generate Data
        if not args.skip_data_gen:
            dataset_path = step_1_generate_data(
                num_rows=args.data_rows,
                force=args.force_data
            )
        else:
            dataset_path = args.data
            print_section("STEP 1: GENERATE SYNTHETIC DATA")
            print("[INFO] Skipping data generation as requested")
        
        # Step 2: Load Data
        df = step_2_load_data(dataset_path)
        
        # Step 3: Clean Data
        df = step_3_clean_data(df)
        
        # Step 4: Engineer Features
        df = step_4_engineer_features(df)
        
        # Step 5: Train Model
        model, preprocessor, metrics = step_5_train_model(df)
        
        # Step 6: Save Artifacts
        step_6_save_artifacts(model, preprocessor, metrics, out_dir=args.out)
        
        # Step 7: Launch App (optional)
        print_section("PIPELINE COMPLETE")
        print("[SUCCESS] All pipeline steps completed successfully!")
        print(f"\nArtifacts saved to: {args.out}/")
        print(f"  • model.pkl")
        print(f"  • preprocessor.pkl")
        print(f"  • metrics.pkl")
        
        if not skip_app:
            print(f"\n[INFO] Launching web application...")
            step_7_launch_app()
        else:
            print(f"\nTo launch the web app, run:")
            print(f"  streamlit run src/app.py")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
