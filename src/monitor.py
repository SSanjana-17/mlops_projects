import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def main():
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    print("=" * 60)
    print("ML Pipeline Data Drift Monitoring")
    print("=" * 60)
    
    # Load the cleaned data
    try:
        data = pd.read_csv("data/processed/clean_data.csv")
        print(f"\n✓ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    except FileNotFoundError:
        print("\n✗ Error: data/processed/clean_data.csv not found")
        print("Please run src/clean.py first to generate clean data")
        return
    
    # Split data into reference (training) and current (monitoring)
    # Use first 70% as reference baseline, last 30% as current data
    split_index = int(len(data) * 0.7)
    reference_data = data[:split_index].copy()
    current_data = data[split_index:].copy()
    
    print(f"\n📊 Data Split:")
    print(f"  Reference data: {reference_data.shape[0]} rows (baseline)")
    print(f"  Current data:   {current_data.shape[0]} rows (monitoring)")
    
    # Create comprehensive drift report
    print("\n🔍 Generating data drift report...")
    report = Report(
        metrics=[
            DataDriftPreset(),      # Detect feature drift
            DataQualityPreset()     # Check data quality issues
        ]
    )
    
    # Run the report
    report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Save reports
    html_report = "reports/data_drift_report.html"
    json_report = "reports/data_drift_report.json"
    
    report.save_html(html_report)
    report.save_json(json_report)
    
    print(f"\n✅ Reports generated successfully!")
    print(f"  📄 HTML: {html_report}")
    print(f"  📄 JSON: {json_report}")
    print(f"\n💡 Open {html_report} in your browser to view the interactive report")
    print("=" * 60)

if __name__ == "__main__":
    main()
