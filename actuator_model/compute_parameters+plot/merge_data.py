import pandas as pd

def merge_test_plan_with_results(test_plan_path, results_path, output_path):
    # Read the test plan and results files
    test_plan_df = pd.read_csv(test_plan_path)
    results_df = pd.read_csv(results_path)

    # Merge the test plan data with the results data
    merged_df = pd.merge(results_df, test_plan_df[['testno', 'rps', 'elevation_input', 'azimuth_input']], on='testno')

    # Rearrange the columns to place 'rps', 'elevation', and 'azimuth' as the 2nd, 3rd, and 4th columns
    columns_order = ['testno', 'rps', 'elevation_input', 'azimuth_input'] + [col for col in results_df.columns if col != 'testno']
    merged_df = merged_df[columns_order]

    # Save the merged data to a new CSV file
    merged_df.to_csv(output_path, index=False)

# Usage
test_plan_path = 'test_plan.csv'  # Replace with the path to your test plan file
results_path = 'ft_results_20DEC23.csv'  # Replace with the path to your results file
output_path = 'full_ft_results_20DEC23.csv'  # Replace with your desired output file path

merge_test_plan_with_results(test_plan_path, results_path, output_path)
