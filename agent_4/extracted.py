import pandas as pd
import os

def get_unique_column_name_for_row(existing_names_in_row, desired_name_base):
    """
    Given a list of column names already used in the *current row* for extracted keys,
    find a unique name by appending a suffix like _1, _2, etc.
    This handles duplicates within the same cell/row.
    
    Args:
        existing_names_in_row (set): A set of names already used in the current row's processing.
        desired_name_base (str): The base name (e.g., 'Extracted_Test_SPV水平峰值').

    Returns:
        str: A unique column name for the current row.
    """
    if desired_name_base not in existing_names_in_row:
        existing_names_in_row.add(desired_name_base)
        return desired_name_base

    counter = 1
    new_name = f"{desired_name_base}_{counter}"
    while new_name in existing_names_in_row:
        counter += 1
        new_name = f"{desired_name_base}_{counter}"
    existing_names_in_row.add(new_name)
    return new_name


def split_csv_with_dynamic_columns_and_row_uniqueness(file_path):
    """
    Reads a CSV file, finds all columns containing '原始' in their name,
    splits their content (semicolon-separated key-value pairs), and adds
    the extracted keys as new columns to the DataFrame. Handles duplicate
    key names *within the same row* by appending numbers (_1, _2, ...).
    """
    # 1. Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    print(f"Processing file: {file_path}")
    print("Original DataFrame shape:", df.shape)
    print("-" * 40)

    # 2. Dynamically find all columns that contain '原始' in their name
    original_cols_mask = df.columns.str.contains('原始', na=False)
    columns_to_split = df.columns[original_cols_mask].tolist()

    print(f"Found {len(columns_to_split)} column(s) with '原始' in the name: {columns_to_split}")
    if not columns_to_split:
        print("No columns containing '原始' were found. Exiting.")
        return

    # 3. Process each identified column
    for col_name in columns_to_split:
        print(f"\nProcessing column: '{col_name}'...")

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            cell_content = row[col_name]
            
            # Keep track of new column names generated for this specific row
            # to handle intra-row duplicates like "SPV水平峰值 尚未分析; SPV水平峰值 -7"
            names_used_in_current_row = set()

            if pd.notna(cell_content) and isinstance(cell_content, str) and cell_content.strip() != '':
                pairs = [pair.strip() for pair in cell_content.split(';')]
                
                for pair in pairs:
                    pair = pair.strip()
                    parts = pair.split(None, 1) 
                    
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        if key:
                            # Create the base new column name with prefix
                            new_col_prefix = f"Extracted_{col_name.replace('原始', '')}_"
                            desired_col_name_base = f"{new_col_prefix}{key}"
                            
                            # Get a unique name specifically for this row
                            unique_col_name_for_row = get_unique_column_name_for_row(
                                names_used_in_current_row, desired_col_name_base
                            )
                            
                            # Assign the value to the uniquely named column for this row
                            df.at[index, unique_col_name_for_row] = value

    # 4. Display information about the final, modified DataFrame
    print("-" * 40)
    print("Final Modified DataFrame shape:", df.shape)
    unique_extracted_cols = [c for c in df.columns if c.startswith('Extracted_')]
    print(f"Number of new 'Extracted_' columns added: {len(unique_extracted_cols)}")
    
    # Show a sample of the new columns
    if unique_extracted_cols:
        print(f"Sample of new columns added: {unique_extracted_cols[:10]} ...")

    # 5. Generate output filename based on input filename
    base_name, ext = os.path.splitext(file_path)
    output_file_path = f"{base_name}_展开后{ext}"

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"\nModified DataFrame saved to: {output_file_path}")
    print("Processing complete!")


# --- Main Execution ---
if __name__ == "__main__":
    # Prompt user for the input file path
    input_file_path = input("请输入要处理的CSV文件路径: ").strip()
    
    # Validate if the file exists
    if not os.path.isfile(input_file_path):
        print(f"错误：找不到文件 '{input_file_path}'。请检查路径是否正确。")
    else:
        try:
            split_csv_with_dynamic_columns_and_row_uniqueness(input_file_path)
        except Exception as e:
            print(f"处理过程中发生错误: {e}")