import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Read cleaned files
cleaned_path = r"C:\Users\stkam\PycharmProjects\port_delays_monitor\src\data_collection\cleaned_data\Piraeus"

print("\n=== CURRENT VESSELS ===")
df_current = pd.read_csv(f"{cleaned_path}/current_vessels.csv")
print("\nColumns:")
print(df_current.columns.tolist())
print("\nFirst 5 rows:")
print(df_current.head().to_string())

print("\n=== EXPECTED ARRIVALS ===")
df_expected = pd.read_csv(f"{cleaned_path}/expected_arrivals.csv")
print("\nColumns:")
print(df_expected.columns.tolist())
print("\nFirst 5 rows:")
print(df_expected.head().to_string())

print("\n=== PORT CALLS ===")
df_calls = pd.read_csv(f"{cleaned_path}/port_calls.csv")
print("\nColumns:")
print(df_calls.columns.tolist())
print("\nFirst 5 rows:")
print(df_calls.head().to_string())
