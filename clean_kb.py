import pandas as pd

# Load your current knowledge base
df = pd.read_csv('knowledge_base.csv')

# Remove duplicates (keeping the first occurrence)
df_clean = df.drop_duplicates(subset=['domain', 'company_type'], keep='first')

# Save cleaned CSV
df_clean.to_csv('knowledge_base_clean.csv', index=False)

print(f"Cleaned KB saved! Rows before: {len(df)}, after: {len(df_clean)}")
