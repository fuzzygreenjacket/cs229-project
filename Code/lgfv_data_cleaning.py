import pandas as pd

df_1 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 西藏-浙江.xlsx")
df_2 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 河北-内蒙古.xlsx")
df_3 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 宁夏-天津.xlsx")
df_4 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 中文示例.xlsx") 

new_df = pd.concat([df_1, df_2, df_3, df_4])

new_df.to_excel("LGFV_merged.xlsx")

df_merged = pd.read_excel("LGFV_merged.xlsx")
df_merged_filtered = df_merged[(df_merged["Listing Date"].notna()) & (df_merged["Delisting Date"].notna()) & (df_merged["Listing Date"] <= "2024-12-31") & (df_merged["Delisting Date"] >= "2018-01-01")].copy()
df_merged_filtered.to_excel("LGFV_merged_2018_to_2024.xlsx")