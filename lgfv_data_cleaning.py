import pandas as pd

df_1 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 西藏-浙江.xlsx")
df_2 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 河北-内蒙古.xlsx")
df_3 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 宁夏-天津.xlsx")
df_4 = pd.read_excel("LGFV DEBT RISK/LGFV Bond Full list/城投债大全(2025-02-05) 中文示例.xlsx") 

new_df = pd.concat([df_1, df_2, df_3, df_4])

new_df.to_excel("LGFV_merged.xlsx")
