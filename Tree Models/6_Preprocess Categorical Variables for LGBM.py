# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:26:55 2019

@author: jing.o.li
"""

# =============================================================================
# # Categorical columns
# cat_cols = data.select_dtypes(include=['object']).columns
# # List of categorical columns to recode
# catCols = ['Sex', 'Embarked', 'CL', 'CN', 'Surname', 'Title']
# # Recode
# for c in catCols:
#     # Convert column to pd.Categotical
#     full[c] = pd.Categorical(full[c])
#     # Extract the cat.codes and replace the column with these
#     full[c] = full[c].cat.codes
#     # Convert the cat codes to categotical...
#     full[c] = pd.Categorical(full[c])
# # Generate a logical index of categorical columns to maybe use with LightGBM later
# catCols = [i for i,v in enumerate(full.dtypes) if str(v)=='category']
# =============================================================================
