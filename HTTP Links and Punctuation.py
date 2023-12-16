#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import string
from collections import Counter


file_path = 'Stack Overflow 1000 Samples.xlsx'
df = pd.read_excel(file_path)


if 'Comment' not in df.columns:
    raise ValueError("The 'Comment' column does not exist in the DataFrame.")

def count_punctuation(text):
    
    punctuation_pattern = f'[{re.escape(string.punctuation)}]'
    return Counter(re.findall(punctuation_pattern, text))

def extract_http_links(text):
    
    http_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(http_pattern, text)

total_punctuation_count = Counter()
total_http_links_count = 0

for i, comment in enumerate(df['Comment']):
    punctuation_count = count_punctuation(str(comment))
    total_punctuation_count += punctuation_count
    
    links = extract_http_links(str(comment))
    total_http_links_count += len(links)
    
    if punctuation_count or links:
        print(f"Comment {i + 1} Analysis:")
        
        if punctuation_count:
            print("Punctuation Count:")
            for symbol, count in punctuation_count.items():
                print(f"{symbol}: {count}")
        
        if links:
            print("HTTP Links:")
            for link in links:
                print(link)
        
        print()
print("Total Punctuation Count:")
for symbol, count in total_punctuation_count.items():
    print(f"{symbol}: {count}")

print(f"Total HTTP links in the 'Comment' column: {total_http_links_count}")


# In[ ]:




