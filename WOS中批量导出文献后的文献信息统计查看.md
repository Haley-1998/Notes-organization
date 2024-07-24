## 1、WOS读取文献信息
WOS要进行全信息的导出，需要先登录，选择WOS核心集，然后再进行检索。

![image](https://github.com/user-attachments/assets/12e63229-eb1e-421c-845f-c109ca02b0ce)
![image](https://github.com/user-attachments/assets/d99d8577-5041-4dc7-bf24-8e727d19fba3)
![image](https://github.com/user-attachments/assets/b66f5eb8-4020-452b-92b3-dd1e48fb1215)


## 2、整理文献信息进Excel
整理文献标题、年份、作者、关键词、摘要、期刊和被引用次数进excel表格。

使用的数据格式为WOS中导出的全信息的plain text file。

基于以下，后续可根据需求查找统计相关文献信息。
```python
import os
import pandas as pd
import re
from collections import Counter
 
# 目录路径
directory_path = r'C:\Users\1'
 
# 提取字段函数
def extract_field(record, field):
    pattern = re.compile(f'{field} (.+)')
    match = pattern.search(record)
    return match.group(1).strip() if match else None
 
# 初始化数据列表
data = []
 
# 初始化频率统计的字典
keyword_frequency = Counter()
 
# 遍历目录中的所有txt文件
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        records = content.split('\nER\n')  # 按照记录之间的分隔符切分内容
        
        for record in records:
            title = extract_field(record, 'TI')
            keywords = extract_field(record, 'DE')
            year = extract_field(record, 'PY')
            authors = extract_field(record, 'C1')
            journal = extract_field(record, 'SO')
            abstract = extract_field(record, 'AB')
            references = re.search(r'TC (\d+)', record)  # 使用正则表达式提取TC后面的数字
            references = references.group(1) if references else None
            
            if title and keywords and year and authors and journal and abstract and references:
                authors = re.findall(r'\b[A-Z][a-z-]+\b [A-Z][a-z-]+\b', authors)  # 提取作者信息
                first_three_authors = ', '.join(authors[:3])
                keywords_list = [kw.strip() for kw in keywords.split(';')]
                
                # 更新频率统计字典
                keyword_frequency.update(keywords_list)
                
                data.append([title, year, first_three_authors, ', '.join(keywords_list), abstract, journal, references])
 
# 创建 DataFrame
results_df = pd.DataFrame(data, columns=['Article Title', 'Year', 'First Three Authors', 'Keywords', 'Abstract', 'Journal', 'References'])
 
# 根据频率高低对关键词进行排序
sorted_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)
 
# 创建第二个工作表 DataFrame
keywords_df = pd.DataFrame(sorted_keywords, columns=['Keyword', 'Frequency'])
 
# 保存结果为Excel文件，路径为与原始文件相同的路径
excel_filename = "processed_abstracts.xlsx"
excel_path = os.path.join(directory_path, excel_filename)
 
# 将结果保存到Excel文件中的两个工作表
with pd.ExcelWriter(excel_path) as writer:
    results_df.to_excel(writer, index=False, sheet_name='Abstracts')
    keywords_df.to_excel(writer, index=False, sheet_name='Keywords')
 
print(f"Excel 文件已生成并保存到路径：{excel_path}。")
```
![image](https://github.com/user-attachments/assets/495ea121-a231-4b30-8943-27c8ee603673)
![image](https://github.com/user-attachments/assets/140f7821-cf92-4102-90c2-031c89cef256)


 

## 3、对摘要信息使用AI总结成一句话（运行较费时间）
相当于关键句，同样用于快速定位文章内容。
```python
import os
import pandas as pd
import re
from collections import Counter
from transformers import T5ForConditionalGeneration, T5Tokenizer
 
# 目录路径
directory_path = r'C:\Users\1'
 
# 提取字段函数
def extract_field(record, field):
    pattern = re.compile(f'{field} (.+)')
    match = pattern.search(record)
    return match.group(1).strip() if match else None
 
# 摘要生成函数
def generate_summary(text, model, tokenizer, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=50, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
 
# 初始化数据列表
data = []
 
# 初始化频率统计的字典
keyword_frequency = Counter()
 
# 加载T5模型和tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
 
# 遍历目录中的所有txt文件
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        records = content.split('\nER\n')  # 按照记录之间的分隔符切分内容
        
        for record in records:
            title = extract_field(record, 'TI')
            keywords = extract_field(record, 'DE')
            year = extract_field(record, 'PY')
            authors = extract_field(record, 'C1')  # 提取'C1'字段中的作者信息
            journal = extract_field(record, 'SO')
            abstract = extract_field(record, 'AB')
            references = re.search(r'TC (\d+)', record)  # 使用正则表达式提取TC后面的数字
            references = references.group(1) if references else None
            
            if title and keywords and year and authors and journal and abstract and references:
                # 生成摘要
                summarized_abstract = generate_summary(abstract, model, tokenizer)
                
                authors = re.findall(r'\b[A-Z][a-z-]+\b [A-Z][a-z-]+\b', authors)  # 提取作者信息
                first_three_authors = ', '.join(authors[:3])
                keywords_list = [kw.strip() for kw in keywords.split(';')]
                
                # 更新频率统计字典
                keyword_frequency.update(keywords_list)
                
                data.append([title, year, first_three_authors, ', '.join(keywords_list), summarized_abstract, journal, references])
 
# 创建 DataFrame
results_df = pd.DataFrame(data, columns=['Article Title', 'Year', 'First Three Authors', 'Keywords', 'Abstract', 'Journal', 'References'])
 
# 根据频率高低对关键词进行排序
sorted_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)
 
# 创建第二个工作表 DataFrame
keywords_df = pd.DataFrame(sorted_keywords, columns=['Keyword', 'Frequency'])
 
# 保存结果为Excel文件，路径为与原始文件相同的路径
excel_filename = "processed_abstracts_with_summary.xlsx"
excel_path = os.path.join(directory_path, excel_filename)
 
# 将结果保存到Excel文件中的两个工作表
with pd.ExcelWriter(excel_path) as writer:
    results_df.to_excel(writer, index=False, sheet_name='Abstracts')
    keywords_df.to_excel(writer, index=False, sheet_name='Keywords')
 
print(f"Excel 文件已生成并保存到路径：{excel_path}。")
```


## 4、英文翻译成中文
googletrans 库利用了Google Translate的翻译功能，因此在翻译准确性和质量上与Google Translate基本一致。
```python
import pandas as pd
from googletrans import Translator
import time
 
# 加载原始Excel文件
file_path = r'C:\Users\1\processed_abstracts1.xlsx'
df = pd.read_excel(file_path)
 
# 初始化翻译器
translator = Translator()
 
# 定义翻译函数
def translate_text(text, src='en', dest='zh-cn'):
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        return text  # 如果翻译出错，返回原文本
 
# 创建保存文件路径
translated_file_path = r'C:\Users\1\translated_processed_abstracts1.xlsx'
 
# 翻译DataFrame中的每个单元格，忽略第3列（First Three Authors）
translated_df = df.copy()
 
batch_size = 10  # 每10条数据保存一次文件
total_rows = len(translated_df)
start_time = time.time()
 
for i in range(0, total_rows, batch_size):
    batch_end = min(i + batch_size, total_rows)
    
    for column in translated_df.columns:
        if column != 'First Three Authors':
            translated_df.loc[i:batch_end-1, column] = translated_df.loc[i:batch_end-1, column].apply(
                lambda x: translate_text(x) if isinstance(x, str) else x
            )
    
    # 保存翻译后的DataFrame到新的Excel文件
    translated_df.to_excel(translated_file_path, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Processed rows {i} to {batch_end}, elapsed time: {elapsed_time:.2f} seconds")
 
print(f"Translated file saved to {translated_file_path}")
```

## 5、生成关键词云
三十个高频关键词一组，搞了近十年的看。（好像也没啥用，纯玩）
```python
import os
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
# 目录路径
directory_path = r'C:\Users\1'
 
# 提取字段函数
def extract_field(record, field):
    pattern = re.compile(f'{field} (.+)')
    match = pattern.search(record)
    return match.group(1).strip() if match else None
 
# 初始化数据列表
data = []
keywords_all = []
 
# 遍历目录中的所有txt文件
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        records = content.split('\n\n')
        
        for record in records:
            keywords = extract_field(record, 'DE')
            
            if keywords:
                # 将关键词转换为小写形式
                keywords_list = [kw.strip().lower() for kw in keywords.split(';')]
                data.append(keywords_list)
                keywords_all.extend(keywords_list)
 
# 计算关键词频率（不区分大小写）
keywords_counter = Counter(keywords_all)
 
# 根据频率排序关键词
sorted_keywords = sorted(keywords_counter.items(), key=lambda x: x[1], reverse=True)
 
# 分组关键词并生成词云
wordcloud = WordCloud(width=800, height=400, background_color='white')
 
# 第一个词云：展示前30个词
wordcloud.generate_from_frequencies(dict(sorted_keywords[:30]))
plt.figure(figsize=(10, 5), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud 1")
plt.savefig("wordcloud_1.png", bbox_inches='tight', dpi=300)
 
# 第二个词云：展示第31到60个词
wordcloud.generate_from_frequencies(dict(sorted_keywords[30:60]))
plt.figure(figsize=(10, 5), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud 2")
plt.savefig("wordcloud_2.png", bbox_inches='tight', dpi=300)
 
# 第三个词云：展示剩余的词
wordcloud.generate_from_frequencies(dict(sorted_keywords[60:]))
plt.figure(figsize=(10, 5), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud 3")
plt.savefig("wordcloud_3.png", bbox_inches='tight', dpi=300)
 
print("词云图已生成并保存。")
```
![image](https://github.com/user-attachments/assets/ac5226b1-4179-4f47-a067-af602e37f0bc)
![image](https://github.com/user-attachments/assets/9ca6bfa5-9901-4edc-a3d8-60d1a4a2d239)
![image](https://github.com/user-attachments/assets/4ec18f41-e85b-4919-8c47-f013de79cd57)


有缺啥的基本都直接pip install就行。
