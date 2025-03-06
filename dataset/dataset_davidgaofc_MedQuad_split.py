from datasets import load_dataset, load_from_disk
import json

# first time you run this script, it will download the dataset and save it to disk
# ds = load_dataset("davidgaofc/MedQuad_split", cache_dir="F:\QA-System-Test\cache")
# ds.save_to_disk("F:/QA-System-Test/dataset")

# next time you run this script, it will load the dataset from disk
ds = load_from_disk("F:/QA-System-Test/dataset")
'''
dict_keys(['RL', 'RM_oos', 'SFT_train2', 'test', 'Shadow_oos', 'SFT_train1'])

SFT_train1
  Dataset({
      features: ['qtype', 'Question', 'Answer'],
      num_rows: 5742
  }) 

'''
def process_davidgaofc_MedQuad_split():
  davidgaofc_MedQuad_split = ['RL', 'RM_oos', 'SFT_train2', 'test', 'Shadow_oos', 'SFT_train1']
  for name in davidgaofc_MedQuad_split:
    output_file = f"F:/QA-System-Test/davidgaofc_MedQuad_split_{name}.json"
    keywords = ['fetal', 'heart rate', 'fetal heart rate']
    filtered_records = list()
    with open(output_file, 'w', encoding='utf-8') as f:
      for idx, item in enumerate(ds[name]):
          cur = item
          question = cur['Question'].lower().strip()
          answer = cur['Answer'].lower().strip()
          # 如果答案不是以句号结尾，则跳过该问答对
          if not answer.endswith('.'):
              continue
          if len(answer) > 512:
              last_period = answer.rfind('.', 0, 512)
              # 如果在前512字符中找不到句号，则跳过该问答对
              if last_period == -1:
                  continue
              answer = answer[:last_period+1].strip()
          # 如果答案包含关键词，则作为语料加入
          if any(keyword in text for keyword in keywords for text in [question, answer]):
              cur['Answer'] = answer
              filtered_records.append(cur)
    with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(filtered_records, f, ensure_ascii=False, indent=4)
      print('Finished writing to file:', output_file)
# uncomment it for filter and save to file
process_davidgaofc_MedQuad_split()


# with open("F:/QA-System-Test/filtered_records.json", 'r', encoding='utf-8') as f:
#   json_data = json.load(f)
#   print(json_data)