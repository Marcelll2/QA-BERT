from datasets import load_dataset, load_from_disk
import json
import copy

'''
pqa_artificial
  Dataset({
        features: ['pubid', 'question', 'context', 'long_answer', 'final_decision'],    
        num_rows: 211269
    })

pqa_labeled
  Dataset({
        features: ['pubid', 'question', 'context', 'long_answer', 'final_decision'],    
        num_rows: 1000
    })

pqa_unlabeled
  train: Dataset({
        features: ['pubid', 'question', 'context', 'long_answer'],
        num_rows: 61249
    })
'''
def download_data():
  ds = load_dataset("highnote/pubmed_qa", "pqa_artificial", cache_dir="F:/QA-System-Test/cache")
  print(f'Finished loading pqa_artificial')
  ds.save_to_disk("F:/QA-System-Test/dataset/pubmed_qa/pqa_artificial")
  print(f'Finished saving pqa_artificial')

  ds = load_dataset("highnote/pubmed_qa", "pqa_labeled", cache_dir="F:/QA-System-Test/cache")
  print(f'Finished loading pqa_labeled')
  ds.save_to_disk("F:/QA-System-Test/dataset/pubmed_qa/pqa_labeled")
  print(f'Finished saving pqa_labeled')

  ds = load_dataset("highnote/pubmed_qa", "pqa_unlabeled", cache_dir="F:/QA-System-Test/cache")
  print(f'Finished loading pqa_unlabeled')
  ds.save_to_disk("F:/QA-System-Test/dataset/pubmed_qa/pqa_unlabeled")
  print(f'Finished saving pqa_unlabeled')

def process(name='pqa_unlabeled'):
  ds = load_from_disk(f'F:/QA-System-Test/dataset/pubmed_qa/{name}')
  print('Loaded dataset:', name)

  output_file = f"F:/QA-System-Test/pubmed_qa_{name}.json"
  keywords = ['fetal', 'heart rate', 'fetal heart rate']
  filtered_records = list()
  with open(output_file, 'w', encoding='utf-8') as f:
    for idx, item in enumerate(ds['train']):
        # cur = item
        question = item['question'].lower().strip()
        # print(cur['context']['contexts'])
        context = item['context']['contexts'][0].lower().strip()
        answer = item['long_answer'].lower().strip()
        # 如果答案不是以句号结尾，则跳过该问答对
        if not answer.endswith('.'):
            continue
        if len(answer) > 512:
            last_period_answer = answer.rfind('.', 0, 512)
            # 如果在前512字符中找不到句号，则跳过该问答对
            if last_period_answer == -1:
                continue
            answer = answer[:last_period_answer+1].strip()

        if len(context) > 512:
            continue
            # last_period_context = context.rfind('.', 0, 512)
            # # 如果在前512字符中找不到句号，则跳过该问答对
            # if last_period_context == -1:
            #     continue
            # answer = answer[:last_period_context+1].strip()

        # 如果答案包含关键词，则作为语料加入
        if any(keyword in text for keyword in keywords for text in [question, context, answer]):
            # cur['Answer'] = answer
            if name == 'pqa_unlabeled':
              temp = {
                'question': question,
                'context': context,
                'answer': answer}
            else:
              temp = {
                'question': question,
                'context': context,
                'answer': answer,
                'final_decision': item['final_decision']}
            filtered_records.append(temp)
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_records, f, ensure_ascii=False, indent=4)
    print('Finished writing to file:', output_file)


if __name__ == "__main__":
  # download_data()

  ds = load_from_disk('F:/QA-System-Test/dataset/pubmed_qa/pqa_unlabeled')

  process('pqa_artificial')