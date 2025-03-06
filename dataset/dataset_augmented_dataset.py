import json

def process_augmented_dataset(state='train'):
    if state == 'train':
      input_path = r'F:\QA-System-Test\dataset\augmented_dataset_train.json'
      output_path = r'F:\QA-System-Test\dataset\augmented_dataset_filtered_train.json'
    elif state == 'eval':
      input_path = r'F:\QA-System-Test\dataset\augmented_dataset_eval.json'
      output_path = r'F:\QA-System-Test\dataset\augmented_dataset_filtered_eval.json'
    elif state == 'test':
      input_path = r'F:\QA-System-Test\dataset\augmented_dataset_test.json'
      output_path = r'F:\QA-System-Test\dataset\augmented_dataset_filtered_test.json'

    # Load the entire JSON array from the file
    with open(input_path, 'r', encoding='utf-8') as f_input:
        entries = json.load(f_input)

    # Process each entry and update with start_char and end_char
    filtered_entries = []
    for entry in entries:
        context = entry["context"]
        answer = entry["answer_text"]
        start_char = context.find(answer)
        end_char = start_char + len(answer) if start_char != -1 else -1

        entry["start_char"] = start_char
        entry["end_char"] = end_char

        filtered_entries.append(entry)

    # Write out the processed entries as a JSON array
    with open(output_path, 'w', encoding='utf-8') as f_output:
        json.dump(filtered_entries, f_output, ensure_ascii=False, indent=2)
        print('Finished writing to file:', output_path)

if __name__ == '__main__':
    process_augmented_dataset('train')
    process_augmented_dataset('eval')
    process_augmented_dataset('test')
