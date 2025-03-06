from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from evaluate import load
import json
from datasets import Dataset
import numpy as np
import torch


def train(model, traindataset, evaldataset, num_train_epochs=3, use_peft=True):
  if use_peft:
    lora_config = LoraConfig(
      task_type=TaskType.QUESTION_ANS, 
      r=8,  
      lora_alpha=16, 
      lora_dropout=0.1,  
      target_modules=["query", "value"]  
      )
    model = get_peft_model(model, lora_config)
    print('PEFT model created')
    model.print_trainable_parameters()

  training_args = TrainingArguments(
    output_dir="./bert-qa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,  
    save_total_limit=2, 
    load_best_model_at_end=True,
    metric_for_best_model="f1_score"  
    )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=traindataset,
    eval_dataset=evaldataset,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics 
    )

  print("Model is on:", next(model.parameters()).device)
  print(f'Starting training for {num_train_epochs} epochs')
  trainer.train()
  print('Finished training')

def compute_metrics(eval_pred):
    f1_metric = load("f1")
    accuracy_metric = load("accuracy")  

    predictions, labels = eval_pred

    start_preds = np.argmax(predictions[0], axis=1)
    end_preds = np.argmax(predictions[1], axis=1)

    start_labels = labels[0]
    end_labels = labels[1]

    start_accuracy = accuracy_metric.compute(predictions=start_preds, references=start_labels)["accuracy"]
    end_accuracy = accuracy_metric.compute(predictions=end_preds, references=end_labels)["accuracy"]
    avg_accuracy = (start_accuracy + end_accuracy) / 2  

    f1_score = (f1_metric.compute(predictions=start_preds, references=start_labels, average='macro')["f1"] +
                f1_metric.compute(predictions=end_preds, references=end_labels, average='macro')["f1"]) / 2

    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "average_accuracy": avg_accuracy,
        "f1_score": f1_score
    }

def squad_data_convert():
  data_path = r'F:\QA-System-Test\dataset\augmented_dataset_filtered.json'
  with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  # Convert data to SQuAD format
  formatted_data = {
      "data": [
          {
              "title": "Fetal Heart Rate",
              "paragraphs": [
                  {
                      "context": entry["context"],
                      "qas": [
                          {
                              "question": entry["question"],
                              "id": str(i),
                              "answers": [
                                  {
                                      "text": entry["answer_text"],
                                      "answer_start": entry["start_char"]
                                  }
                              ]
                          }
                      ]
                  }
                  for i, entry in enumerate(raw_data)
              ]
          }
      ]
  }
  print('Data formatted to SQuAD format')

  save_path = r'F:\QA-System-Test\dataset\augmented_dataset_filtered_squad.json'
  with open(save_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)
    print('Data saved to', save_path)

def get_dataset(state="train"):
  data_path = f'F:\\QA-System-Test\dataset\\augmented_dataset_filtered_{state}.json'
  with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  dataset = Dataset.from_dict({
    "context": [entry["context"] for entry in raw_data],
    "question": [entry["question"] for entry in raw_data],
    "answer_text": [entry["answer_text"] for entry in raw_data],
    "start_char": [entry["start_char"] for entry in raw_data]
    })
  
  tokenized_dataset = dataset.map(preprocess_function, batched=True)
  return tokenized_dataset

def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i in range(len(examples["context"])):
        context = examples["context"][i]
        answer = examples["answer_text"][i]
        start_char = examples["start_char"][i]
        end_char = start_char + len(answer)

        # Tokenize context separately to get token offsets
        tokenized_context = tokenizer(context, return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = tokenized_context["offset_mapping"]

        # Convert character positions to token positions
        token_start, token_end = None, None
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                token_start = idx
            if start < end_char <= end:
                token_end = idx
                break

        if token_start is None or token_end is None:  # Handle cases where alignment fails
            token_start = 0
            token_end = 0

        start_positions.append(token_start)
        end_positions.append(token_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def test(question, context, checkpoint_path, use_peft=True):
    model_name = "bert-base-uncased"
    cache_dir = r"F:\QA-System-Test\model_cache"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)

    model = PeftModel.from_pretrained(model, checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")  
    inputs = {key: value.to(device) for key, value in inputs.items()}  

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores).item()  
    end_index = torch.argmax(end_scores).item()  

    if end_index < start_index:
        end_index = start_index

    start_char = offset_mapping[0][start_index][0]
    end_char = offset_mapping[0][end_index][1]

    extracted_answer = context[start_char:end_char]
    return extracted_answer


if __name__ == "__main__":
  cache_dir = r'F:\QA-System-Test\model_cache'
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

  '''uncomment for train'''
  # model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
  # train_dataset = get_dataset()
  # eval_dataset = get_dataset("eval")
  # train(model, train_dataset, eval_dataset, num_train_epochs=20, use_peft=True)

  '''uncomment for evaluation'''
  qa = [
     ["The normal fetal heart rate ranges from 110 to 160 beats per minute.", 
      "What is the normal fetal heart rate range?"],
     ["A high fetal heart rate can be caused by maternal fever or fetal distress.",
      "What causes a high fetal heart rate?"]
  ]
  idx = 1
  context = qa[idx][0]
  question = qa[idx][1]
  answer = test(question, context, r'F:\QA-System-Test\bert-qa\checkpoint-220', use_peft=True)
  print(f'Context: {context}')
  print(f"Question: {question}")
  print(f"Answer: {answer}")
  