import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from peft import get_peft_model, LoraConfig
import json
import os

'''
dataset = {
    'train': [
        {'question': '...', 'image': 'images/image001.png', 'discription': 'clinical_reports/report001.txt'},
        {'question': '...', 'image': 'images/image002.png', 'discription': 'clinical_reports/report002.txt'},
        #...
    ],
    'val': [
        #...
    ],
    'test': [
        #...
    ]
}
'''
save_dir = r'clip_finetuned'

# 1. 数据集处理
class MedicalDataset(Dataset):
    def __init__(self, data, clip_processor, gpt_tokenizer, mode='train'):
        self.data = data
        self.clip_processor = clip_processor
        self.gpt_tokenizer = gpt_tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]      
        image = Image.open(item['image']).convert('RGB')
        image = self.transform(image)
        question = open(item['report'], 'r').read().strip()

        clip_inputs = self.clip_processor(
            text=[question],
            images=image.unsqueeze(0), 
            return_tensors="pt",
            padding=True
        )      
        gpt_inputs = self.gpt_tokenizer(
            item['discription'],
            return_tensors="pt",
            padding='max_length',
            max_length=512,
            truncation=True
        )
        
        return {
            'pixel_values': clip_inputs['pixel_values'].squeeze(0),
            'input_ids': clip_inputs['input_ids'].squeeze(0),
            'attention_mask': clip_inputs['attention_mask'].squeeze(0),
            'labels': gpt_inputs['input_ids'].squeeze(0)
        }

# 2. CLIP微调
class CLIPFinetuner:
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def contrastive_loss(self, image_features, text_features):
        logits = (text_features @ image_features.T) * torch.exp(torch.tensor(0.07))
        targets = torch.arange(len(image_features)).to(self.device)
        return F.cross_entropy(logits, targets)

    def train(self, train_loader, val_loader, epochs=10, save_path=save_dir):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                inputs = {
                    'pixel_values': batch['pixel_values'].to(self.device),
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                loss = self.contrastive_loss(image_features, text_features)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {
                        'pixel_values': batch['pixel_values'].to(self.device),
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    
                    outputs = self.model(**inputs)
                    loss = self.contrastive_loss(outputs.image_embeds, outputs.text_embeds)
                    val_loss += loss.item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# 3. 多模态融合模型
class MultimodalModel(nn.Module):
    def __init__(self, clip_path=None, use_lora=True):
        super().__init__()
        if not clip_path:
            clip_path = save_dir
        self.clip = CLIPModel.from_pretrained(clip_path)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        
        for param in self.clip.parameters():
            param.requires_grad = False
            
        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_dora=True,
                use_dora=True
            )
            self.gpt = get_peft_model(self.gpt, peft_config)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fusion = nn.Sequential(
            nn.Linear(512*2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 768)  # GPT-2的hidden_size
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            clip_outputs = self.clip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds
        attn_out, _ = self.cross_attention(image_features.unsqueeze(0), text_features.unsqueeze(0), text_features.unsqueeze(0))
        attn_out = attn_out.squeeze(0)
        
        combined = torch.cat([image_features, text_features], dim=-1)
        fused = self.fusion(combined)
        
        outputs = self.gpt(inputs_embeds=fused.unsqueeze(1))
        return outputs.logits


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_base = 'openai/clip-vit-base-patch32'
    gpt_base = 'gpt2'
    
    with open('data.json') as f:
        data = json.load(f)
    
    clip_processor = CLIPProcessor.from_pretrained(clip_base)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_base)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
    # 第一阶段：微调CLIP
    print("Stage 1: Finetuning CLIP")
    clip_trainer = CLIPFinetuner()
    
    train_dataset = MedicalDataset(data['train'], clip_processor, gpt_tokenizer)
    val_dataset = MedicalDataset(data['val'], clip_processor, gpt_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    clip_trainer.train(train_loader, val_loader, epochs=10, save_path='finetuned_clip')
    
    # 第二阶段：训练多模态模型
    print("\nStage 2: Training Multimodal Model")
    model = MultimodalModel('finetuned_clip').to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.gpt.parameters()},
        {'params': model.fusion.parameters()}
    ], lr=5e-5)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                'pixel_values': batch['pixel_values'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            logits = model(**inputs)
            labels = batch['labels'].to(device)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=gpt_tokenizer.pad_token_id
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), 'multimodal_model.pth')


def generate_report(image_path, question, model, clip_processor, gpt_tokenizer, device='cuda'):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    inputs = clip_processor(
        text=[question],
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        clip_outputs = model.clip(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds
        attn_out, _ = model.cross_attention(
            image_features.unsqueeze(0),
            text_features.unsqueeze(0),
            text_features.unsqueeze(0)
        )
        attn_out = attn_out.squeeze(0)        
        combined = torch.cat([image_features, text_features], dim=-1)
        fused_features = model.fusion(combined)
        
        current_ids = gpt_tokenizer.encode("[CLS]", return_tensors='pt').to(device)
        
        for _ in range(512): 
            # 使用融合特征和当前序列生成下一个token
            outputs = model.gpt(
                input_ids=current_ids,
                inputs_embeds=fused_features.unsqueeze(1)
            )            

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)            
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == gpt_tokenizer.eos_token_id:
                break
    
    return gpt_tokenizer.decode(current_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    # train
    main()
    
    # inference
    question = ''
    image_path = ''
    model = MultimodalModel()
    clipprocessor = CLIPProcessor()
    tokenizer = GPT2Tokenizer()
    generate_report(image_path=image_path, question=question, model=model, clip_processor=clipprocessor, gpt_tokenizer=tokenizer)