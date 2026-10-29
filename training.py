import os
os.environ['TRANSFORMERS_CACHE'] = 'path_to_cache'
os.environ['HF_HOME'] = 'hf_home'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import math
import random
import numpy as np
from tqdm import tqdm
import warnings
import pathlib
import sys
import gc
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.serialization

# ===== ОТКЛЮЧАЕМ ПРОКСИ =====
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

warnings.filterwarnings('ignore')

# ================ РАБОТАЕМ В ПАПКЕ MODEL =================
model_dir = pathlib.Path(__file__).parent.resolve()
print(f"📁 Рабочая папка: {model_dir}")

# ================ КОНФИГ ДЛЯ DEEPSEEK-R1-DISTILL-QWEN-7B (RTX 4060 8GB) =================
CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_save_dir": str(model_dir / "qwen_model"),
    
    # ===== QLoRA - МАКСИМАЛЬНАЯ ЭКОНОМИЯ ПАМЯТИ =====
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "target_modules": [  # ВСЕ ВНИМАНИЕ!!!
        "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
        "gate_proj", "up_proj", "down_proj",          # Feed-forward
    ],
    
    # ===== ДАННЫЕ =====
    "jsonl_path": str(model_dir / "dataset.jsonl"),
    "output_dir": str(model_dir / "qwen_checkpoints"),
    "max_seq_len": 512,  # УМЕНЬШЕНО для экономии памяти
    
    # ===== ОБУЧЕНИЕ =====
    "batch_size": 1,  # МИНИМАЛЬНЫЙ батч
    "gradient_accumulation_steps": 16,  # Эфф. батч = 16
    "epochs": 2,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "grad_clip": 1.0,
    "use_amp": True,
    "early_stopping_patience": 2,
    "use_gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": False,
    "report_to": "none",
    "logging_steps": 20,
    "save_steps": 100,
    "save_total_limit": 1,
}

print(f"💻 Устройство: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 VRAM: {total_vram:.1f} GB")

# Разрешаем pathlib для загрузки
try:
    torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath])
except:
    pass

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ================ ДАТАСЕТ С ОПТИМИЗАЦИЕЙ =================
class QwenDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        
        print(f"📂 Загрузка датасета из {jsonl_path}...")
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if 'messages' in entry:
                            text = self.format_dialog(entry['messages'])
                            if text:
                                self.data.append(text)
                        elif 'content' in entry:
                            self.data.append(entry['content'])
                    except:
                        continue
        except FileNotFoundError:
            print(f"⚠️ Файл {jsonl_path} не найден, создаю пример...")
            self.create_sample_data()
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if 'messages' in entry:
                            text = self.format_dialog(entry['messages'])
                            if text:
                                self.data.append(text)
                    except:
                        continue
        
        print(f"✅ Загружено {len(self.data)} диалогов")
        self._prepare_data()
    
    def create_sample_data(self):
        """Создает пример датасета если файл не найден"""
        sample_data = [
            {"messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Привет! Я FlorestGPT, твой виртуальный помощник."}]},
            {"messages": [{"role": "user", "content": "Что такое Python?"}, {"role": "assistant", "content": "Python — это высокоуровневый язык программирования."}]},
            {"messages": [{"role": "user", "content": "Расскажи шутку"}, {"role": "assistant", "content": "Почему программисты путают Хэллоуин и Рождество? Oct 31 = Dec 25!"}]},
        ]
        with open(CONFIG['jsonl_path'], 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def format_dialog(self, messages):
        parts = []
        for msg in messages:
            role = msg.get('role', 'user').lower()
            content = msg.get('content', '').strip()
            if not content:
                continue
            if role == 'user':
                parts.append(f"Пользователь: {content}")
            elif role == 'assistant':
                parts.append(f"FlorestGPT: {content}")
            else:
                parts.append(f"Пользователь: {content}")
        if not parts:
            return ""
        return "\n".join(parts)
    
    def _prepare_data(self):
        self.encoded_data = []
        assistant_tokens = self.tokenizer("FlorestGPT", add_special_tokens=False)['input_ids']
        
        for text in tqdm(self.data, desc="Токенизация"):
            lines = text.split("\n")
            user_text = lines[0] if lines else text
            assistant_text = lines[-1] if len(lines) > 1 else "Ответ на вопрос"
            
            user_text = user_text.replace("Пользователь: ", "").strip()
            assistant_text = assistant_text.replace("FlorestGPT: ", "").strip()
            
            formatted = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\nFlorestGPT: {assistant_text}<|im_end|>"
            
            tokens = self.tokenizer(
                formatted,
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
                return_tensors=None
            )
            
            input_ids = tokens['input_ids']
            labels = input_ids.copy()
            
            # Находим начало ответа ассистента
            assistant_start = -1
            for i in range(len(input_ids) - len(assistant_tokens)):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    assistant_start = i + len(assistant_tokens)
                    break
            
            if assistant_start > 0:
                labels[:assistant_start] = [-100] * assistant_start
            
            self.encoded_data.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': [1] * len(input_ids)
            })
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        
        pad_len = self.max_seq_len - len(item['input_ids'])
        if pad_len > 0:
            input_ids = item['input_ids'] + [self.tokenizer.pad_token_id] * pad_len
            labels = item['labels'] + [-100] * pad_len
            attention_mask = item['attention_mask'] + [0] * pad_len
        else:
            input_ids = item['input_ids'][:self.max_seq_len]
            labels = item['labels'][:self.max_seq_len]
            attention_mask = item['attention_mask'][:self.max_seq_len]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

# ================ ЗАГРУЗКА МОДЕЛИ С ОБРАБОТКОЙ ОШИБОК =================
def load_qwen_with_lora():
    print("=" * 70)
    print("🤖 ЗАГРУЗКА Qwen")
    print("=" * 70)
    
    device = torch.device(CONFIG['device'])
    
    # Загрузка токенизатора
    print("📚 Загрузка токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['model_name'],
            trust_remote_code=True
        )
        print("✅ Токенизатор загружен")
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        print("🔄 Попытка через зеркало...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['model_name'],
            trust_remote_code=True
        )
        print("✅ Токенизатор загружен через зеркало")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit квантование с МАКСИМАЛЬНОЙ экономией памяти
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage_dtype=torch.uint8,
    )
    
    print("🏗️ Загрузка модели 7B (4-бит)...")
    print("⏳ Это может занять 5-10 минут...")
    
    # Пробуем разные методы загрузки
    load_methods = [
        # Метод 1: Стандартная загрузка с offload на CPU
        lambda: AutoModelForCausalLM.from_pretrained(
            CONFIG['model_name'],
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map="auto",  # Автоматическое распределение
            offload_folder="./offload",  # Временная папка для offload
        ),
        # Метод 2: CPU-only загрузка (если не хватает VRAM)
        lambda: AutoModelForCausalLM.from_pretrained(
            CONFIG['model_name'],
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    ]
    
    model = None
    for i, load_fn in enumerate(load_methods, 1):
        try:
            print(f"🔄 Попытка {i}...")
            model = load_fn()
            print("✅ Модель загружена")
            break
        except Exception as e:
            print(f"❌ Попытка {i} не удалась: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            if i == len(load_methods):
                print("❌ Все попытки загрузки не удались.")
                print("💡 Рекомендации:")
                print("   1. Увеличьте файл подкачки Windows до 32GB")
                print("   2. Закройте все тяжелые программы")
                print("   3. Попробуйте модель 1.5B или 3B")
                raise
            continue
    
    if model is None:
        raise RuntimeError("Не удалось загрузить модель")
    
    # Включаем gradient checkpointing для экономии памяти
    if CONFIG['use_gradient_checkpointing']:
        print("🔧 Включение gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Подготовка для LoRA
    print("🔧 Подготовка для LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    # Настройка LoRA
    print("🔧 Настройка LoRA...")
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Перемещаем на GPU если нужно
    if CONFIG['device'] == 'cuda' and str(model.device) != 'cuda':
        model = model.to(device)
    
    # Очистка памяти
    if CONFIG['device'] == 'cuda':
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 VRAM занято: {allocated:.2f} GB")
        print(f"💾 VRAM зарезервировано: {reserved:.2f} GB")
        free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        print(f"💾 Свободно VRAM: {free:.2f} GB")
    
    print("=" * 70)
    return model, tokenizer

def collate_fn(batch):
    """Функция для объединения батча с динамической паддингом"""
    max_len = max([item['input_ids'].shape[0] for item in batch])
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        pad_len = max_len - item['input_ids'].shape[0]
        input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), 0)]))
        labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100)]))
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros(pad_len)]))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }

# ================ ОБУЧЕНИЕ С ОБРАБОТКОЙ ОШИБОК =================
def train():
    print("=" * 70)
    print("🚀 ОБУЧЕНИЕ 7B ЧЕРЕЗ LoRA")
    print("=" * 70)
    
    device = torch.device(CONFIG['device'])
    
    # Проверка доступной памяти перед началом
    if CONFIG['device'] == 'cuda':
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"💾 Доступно VRAM: {free_mem / 1024**3:.2f} GB")
        if free_mem < 2 * 1024**3:  # < 2GB
            print("⚠️ КРИТИЧЕСКИ МАЛО VRAM! Очистка памяти...")
            torch.cuda.empty_cache()
            gc.collect()
    
    try:
        model, tokenizer = load_qwen_with_lora()
    except Exception as e:
        print(f"❌ Критическая ошибка загрузки модели: {e}")
        print("\n💡 РЕШЕНИЕ:")
        print("1. Увеличьте файл подкачки Windows:")
        print("   - Панель управления → Система → Дополнительные параметры")
        print("   - Производительность → Параметры → Дополнительно")
        print("   - Виртуальная память → Изменить")
        print("   - Установите 32-64 GB")
        return None, None
    
    # Создание датасета
    dataset = QwenDataset(
        CONFIG['jsonl_path'],
        tokenizer,
        CONFIG['max_seq_len']
    )
    
    if len(dataset) == 0:
        print("❌ Датасет пуст!")
        return None, None
    
    # Разделение на train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Оптимизатор
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if CONFIG['use_amp'] and CONFIG['device'] == 'cuda' else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ:")
    print(f"   Эпох: {CONFIG['epochs']}")
    print(f"   Train батчей: {len(train_loader)}")
    print(f"   Эффективный batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"   Макс. длина: {CONFIG['max_seq_len']}")
    print("-" * 70)
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n📚 Эпоха {epoch + 1}/{CONFIG['epochs']}")
        
        # ===== TRAIN =====
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="Обучение")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Проверка на OOM
                if CONFIG['device'] == 'cuda':
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    if mem_alloc > 5.5:  # Если > 5.5 GB
                        print(f"⚠️ VRAM почти заполнена ({mem_alloc:.2f} GB), очистка...")
                        torch.cuda.empty_cache()
                
                if scaler:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask
                        )
                        loss = outputs.loss
                        loss = loss / CONFIG['gradient_accumulation_steps']
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask
                    )
                    loss = outputs.loss
                    loss = loss / CONFIG['gradient_accumulation_steps']
                    loss.backward()
                    
                    if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                epoch_loss += loss.item() * CONFIG['gradient_accumulation_steps']
                
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * CONFIG["gradient_accumulation_steps"]:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Периодическая очистка памяти
                if CONFIG['device'] == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM! Очистка памяти...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"✅ Train loss: {avg_train_loss:.4f}")
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"✅ Val loss: {avg_val_loss:.4f}")
        
        if CONFIG['device'] == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # ===== EARLY STOPPING =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            model.save_pretrained(CONFIG['output_dir'])
            print(f"⭐ Лучшая модель сохранена")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
        
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"🛑 Early stopping на эпохе {epoch + 1}")
            break
        
        # Тестовая генерация
        if (epoch + 1) % 1 == 0:
            print("\n🧪 Тестовая генерация:")
            test_generate(model, tokenizer, device)
        
        if CONFIG['device'] == 'cuda':
            print(f"💾 VRAM занято: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"📁 LoRA веса сохранены в: {CONFIG['output_dir']}")
    
    return model, tokenizer

# ================ ГЕНЕРАЦИЯ =================
def test_generate(model, tokenizer, device, prompt=None):
    """Тестовая генерация текста"""
    model.eval()
    
    if prompt is None:
        prompt = "Расскажи про Python"
    
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nFlorestGPT:"
    
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "FlorestGPT:" in response:
        response = response.split("FlorestGPT:")[-1].strip()
    
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 FlorestGPT: {response}")
    print("-" * 70)

# ================ ЧАТ =================
def chat():
    """Режим чата с FlorestGPT"""
    print("\n💬 РЕЖИМ ЧАТА С FLORESTGPT")
    print("Введите 'exit' для выхода")
    print("-" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_name'],
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if CONFIG['device'] == 'cuda' else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        from peft import PeftModel
        if os.path.exists(CONFIG['output_dir']):
            model = PeftModel.from_pretrained(model, CONFIG['output_dir'])
            print("✅ LoRA веса загружены")
        else:
            print("⚠️ LoRA веса не найдены, использую базовую модель")
        
        model.eval()
        
        device = torch.device(CONFIG['device'])
        model.to(device)
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    while True:
        user_input = input("\n👤 Вы: ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        
        formatted = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\nFlorestGPT:"
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "FlorestGPT:" in response:
            response = response.split("FlorestGPT:")[-1].strip()
        elif "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"🤖 FlorestGPT: {response}")

# ================ ЗАПУСК =================
if __name__ == "__main__":
    print("=" * 70)
    
    print("\n📌 ВЫБЕРИТЕ РЕЖИМ:")
    print("1. Дообучение 7B через LoRA")
    print("2. Чат с FlorestGPT")
    print("3. Информация о системе")
    print("-" * 70)
    
    choice = input("Ваш выбор (1, 2 или 3): ").strip()
    
    if choice == '1':
        try:
            model, tokenizer = train()
            if model is not None:
                print("\n✅ Обучение завершено!")
                
                chat_choice = input("\nНачать чат с FlorestGPT? (y/n): ").strip().lower()
                if chat_choice == 'y':
                    chat()
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("\n🔧 Рекомендации для решения проблемы с памятью:")
            print("1. Увеличьте файл подкачки Windows до 32-64 GB")
            print("2. Закройте все тяжелые программы (браузеры, IDE, игры)")
            print("3. Уменьшите max_seq_len до 256-512")
            print("4. Используйте модель 1.5B или 3B вместо 7B")
            import traceback
            traceback.print_exc()
            
    elif choice == '2':
        chat()
        
    elif choice == '3':
        print("\n📊 СИСТЕМНАЯ ИНФОРМАЦИЯ:")
        print(f"PyTorch версия: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {total_vram:.1f} GB")
            free = total_vram - torch.cuda.memory_allocated() / 1024**3
            print(f"Свободно VRAM: {free:.1f} GB")
        else:
            print("CUDA НЕ ДОСТУПНА")
        print(f"CPU: {os.cpu_count()} ядер")
        print("-" * 70)
            
    else:
        print("❌ Неверный выбор.")