import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import random
import warnings
import pathlib
import sys
import pickle
warnings.filterwarnings('ignore')

# ================ РАБОЧАЯ ПАПКА =================
model_dir = pathlib.Path(__file__).parent.resolve()
print(f"📁 Рабочая папка: {model_dir}")

# ==================== КОНФИГ ====================
CONFIG = {
    "vocab_size": 15000,
    "embed_dim": 384,
    "num_heads": 12,
    "num_layers": 6,
    "ffn_dim": 1024,
    "max_seq_len": 512,
    "device": "cpu"
}

# ==================== МОДЕЛЬ (ТА ЖЕ, ЧТО В TRAINING.PY) ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class SwiGLU(nn.Module):
    """SwiGLU — активация для FFN (как в LLaMA)"""
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = RMSNorm(embed_dim)
        
        self.ffn = SwiGLU(embed_dim, ffn_dim)  # <-- SwiGLU, а не SimpleFFN
        self.norm2 = RMSNorm(embed_dim)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

class LightLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.max_seq_len = config['max_seq_len']
        
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.position_embedding = nn.Embedding(config['max_seq_len'], config['embed_dim'])
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                config['embed_dim'],
                config['num_heads'],
                config['ffn_dim']
            ) for _ in range(config['num_layers'])
        ])
        
        self.norm_final = RMSNorm(config['embed_dim'])
        self.lm_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm_final(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=150, temperature=0.9, top_k=40, repetition_penalty=1.15):
        self.eval()
        with torch.no_grad():
            generated_tokens = []
            
            for _ in range(max_new_tokens):
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                
                logits = self(input_ids)
                logits = logits[:, -1, :] / max(temperature, 0.1)
                
                if repetition_penalty != 1.0 and generated_tokens:
                    unique_tokens = list(set(generated_tokens))
                    for token_id in unique_tokens:
                        if logits[0, token_id] < 0:
                            logits[0, token_id] *= repetition_penalty
                        else:
                            logits[0, token_id] /= repetition_penalty
                
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == 1:
                    break
            
            return input_ids

# ==================== ВОКАБУЛЯР ====================
def build_minimal_vocab():
    vocab = {
        '<PAD>': 0,
        '<EOS>': 1,
        '<UNK>': 2,
        '<BOS>': 3,
        '<USER>': 4,
        '<ASSISTANT>': 5,
        '<SEP>': 6,
        '<SYSTEM>': 7,
    }
    return vocab

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
def load_model_safe(model_path):
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        try:
            with open(model_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', pickle_module=pickle)
            return checkpoint
        except Exception as e2:
            print(f"❌ Альтернативный способ не сработал: {e2}")
            return None

# ==================== ЧАТ-БОТ ====================
class ChatBot:
    def __init__(self, model_path=None):
        self.device = torch.device('cpu')
        print(f"💻 Device: {self.device}")
        
        if model_path is None:
            model_paths = [
                model_dir / "final_model.pt",
                model_dir / "best_model.pt",
                model_dir / "checkpoints" / "model_epoch_5.pt",
                model_dir / "checkpoints" / "model_epoch_4.pt",
                model_dir / "checkpoints" / "model_epoch_3.pt",
                model_dir / "demo_model.pt",
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Найдена модель: {model_path}")
                    break
            else:
                print("❌ Модель не найдена!")
                self.model = None
                self.vocab = None
                return
        
        print(f"🤖 Загрузка модели: {model_path}")
        
        checkpoint = load_model_safe(model_path)
        
        if checkpoint is None:
            print("❌ Не удалось загрузить модель!")
            self.model = None
            self.vocab = None
            return
        
        try:
            # Извлекаем словарь
            self.vocab = checkpoint.get('vocab', {})
            if not self.vocab:
                print("⚠️ Словарь не найден, создаю минимальный...")
                self.vocab = build_minimal_vocab()
            
            # Обновляем конфиг из чекпоинта
            if 'config' in checkpoint:
                for key, value in checkpoint['config'].items():
                    if key in CONFIG:
                        CONFIG[key] = value
            
            CONFIG['vocab_size'] = len(self.vocab)
            
            # Определяем количество слоев из чекпоинта
            state_dict = checkpoint['model_state_dict']
            
            # Считаем количество слоев по ключам
            layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
            num_layers = 0
            for k in layer_keys:
                parts = k.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_num = int(parts[1])
                    if layer_num + 1 > num_layers:
                        num_layers = layer_num + 1
            
            if num_layers > 0:
                print(f"🔍 Определено слоев: {num_layers}")
                CONFIG['num_layers'] = num_layers
            
            # Определяем embed_dim
            if 'token_embedding.weight' in state_dict:
                embed_dim = state_dict['token_embedding.weight'].shape[1]
                CONFIG['embed_dim'] = embed_dim
                print(f"🔍 Определен embed_dim: {embed_dim}")
            
            # Определяем ffn_dim
            if 'layers.0.ffn.w1.weight' in state_dict:
                ffn_dim = state_dict['layers.0.ffn.w1.weight'].shape[0]
                CONFIG['ffn_dim'] = ffn_dim
                print(f"🔍 Определен ffn_dim: {ffn_dim}")
            
            print(f"📋 Итоговый CONFIG: {CONFIG}")
            
            # Создаем модель
            print("🏗️ Создаю модель...")
            self.model = LightLLM(CONFIG)
            
            # Загружаем веса с игнорированием несовпадающих ключей
            print("📥 Загружаю веса...")
            missing, unexpected = self.model.load_state_dict(
                state_dict, 
                strict=False  # <-- ВАЖНО: не строгий режим
            )
            
            if missing:
                print(f"⚠️ Пропущенные ключи: {len(missing)}")
                # Показываем только первые 5
                for k in missing[:5]:
                    print(f"   - {k}")
                if len(missing) > 5:
                    print(f"   ... и еще {len(missing)-5}")
            
            if unexpected:
                print(f"⚠️ Неожиданные ключи: {len(unexpected)}")
                for k in unexpected[:5]:
                    print(f"   - {k}")
                if len(unexpected) > 5:
                    print(f"   ... и еще {len(unexpected)-5}")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.idx_to_word = {v: k for k, v in self.vocab.items()}
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Модель загружена!")
            print(f"   Словарь: {len(self.vocab)} токенов")
            print(f"   Параметров: {total_params:,} ({total_params / 1e6:.2f}M)")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации модели: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.vocab = None
    
    def tokenize(self, text):
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get('<UNK>', 2))
        
        return tokens
    
    def generate_response(self, prompt, max_tokens=150, temperature=0.9, repetition_penalty=1.15):
        if self.model is None:
            return "❌ Модель не загружена"
        
        input_ids = [self.vocab.get('<BOS>', 3)]
        
        if '<USER>' in self.vocab:
            input_ids.append(self.vocab['<USER>'])
        
        input_ids.extend(self.tokenize(prompt))
        
        if '<ASSISTANT>' in self.vocab:
            input_ids.append(self.vocab['<ASSISTANT>'])
        
        if len(input_ids) > CONFIG['max_seq_len'] - 50:
            input_ids = input_ids[:CONFIG['max_seq_len'] - 50]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=40,
                    repetition_penalty=repetition_penalty
                )
            
            result = []
            assist_mode = False
            
            for idx in generated[0].tolist():
                word = self.idx_to_word.get(idx, '<UNK>')
                
                if word in ['<PAD>', '<EOS>', '<BOS>', '<USER>', '<SYSTEM>']:
                    if word == '<EOS>':
                        break
                    continue
                
                if word == '<ASSISTANT>':
                    assist_mode = True
                    continue
                
                if word == '<SEP>':
                    continue
                
                if assist_mode:
                    result.append(word)
            
            response = ' '.join(result)
            
            if len(response.strip()) < 3:
                responses = [
                    "Не, ну ты чё, я же токо учусь. Спроси чё попроще.",
                    "Вот хуйня, я ещё не до конца выучил этот вопрос.",
                    "Мне надо ещё пару гигов текста скормить, потом отвечу.",
                    "Я пока не знаю ответа на этот вопрос. Учись, брат!"
                ]
                return random.choice(responses)
            
            return response
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return "Пиздец, ошибка в генерации. Попробуй ещё раз."

# ==================== СОЗДАНИЕ ДЕМО ====================
def create_demo_model():
    demo_path = model_dir / "demo_model.pt"
    
    if os.path.exists(demo_path):
        print(f"✅ Демо-модель уже существует: {demo_path}")
        return demo_path
    
    print("🤖 Создаю демо-модель...")
    vocab = build_minimal_vocab()
    CONFIG['vocab_size'] = len(vocab)
    model = LightLLM(CONFIG)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': CONFIG
    }, demo_path)
    
    print(f"✅ Демо-модель создана: {demo_path}")
    return demo_path

# ==================== ЗАПУСК ====================
def main():
    print("=" * 70)
    print("🤖 FLORESTGPT — ЛОКАЛЬНАЯ LLM НА CPU")
    print("=" * 70)
    print(f"📁 Папка: {model_dir}")
    
    model_path = None
    possible_paths = [
        model_dir / "final_model.pt",
        model_dir / "best_model.pt",
        model_dir / "demo_model.pt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("\n⚠️ Модель не найдена!")
        choice = input("Создать демо-модель? (y/n): ").strip().lower()
        if choice == 'y':
            model_path = create_demo_model()
        else:
            print("❌ Без модели работать не буду.")
            return
    
    print("\n" + "=" * 70)
    bot = ChatBot(model_path)
    
    if bot.model is None:
        print("\n❌ Не удалось загрузить модель!")
        print("   Попробуй создать демо-модель.")
        return
    
    print("\n" + "=" * 70)
    print("💬 ЧАТ ЗАПУЩЕН")
    print("   exit  - выход")
    print("   temp 0.9  - изменить температуру")
    print("   rep 1.15  - изменить repetition penalty")
    print("=" * 70)
    
    temperature = 0.9
    repetition_penalty = 1.15
    
    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("👋 Пока, брат!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 1.5:
                        temperature = new_temp
                        print(f"🌡️ Температура: {temperature:.2f}")
                    else:
                        print("⚠️ Температура должна быть от 0.1 до 1.5")
                except:
                    print("⚠️ Использование: temp 0.9")
                continue
            
            if user_input.lower().startswith('rep '):
                try:
                    new_rep = float(user_input.split()[1])
                    if 1.0 <= new_rep <= 2.0:
                        repetition_penalty = new_rep
                        print(f"🔁 Repetition penalty: {repetition_penalty:.2f}")
                    else:
                        print("⚠️ Repetition penalty должен быть от 1.0 до 2.0")
                except:
                    print("⚠️ Использование: rep 1.15")
                continue
            
            print("🤔 Думаю...")
            response = bot.generate_response(
                user_input,
                max_tokens=150,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Пока, брат!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()