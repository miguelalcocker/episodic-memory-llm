import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import time

print("=" * 50)
print("🚀 VERIFICACIÓN DEL SISTEMA")
print("=" * 50)

# 1. VERIFICAR TORCH Y CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA no disponible - PROBLEMA!")

# 2. TEST TRANSFORMERS
print("\n" + "=" * 50)
print("🤖 TEST TRANSFORMERS")
print("=" * 50)

try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    print("✅ Transformers funcionando correctamente")
except Exception as e:
    print(f"❌ Error con transformers: {e}")

# 3. TEST MEMORIA GPU
print("\n" + "=" * 50)
print("💾 TEST MEMORIA GPU")
print("=" * 50)

if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        
        # Test con tensor grande
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = time.time()
        z = torch.mm(x, y)
        end_time = time.time()
        
        print(f"✅ Multiplicación matricial en GPU: {end_time - start_time:.4f}s")
        print(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error en test GPU: {e}")

print("\n" + "=" * 50)
print("🎉 VERIFICACIÓN COMPLETADA")
print("=" * 50)

if torch.cuda.is_available():
    print("✅ Todo listo para empezar el proyecto!")
else:
    print("❌ Necesitas configurar CUDA correctamente")
