import torch
import transformers
import tokenizers
import sentencepiece
import datasets

print("="*60)

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Tokenizers:", tokenizers.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("Datasets:", datasets.__version__)

print("="*60)

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

print("="*60)