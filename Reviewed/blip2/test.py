from tokenizers import Tokenizer
try:
    tok = Tokenizer.from_file("/raid5/lzy/project/weight/tokenizer.json")
    print("✅ tokenizer.json is valid")
except Exception as e:
    print("❌ Failed to load tokenizer.json:", e)