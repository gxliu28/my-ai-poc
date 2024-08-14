import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def evaluate_model(input_text):
    model_path = 'output_dir'  # モデルとトークナイザーのパスを直接指定
    tokenizer_path = 'output_dir'

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None

    try:
        outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,  # Adjust max_length as needed
                pad_token_id=tokenizer.eos_token_id
                )
    except Exception as e:
        print(f"Error during text generation: {e}")
        return

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for generation")
    args = parser.parse_args()

    evaluate_model(args.input_text)

