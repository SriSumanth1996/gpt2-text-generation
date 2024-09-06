from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_text(prompt, max_length=250, num_return_sequences=1):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Create attention mask (1 for real tokens)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Generate text with sampling enabled
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Provide attention mask
        max_length=max_length,  # Set max_length to 250
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Control randomness
        top_k=50,  # Control sampling
        pad_token_id=tokenizer.eos_token_id  # Handle pad token id
    )

    # Decode and return the generated text
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Example usage
prompt = input("Enter your prompt: ")  # Input
generated_texts = generate_text(prompt)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}:\n{text}\n")
