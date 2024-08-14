import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset.text_dataset import TextDataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    labels = input_ids_padded.clone()  # For language modeling, labels are typically the same as input_ids
    return {'input_ids': input_ids_padded, 'labels': labels}

def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    dataset = TextDataset(tokenizer=tokenizer, file_path=args.data_path, max_length=tokenizer.model_max_length)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the text file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the pre-trained tokenizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    main(args)

