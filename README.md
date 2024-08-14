## my-ai-poc
Build a Retrieval-Augmented Generation (RAG) system.

Components

- Train Model: Use train_model.py to train a text generation model.
- Evaluate Model: Use evaluate_model.py to generate and test responses.

Test Code Features:

- Train Model Test: Verify the training process and ensure the model learns from the provided data.
- Evaluate Model Test: Check the model's ability to generate relevant and accurate responses based on different inputs.

## Test Result
[![CI Pipeline](https://github.com/gxliu28/my-ai-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/gxliu28/my-ai-poc/actions/workflows/ci.yml)

## Installation 
- Setting up a virtual environment

  ```sh
  $ python3 -m venv venv
  $ source venv/bin/activate
  ```

- Installing required libraries  
  - Specify the necessary libraries in `requirements.txt`

    ```sh
    torch
    transformers
    datasets
    pytest
    ```

  - Run the following command to install the libraries

    ```sh
    $ pip install -r requirements.txt
    ```

- Creating the project configuration file `setup.py`

  ```py
  from setuptools import setup, find_packages

  setup(
      name="my-ai-poc",
      version="0.1.0",
      packages=find_packages(),
      install_requires=[
          "torch",
          "transformers",
          "datasets",
          "pytest"
      ],
  )
  ```

## Preparing the model

- Preparing the dataset
  - Preparing the training data `test/test_sample.txt`

    ```
    Hello, this is a test sentence.
    ```

  - Preparing the `dataset/text_dataset.py`
    <details>
      <summary>Click to expand</summary>

      ```py
      import torch
      from torch.utils.data import Dataset
      from transformers import GPT2Tokenizer

      class TextDataset(Dataset):
        def __init__(self, tokenizer, file_path, max_length=512):
            self.tokenizer = tokenizer
            self.file_path = file_path
            self.max_length = max_length
            self.examples = self._load_data()

        def _load_data(self):
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
            encodings = [self.tokenizer.encode(line, truncation=True, max_length=self.max_length) for line in lines]
            return encodings

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            encoding = self.examples[idx]
            item = {'input_ids': torch.tensor(encoding, dtype=torch.long)}
            item['labels'] = item['input_ids'].clone()
            return item
      ```
    </details>

- Creating a script to train the model `scripts/train_model.py`
  <details>
    <summary>Click to expand</summary>

    ```py
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
    ```
  </details>

## Writing test codes
- Writing tests for the data loader and training loop `test/train_model_test.py`
  <details>
    <summary>Click to expand</summary>

    ```py
    import pytest
    from transformers import GPT2Tokenizer
    from dataset.text_dataset import TextDataset

    @pytest.fixture
    def sample_dataset():
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return TextDataset(tokenizer, 'test/test_sample.txt', max_length=10)

    def test_text_dataset_len(sample_dataset):
        assert len(sample_dataset) == 1  # Assuming test_sample.txt has 1 line

    def test_text_dataset_item(sample_dataset):
        item = sample_dataset[0]
        assert 'input_ids' in item
        assert 'labels' in item
        assert len(item['input_ids']) <= 10
        assert len(item['labels']) <= 10
    ```
  </details>

- Writing tests code to evaluate the model `test/evaluate_model_test.py`
  <details>
    <summary>Click to expand</summary>

    ```py
    import subprocess

    def evaluate_model_test():
        input_text = "This is a test sentence."

        # コマンドを準備
        command = [
                'python', 'scripts/evaluate_model.py',
                '--input_text', input_text
                ]

        # コマンドを実行
        result = subprocess.run(command, capture_output=True, text=True)

        # エラーメッセージの確認
        if result.returncode != 0:
            print(f"Error Output: {result.stderr}")

        assert result.returncode == 0, f"Command failed with exit code {result.returncode}"

        assert "Generated text:" in result.stdout, "Output should contain 'Generated text:'"

        output_text = result.stdout.split("Generated text:")[1].strip()

        expected_phrase = "test sentence"
        assert expected_phrase in output_text, f"Expected phrase '{expected_phrase}' not found in generated text"

        print(f"Generated text: {output_text}")

    if __name__ == "__main__":
        evaluate_model_test()
    ```
  </details>

## Creating automated CI workflows for GitHub.com
```sh
name: CI

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run unit tests
      run: |
        pytest --maxfail=1 --disable-warnings -q

    - name: Run evaluate_model test
      run: |
        python test/evaluate_model_test.py
```

## Training
### Building a model using training data
```sh
$ python scripts/train_model.py --data_path test/test_sample.txt --output_dir output_dir --model_path gpt2 --tokenizer_path gpt2 --epochs 5 --batch_size 16 --learning_rate 5e-5
```

Then a model will be built in directory `output_dir`, in which there are the following files
```plaintext
ls output_dir/
config.json		merges.txt		special_tokens_map.json	vocab.json
generation_config.json	pytorch_model.bin	tokenizer_config.json
```

### Analyzing the model
The script `scripts/analyze_model.py' can analyze the model built by `scripts/train_model.py`.
```py
import torch
from transformers import AutoModel

# モデルのロード
model = AutoModel.from_pretrained('output_dir')

# モデルの構造とパラメータの表示
print(model)

for name, param in model.named_parameters():
    print(f'Parameter name: {name}, Shape: {param.shape}')

# パラメータの統計情報を表示
for name, param in model.named_parameters():
    print(f'Parameter name: {name}')
    print(f'Mean: {param.mean().item()}')
    print(f'Stdev: {param.std().item()}')
    print(f'Min: {param.min().item()}')
    print(f'Max: {param.max().item()}')
    print('---')
```

We can get the following information via the analyze script 
```py
$ python scripts/analyze_model.py
...omitted..
---
Parameter name: ln_f.weight
Mean: 1.5078967809677124
Stdev: 1.3910634517669678
Min: 0.004626442678272724
Max: 17.419328689575195
---
...omitted..
```

## Evaluating the model
The following script takes an `--input_text` parameter and uses the trained model to generate text.

<details>
  <summary>Click to expand</summary>

  ```py
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
  ```
</details>

### When there is only one line of training data
Assuming the training data consists of the following single line
```plaintext
Hello, this is a test sentence.
```

We can generate a text like this
```py
$ python scripts/evaluate_model.py --input_text 'This is a pen'
Generated text: This is a pen and paper test.

The test is a test sentence.

The test sentence is a test sentence.

The test sentence is a test sentence.

The test sentence is a test sentence.

The
```


### When there are multiple lines of training data
Assuming the training data consists of the following lines
```plaintext
Hello, this is a test sentence.
The study of physics explores the fundamental principles governing the natural world. Concepts such as force, energy, and momentum are central to understanding the behavior of objects and systems.
Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. This includes activities such as problem-solving, pattern recognition, and language understanding.
A balanced diet is essential for maintaining good health. It should include a variety of foods, such as fruits, vegetables, whole grains, and lean proteins, to provide all the necessary nutrients and energy for the body.
Global economic trends are influenced by a range of factors including trade policies, inflation rates, and consumer behavior. Understanding these trends helps businesses and governments make informed decisions.
The Renaissance was a period of great cultural and artistic achievement that began in Italy during the 14th century. It marked the transition from the Middle Ages to the modern era and brought significant advancements in art, science, and philosophy.
Classic literature often reflects the values and beliefs of the time in which it was written. Works by authors such as William Shakespeare, Jane Austen, and Charles Dickens offer insight into historical contexts and human nature.
Climate change is a critical issue that affects ecosystems and weather patterns worldwide. Reducing greenhouse gas emissions and transitioning to renewable energy sources are important steps in mitigating its impact.
```

It is necessary to rebuild the model using the following command again.
```py
$ python scripts/train_model.py --data_path test/test_sample.txt --output_dir output_dir --model_path gpt2 --tokenizer_path gpt2 --epochs 5 --batch_size 16 --learning_rate 5e-5
```

After rebuilding the model, we can generate a text like this
```py
$ python scripts/evaluate_model.py --input_text 'This is a pen'
Generated text: This is a pen that is designed to be used with a pen holder. It is designed to be used with a pen holder that is not a pen holder.

The pen holder is designed to be used with a pen holder that is not a
```

Another generating demo
```py
$ python scripts/evaluate_model.py --input_text 'This is a test sentence.'
Generated text: This is a test sentence. It is a sentence that is written in a sentence. It is a sentence that is written in a sentence. It is a sentence that is written in a sentence. It is a sentence that is written in a sentence.
```

## Trouble Shooting

### Error executing `pytest` after adding training data
In the beginning, there was only one line in `test/test_sample.txt`, this will lead to overfitting and poor generalization, as the model lacks sufficient examples to learn meaningful patterns. 

So, I added some lines to the training data as following
```plaintext
1. Hello, this is a test sentence.
2. The study of physics explores the fundamental principles governing the natural world. Concepts such as force, energy, and momentum are central to understanding the behavior of objects and systems.
3. Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. This includes activities such as problem-solving, pattern recognition, and language understanding.
4. A balanced diet is essential for maintaining good health. It should include a variety of foods, such as fruits, vegetables, whole grains, and lean proteins, to provide all the necessary nutrients and energy for the body.
5. Global economic trends are influenced by a range of factors including trade policies, inflation rates, and consumer behavior. Understanding these trends helps businesses and governments make informed decisions.
6. The Renaissance was a period of great cultural and artistic achievement that began in Italy during the 14th century. It marked the transition from the Middle Ages to the modern era and brought significant advancements in art, science, and philosophy.
7. Classic literature often reflects the values and beliefs of the time in which it was written. Works by authors such as William Shakespeare, Jane Austen, and Charles Dickens offer insight into historical contexts and human nature.
8. Climate change is a critical issue that affects ecosystems and weather patterns worldwide. Reducing greenhouse gas emissions and transitioning to renewable energy sources are important steps in mitigating its impact.
```

Then when I ran the following test command, an error occured.
<details>
  <summary>Click to expand</summary>

  ```plaintext
  $ pytest --maxfail=1 --disable-warnings -q
  F
  ========================================================= FAILURES ==========================================================
  ___________________________________________________ test_text_dataset_len ___________________________________________________

  sample_dataset = <dataset.text_dataset.TextDataset object at 0x12f566890>

      def test_text_dataset_len(sample_dataset):
  >       assert len(sample_dataset) == 1  # Assuming test_sample.txt has 1 line
  E       assert 9 == 1
  E        +  where 9 = len(<dataset.text_dataset.TextDataset object at 0x12f566890>)

  test/train_model_test.py:11: AssertionError
  ================================================== short test summary info ==================================================
  FAILED test/train_model_test.py::test_text_dataset_len - assert 9 == 1
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  1 failed, 1 warning in 3.50s
  ```
</details>

In order to solve the problem above, I changed the test code `test/train_model_test.py`
```py
- assert len(sample_dataset) == 1  # Assuming test_sample.txt has 1 line 
+ assert len(sample_dataset) == 8  # Assuming test_sample.txt has 8 line 
```

But another error occured
<details>
  <summary>Click to expand</summary>

  ```plaintext
  $ pytest --maxfail=1 --disable-warnings -q

  ========================================================== ERRORS ===========================================================
  _________________________________________ ERROR collecting test/train_model_test.py _________________________________________
  ImportError while importing test module '/Users/gxliu/github/main/my-ai-poc/test/train_model_test.py'.
  Hint: make sure your test modules/packages have valid Python names.
  Traceback:
  /usr/local/Cellar/python@3.11/3.11.9/Frameworks/Python.framework/Versions/3.11/lib/python3.11/importlib/__init__.py:126: in import_module
      return _bootstrap._gcd_import(name[level:], package, level)
  test/train_model_test.py:7: in <module>
      from dataset.text_dataset import TextDataset
  E   ModuleNotFoundError: No module named 'dataset'
  ================================================== short test summary info ==================================================
  ERROR test/train_model_test.py
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  1 error in 0.97s
  ```
</details>

So I follew the advice of ChatGPT and added the following lines to `test/train_model_test.py` before line `from dataset.text_dataset import TextDataset`


```py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

