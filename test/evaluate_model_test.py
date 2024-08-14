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

