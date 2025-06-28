# Customed Single Block GPT-2 (238k params) for TinyStories with Continuous Training

This project demonstrates the training of a custom, single-block GPT-2-like language model on the TinyStories dataset. A key feature of this implementation is **continuous training**, allowing the training process to be stopped and resumed from the last saved checkpoint. The model has approximately **238,000 parameters**.

This version is adapted to train on the **TinyStories dataset**, demonstrating its ability to learn from a larger corpus of text and generate simple stories.

Đánh dấu một bước tiến trong việc huấn luyện một mô hình GPT-2 tùy chỉnh trên tập dữ liệu lớn hơn như TinyStories. Điểm nổi bật của dự án này là khả năng **huấn luyện liên tục (continuous training)**, cho phép lưu lại trạng thái và tiếp tục quá trình huấn luyện từ checkpoint gần nhất. Mô hình này có khoảng **238,000 tham số**.

Note: By downgrading to NumPy 1.26.4, we provided a version that PyTorch 2.2.2 can properly interact with Python 3.10~3.12


```
--- Model Architecture ---
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(94, 128)
  (wpe): Embedding(128, 128)
  (drop): Dropout(p=0.01, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=128, out_features=384, bias=True)
      (c_proj): Linear(in_features=128, out_features=128, bias=True)
      (attn_dropout): Dropout(p=0.01, inplace=False)
      (resid_dropout): Dropout(p=0.01, inplace=False)
    )
    (ln_2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=128, out_features=512, bias=True)
      (c_proj): Linear(in_features=512, out_features=128, bias=True)
      (dropout): Dropout(p=0.01, inplace=False)
    )
  )
  (ln_f): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=128, out_features=94, bias=False)
)
Number of trainable params: 238,976
```

## Project Structure

-   `single_block_gpt2_model.py`: Defines the model architecture, including the `CustomGPT2Config`, `SelfAttention`, `MLP`, `TransformerBlock`, and the main `SingleBlockGPT2ModelNoDepend` class.
-   `character_tokenizer.py`: Implements a simple character-level tokenizer.
-   `continuous_training_custom_single_block_gpt2_tinydataset.py`: Script for training the model.
-   `inference_custom_single_block_gpt2.py`: Script for generating text using a trained model.

## Getting Started

1.  Ensure you have Python and PyTorch installed. You can install PyTorch by following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  git clone https://github.com/Mr-Jack-Tung/CustomSingleBlockGPT2_238k_params_TinyStories_OMG
3.  Navigate to the `CustomSingleBlockGPT2_238k_params_TinyStories_OMG` directory in your terminal.

## Usage

### Training the Model on TinyStories

The `continuous_training_custom_single_block_gpt2_tinydataset.py` script is used for training the model on the `roneneldan/TinyStories` dataset from Hugging Face.

To train the model, first install the necessary libraries, then run the training script:

```bash
git clone https://github.com/Mr-Jack-Tung/CustomSingleBlockGPT2_238k_params_TinyStories_OMG
cd CustomSingleBlockGPT2_238k_params_TinyStories_OMG
python3.10 -m venv .venv
source .venv/bin/activate
pip install datasets==3.6.0 torch=2.2.2 numpy==1.26.4
python continuous_training_custom_single_block_gpt2_tinydataset.py # Example command
```

### Generating Text (Inference)

The `inference_custom_single_block_gpt2.py` script uses a trained model to generate text.

To run inference using a trained model, run the following command:

```bash
python inference_custom_single_block_gpt2.py # Example command
```

Example output:
```
Character tokenizer loaded from TrainedSingleBlockGPT2_238k_params_TinyStories
Trained model state dictionary loaded from TrainedSingleBlockGPT2_238k_params_TinyStories/single_block_model_state_dict.pth
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(94, 128)
  (wpe): Embedding(128, 128)
  (drop): Dropout(p=0.01, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=128, out_features=384, bias=True)
      (c_proj): Linear(in_features=128, out_features=128, bias=True)
      (attn_dropout): Dropout(p=0.01, inplace=False)
      (resid_dropout): Dropout(p=0.01, inplace=False)
    )
    (ln_2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=128, out_features=512, bias=True)
      (c_proj): Linear(in_features=512, out_features=128, bias=True)
      (dropout): Dropout(p=0.01, inplace=False)
    )
  )
  (ln_f): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=128, out_features=94, bias=False)
)

Input text: 'Once upon a time,'
Generating exactly 128 tokens...
Generated text: 'Once upon a time, they find her peach other blocks. Once upon a time the for started the recould and played the saw them. He was'
```

## Key Findings

-   **Model Minimization:** Successfully reduced the parameter count of a custom GPT-2-like model to **238k parameters** while still achieving accurate training results on the target data. This demonstrates the potential for creating very small yet effective language models for specific tasks.
-   **Training for Accuracy:** Achieving accurate training results with a minimized model requires careful consideration of the model architecture, training data, and training process.
