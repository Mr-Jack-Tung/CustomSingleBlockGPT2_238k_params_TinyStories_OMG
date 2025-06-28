# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 28 Jun 2025

import torch
import torch.nn as nn
import time
import os
# Import the custom tokenizer
from character_tokenizer import CharacterTokenizer

# Import the custom model and config
from custom_single_block_gpt2_model import SingleBlockGPT2ModelNoDepend, CustomGPT2Config

# Import the datasets library
from datasets import load_dataset

# Define the path to the trained single block model directory
# This is where the model and tokenizer will be saved and loaded from.
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_238k_params_TinyStories'
MODEL_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

# Main function to run training
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Dataset ---
    print("Loading TinyStories dataset...")
    try:
        # Using a small subset for demonstration to manage memory and training time
        # stream=True allows to iterate without downloading the whole dataset
        dataset = load_dataset("roneneldan/TinyStories", split='train', streaming=True)
        # Note: The dataset is a streaming dataset, so we can iterate over it without loading the entire dataset into memory at once.
        # dataset = dataset.shuffle()  # Shuffle the dataset for better training results.
        # Take a subset of the streaming dataset
        data_iterator = iter(dataset)
        # Let's take 10.000 stories for this training example
        num_stories = 10000
        stories = [next(data_iterator)['text'] for _ in range(num_stories)]
        text_data = " ".join(stories)
        print(f"Dataset loaded. Using {num_stories} stories for training.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is installed (`pip install datasets`).")
        exit()


    # --- 2. Load or Create Tokenizer ---
    # Try to load the tokenizer. If it doesn't exist, train a new one.
    tokenizer = None
    if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
        print(f"Attempting to load tokenizer from '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'...")
        try:
            tokenizer = CharacterTokenizer.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
            print(f"Tokenizer loaded successfully. Vocabulary size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"Could not load tokenizer: {e}. A new tokenizer will be trained.")
    
    if tokenizer is None:
        print("No existing tokenizer found or loading failed. Training a new one...")
        tokenizer = CharacterTokenizer()
        tokenizer.train(text_data)
        print(f"New character tokenizer trained with vocabulary size: {tokenizer.vocab_size}")


    # --- 3. Model and Config ---
    # Define model configuration using the loaded/trained tokenizer's vocab size
    config = CustomGPT2Config(model_type="small", vocab_size=tokenizer.vocab_size)
    # Set model parameters to match the desired architecture (20k params)
    # This must be consistent across training runs.
    config.n_positions = 128  # Block size
    config.n_embd = 128
    config.n_head = 4

    model = SingleBlockGPT2ModelNoDepend(config)

    # Load model state if it exists to continue training
    if os.path.exists(MODEL_STATE_DICT_PATH):
        print(f"Loading existing model state from '{MODEL_STATE_DICT_PATH}'...")
        try:
            # Load state dict, ensuring it's on the correct device
            model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=device))
            print("Model state loaded successfully. Training will continue from this point.")
        except Exception as e:
            print(f"Error loading model state: {e}. Starting with a newly initialized model.")
    else:
        print("No existing model state found. Starting with a newly initialized model.")

    model.to(device)

    print("\n--- Model Architecture ---")
    print(model)
    print(f"Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")


    # --- 4. Prepare Data for Training ---
    print("\nEncoding data...")
    # We'll process the data in chunks to avoid using too much memory
    encoded_data = tokenizer.encode(text_data)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    # Create chunks of block_size
    block_size = config.n_positions # 64
    # Drop the last chunk if it's not a full block
    num_chunks = len(data_tensor) // block_size
    print(f"Total characters: {len(text_data):,d}")
    print(f"Total tokens: {len(data_tensor):,d}")
    print(f"Block size: {block_size}")
    print(f"Number of training chunks: {num_chunks:,d}")


    # --- 5. Training Loop ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 5      # A few epochs for demonstration (5 epochs with 2-3 times is typical for small datasets)
    batch_size = 64     # Use batches to stabilize training

    model.train()
    print(f"\n--- Starting Training on TinyStories for {num_epochs} epochs ---")

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        
        # Shuffle the chunks at the beginning of each epoch
        shuffled_indices = torch.randperm(num_chunks)
        
        # Process data in batches
        for i in range(0, num_chunks, batch_size):
            # Get indices for the current batch
            batch_indices = shuffled_indices[i:i+batch_size]
            
            # Construct the batch from chunks
            batch_chunks = [data_tensor[idx*block_size : (idx+1)*block_size] for idx in batch_indices]
            
            # Skip if the last batch is empty
            if not batch_chunks:
                continue
                
            input_ids = torch.stack(batch_chunks).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs[0]

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True) # More efficient
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print progress occasionally
            num_batches = (num_chunks + batch_size - 1) // batch_size
            current_batch_num = i // batch_size + 1
            if current_batch_num % 100 == 0 or current_batch_num == num_batches:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {current_batch_num}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / ((num_chunks + batch_size - 1) // batch_size)
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"--- Epoch {epoch+1}/{num_epochs} Finished ---")
        print(f"Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.3f} seconds")


    # --- 6. Save the Model ---
    print("\n--- Saving the trained model ---")
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        # Save the model's state dictionary
        torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)

        # Save the character tokenizer
        tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        print(f"Trained model and tokenizer saved to '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'.")
    except Exception as e:
        print(f"Error saving trained model or tokenizer: {e}")

    print("\nTraining finished.")
    
'''
% python continuous_training_custom_single_block_gpt2_tinydataset.py
Using device: cpu
Loading TinyStories dataset...
Dataset loaded. Using 10000 stories for training.
No existing tokenizer found or loading failed. Training a new one...
New character tokenizer trained with vocabulary size: 94
No existing model state found. Starting with a newly initialized model.

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

Encoding data...
Total characters: 8,684,760
Total tokens: 8,684,760
Block size: 128
Number of training chunks: 67,849

--- Starting Training on TinyStories for 5 epochs ---
Epoch 1/5, Batch 100/1061, Loss: 2.5985
Epoch 1/5, Batch 200/1061, Loss: 2.4444
Epoch 1/5, Batch 300/1061, Loss: 2.3650
Epoch 1/5, Batch 400/1061, Loss: 2.3343
Epoch 1/5, Batch 500/1061, Loss: 2.2946
Epoch 1/5, Batch 600/1061, Loss: 2.2473
Epoch 1/5, Batch 700/1061, Loss: 2.1928
Epoch 1/5, Batch 800/1061, Loss: 2.1211
Epoch 1/5, Batch 900/1061, Loss: 2.0535
Epoch 1/5, Batch 1000/1061, Loss: 2.0139
Epoch 1/5, Batch 1061/1061, Loss: 2.0150
--- Epoch 1/5 Finished ---
Average Loss: 2.3307, Duration: 196.305 seconds
Epoch 2/5, Batch 100/1061, Loss: 1.9226
Epoch 2/5, Batch 200/1061, Loss: 1.8726
Epoch 2/5, Batch 300/1061, Loss: 1.8249
Epoch 2/5, Batch 400/1061, Loss: 1.8363
Epoch 2/5, Batch 500/1061, Loss: 1.8345
Epoch 2/5, Batch 600/1061, Loss: 1.7780
Epoch 2/5, Batch 700/1061, Loss: 1.7417
Epoch 2/5, Batch 800/1061, Loss: 1.6920
Epoch 2/5, Batch 900/1061, Loss: 1.6512
Epoch 2/5, Batch 1000/1061, Loss: 1.6425
Epoch 2/5, Batch 1061/1061, Loss: 1.6859
--- Epoch 2/5 Finished ---
Average Loss: 1.7780, Duration: 214.699 seconds
Epoch 3/5, Batch 100/1061, Loss: 1.6050
Epoch 3/5, Batch 200/1061, Loss: 1.5521
Epoch 3/5, Batch 300/1061, Loss: 1.6144
Epoch 3/5, Batch 400/1061, Loss: 1.5713
Epoch 3/5, Batch 500/1061, Loss: 1.5394
Epoch 3/5, Batch 600/1061, Loss: 1.4889
Epoch 3/5, Batch 700/1061, Loss: 1.4809
Epoch 3/5, Batch 800/1061, Loss: 1.5114
Epoch 3/5, Batch 900/1061, Loss: 1.4698
Epoch 3/5, Batch 1000/1061, Loss: 1.4731
Epoch 3/5, Batch 1061/1061, Loss: 1.3210
--- Epoch 3/5 Finished ---
Average Loss: 1.5285, Duration: 200.616 seconds
Epoch 4/5, Batch 100/1061, Loss: 1.4601
Epoch 4/5, Batch 200/1061, Loss: 1.4240
Epoch 4/5, Batch 300/1061, Loss: 1.4663
Epoch 4/5, Batch 400/1061, Loss: 1.4241
Epoch 4/5, Batch 500/1061, Loss: 1.4060
Epoch 4/5, Batch 600/1061, Loss: 1.4450
Epoch 4/5, Batch 700/1061, Loss: 1.4227
Epoch 4/5, Batch 800/1061, Loss: 1.3974
Epoch 4/5, Batch 900/1061, Loss: 1.3862
Epoch 4/5, Batch 1000/1061, Loss: 1.3547
Epoch 4/5, Batch 1061/1061, Loss: 1.4328
--- Epoch 4/5 Finished ---
Average Loss: 1.4211, Duration: 197.247 seconds
Epoch 5/5, Batch 100/1061, Loss: 1.3489
Epoch 5/5, Batch 200/1061, Loss: 1.3881
Epoch 5/5, Batch 300/1061, Loss: 1.3461
Epoch 5/5, Batch 400/1061, Loss: 1.3496
Epoch 5/5, Batch 500/1061, Loss: 1.3557
Epoch 5/5, Batch 600/1061, Loss: 1.3605
Epoch 5/5, Batch 700/1061, Loss: 1.3441
Epoch 5/5, Batch 800/1061, Loss: 1.3538
Epoch 5/5, Batch 900/1061, Loss: 1.3235
Epoch 5/5, Batch 1000/1061, Loss: 1.3609
Epoch 5/5, Batch 1061/1061, Loss: 1.3898
--- Epoch 5/5 Finished ---
Average Loss: 1.3604, Duration: 197.448 seconds

--- Saving the trained model ---
Trained model and tokenizer saved to 'TrainedSingleBlockGPT2_238k_params_TinyStories'.

Training finished.

# ==================== Continue training ==================

% python continuous_training_custom_single_block_gpt2_tinydataset.py
Using device: cpu
Loading TinyStories dataset...
Dataset loaded. Using 10000 stories for training.
Attempting to load tokenizer from 'TrainedSingleBlockGPT2_238k_params_TinyStories'...
Tokenizer loaded successfully. Vocabulary size: 94
Loading existing model state from 'TrainedSingleBlockGPT2_238k_params_TinyStories/single_block_model_state_dict.pth'...
Model state loaded successfully. Training will continue from this point.

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

Encoding data...
Total characters: 8,684,760
Total tokens: 8,684,760
Block size: 128
Number of training chunks: 67,849

--- Starting Training on TinyStories for 5 epochs ---
Epoch 1/5, Batch 100/1061, Loss: 2.1809
Epoch 1/5, Batch 200/1061, Loss: 1.9568
Epoch 1/5, Batch 300/1061, Loss: 1.8403
Epoch 1/5, Batch 400/1061, Loss: 1.7206
Epoch 1/5, Batch 500/1061, Loss: 1.6441
Epoch 1/5, Batch 600/1061, Loss: 1.6408
Epoch 1/5, Batch 700/1061, Loss: 1.5899
Epoch 1/5, Batch 800/1061, Loss: 1.5246
Epoch 1/5, Batch 900/1061, Loss: 1.5243
Epoch 1/5, Batch 1000/1061, Loss: 1.5103
Epoch 1/5, Batch 1061/1061, Loss: 1.4604
--- Epoch 1/5 Finished ---
Average Loss: 1.8040, Duration: 204.483 seconds
Epoch 2/5, Batch 100/1061, Loss: 1.4756
Epoch 2/5, Batch 200/1061, Loss: 1.4907
Epoch 2/5, Batch 300/1061, Loss: 1.4694
Epoch 2/5, Batch 400/1061, Loss: 1.4125
Epoch 2/5, Batch 500/1061, Loss: 1.3981
Epoch 2/5, Batch 600/1061, Loss: 1.4211
Epoch 2/5, Batch 700/1061, Loss: 1.4015
Epoch 2/5, Batch 800/1061, Loss: 1.4079
Epoch 2/5, Batch 900/1061, Loss: 1.4419
Epoch 2/5, Batch 1000/1061, Loss: 1.3858
Epoch 2/5, Batch 1061/1061, Loss: 1.3474
--- Epoch 2/5 Finished ---
Average Loss: 1.4391, Duration: 208.441 seconds
Epoch 3/5, Batch 100/1061, Loss: 1.3457
Epoch 3/5, Batch 200/1061, Loss: 1.3765
Epoch 3/5, Batch 300/1061, Loss: 1.3528
Epoch 3/5, Batch 400/1061, Loss: 1.3405
Epoch 3/5, Batch 500/1061, Loss: 1.3513
Epoch 3/5, Batch 600/1061, Loss: 1.3061
Epoch 3/5, Batch 700/1061, Loss: 1.3457
Epoch 3/5, Batch 800/1061, Loss: 1.3491
Epoch 3/5, Batch 900/1061, Loss: 1.3292
Epoch 3/5, Batch 1000/1061, Loss: 1.3613
Epoch 3/5, Batch 1061/1061, Loss: 1.3450
--- Epoch 3/5 Finished ---
Average Loss: 1.3587, Duration: 201.358 seconds
Epoch 4/5, Batch 100/1061, Loss: 1.2865
Epoch 4/5, Batch 200/1061, Loss: 1.3186
Epoch 4/5, Batch 300/1061, Loss: 1.2989
Epoch 4/5, Batch 400/1061, Loss: 1.3216
Epoch 4/5, Batch 500/1061, Loss: 1.3174
Epoch 4/5, Batch 600/1061, Loss: 1.2914
Epoch 4/5, Batch 700/1061, Loss: 1.2887
Epoch 4/5, Batch 800/1061, Loss: 1.2746
Epoch 4/5, Batch 900/1061, Loss: 1.2797
Epoch 4/5, Batch 1000/1061, Loss: 1.3189
Epoch 4/5, Batch 1061/1061, Loss: 1.4089
--- Epoch 4/5 Finished ---
Average Loss: 1.3127, Duration: 200.955 seconds
Epoch 5/5, Batch 100/1061, Loss: 1.3077
Epoch 5/5, Batch 200/1061, Loss: 1.3200
Epoch 5/5, Batch 300/1061, Loss: 1.3198
Epoch 5/5, Batch 400/1061, Loss: 1.3332
Epoch 5/5, Batch 500/1061, Loss: 1.2788
Epoch 5/5, Batch 600/1061, Loss: 1.3150
Epoch 5/5, Batch 700/1061, Loss: 1.2648
Epoch 5/5, Batch 800/1061, Loss: 1.2840
Epoch 5/5, Batch 900/1061, Loss: 1.2734
Epoch 5/5, Batch 1000/1061, Loss: 1.2548
Epoch 5/5, Batch 1061/1061, Loss: 1.2904
--- Epoch 5/5 Finished ---
Average Loss: 1.2811, Duration: 198.231 seconds

--- Saving the trained model ---
Trained model and tokenizer saved to 'TrainedSingleBlockGPT2_238k_params_TinyStories'.

Training finished.
'''