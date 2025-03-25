import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

def evaluate_model(model, tokenizer, dataset_name="imdb", batch_size=32):
    """
    Evaluates a quantized BERT model on a given dataset and plots accuracy over batches.
    
    Args:
        model: The quantized BERT model.
        tokenizer: The tokenizer for the model.
        dataset_name: The dataset to evaluate on (default is 'imdb').
        batch_size: Batch size for evaluation (default is 16).
    
    Returns:
        accuracy: Final accuracy of the model.
    """

    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Tokenize test data
    test_data = dataset["test"].map(tokenize_function, batched=True)

    # Custom collate function to pad sequences
    def collate_fn(batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        attention_masks = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)

        # Pad sequences to the same length within a batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)  # 0 for padding mask

        return input_ids, attention_masks, labels

    # Create DataLoader
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize variables
    accuracies = []
    batch_count = len(test_dataloader)
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_dataloader):
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            print('Accuracy at batch {}: {:.4f}'.format(batch_idx+1, correct / total), end='\r')
            accuracies.append(correct / total)  # Track accuracy over batches

    # Final Accuracy
    final_accuracy = correct / total
    print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")

    # Plot accuracy graph
    plt.plot(range(1, batch_count + 1), accuracies, label="Accuracy")
    plt.xlabel("Batch Number")
    plt.ylabel("Accuracy")
    plt.title("Saved Client Model Accuracy Over Batches")
    plt.legend()
    plt.show()

    return final_accuracy


if __name__ == '__main__':

    from load_quantized import load_model

    # Load the quantized model and tokenizer
    model_path = "D:\\fed_up\\NEW_INCRE\\client_model"
    # quatized_model_path = 'D:\\fed_up\\Quantized' Use this path, if you want to evaluate quanitzed model.
    model, tokenizer = load_model(model_path)

    # Run the evaluation
    final_acc = evaluate_model(model, tokenizer, dataset_name="imdb", batch_size=32)

    print(f"Final Accuracy: {final_acc:.4f}")

    '''
    Batch Size: 16 
    Accuracy at batch 1563: 0.7661
    Final Model Accuracy: 0.7661
    Final Accuracy: 0.7661

    Batch size: 32
    Accuracy at batch 782: 0.7906
    Final Model Accuracy: 0.7906
    Final Accuracy: 0.7906

    '''
    