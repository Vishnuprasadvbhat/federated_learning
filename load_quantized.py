import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model_path = "D:\\fed_up\\Quantized"


def load_model(model_path):

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  # Load model using the correct method for SafeTensors
  model = AutoModelForSequenceClassification.from_pretrained(
      model_path,
      use_safetensors=True  # Ensure this is still there for SafeTensors format
  ).to(torch.float32)

  ''' if its dynamic quantization
  
        model = torch.quantization.quantize_dynamic(
        BertForSequenceClassification.from_pretrained(model_path),
        {torch.nn.Linear},  # Apply dynamic quantization only on Linear layers
        dtype=torch.qint8)

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)
 '''
  # print(model)
  return model , tokenizer

if __name__ == '__main__':
        model, tokenizer = load_model(model_path=model_path)

        print(model)
        
        '''
        Reality.this
        In IMDb sentiment analysis datasets, the labels typically indicate whether a movie review 
        conveys a positive or negative sentiment. Here's how they are commonly represented:

        - **Positive Sentiment**: Labeled as "1" or "positive."
        - **Negative Sentiment**: Labeled as "0" or "negative."
        
        '''

# # define label mapping (adjust based on your specific model's labels)
# label_map = {0: "Negative", 1 : "Positive"}

# # texts = ["The movie was fantastic!", "The film was terrible."]
# # texts = ["The movie was okay okay!", "I dind't like this film."]
# texts = ['BAD', 'BAD', "BAD Movie ever", "BEST"]

# # Tokenize batch of texts
# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# # Forward pass
# with torch.no_grad():
#     outputs = model(**inputs)


# # Convert logits to probabilities using softmax
# probabilities = F.softmax(outputs.logits, dim=-1)
# print(probabilities)



# # Get the predicted class (with the highest probability)
# predicted_classes = torch.argmax(probabilities, dim=-1)
# print(predicted_classes)


# # Map predicted class indices to label names
# predicted_labels = [label_map[int(idx)] for idx in predicted_classes]
# print(predicted_labels)



# # Print the results
# for text, label, prob in zip(texts, predicted_labels, probabilities):
#     print(f"Text: '{text}' | Predicted Label: {label} | Probabilities: {prob.tolist()}")

'''
tensor([[0.5167, 0.4833],
        [0.5174, 0.4826]])
tensor([0, 0])
['Positive', 'Positive']
Text: 'The movie was okay okay!' | Predicted Label: Positive | Probabilities: [0.5167193412780762, 0.4832806885242462]
Text: 'I dind't like this film.' | Predicted Label: Positive | Probabilities: [0.5174391865730286, 0.48256081342697144]'''


'''

tensor([[0.4793, 0.5207],
        [0.4793, 0.5207],
        [0.4793, 0.5207],
        [0.4790, 0.5210]])
tensor([1, 1, 1, 1])
['Negative', 'Negative', 'Negative', 'Negative']
Text: 'BAD' | Predicted Label: Negative | Probabilities: [0.47927355766296387, 0.5207264423370361]
Text: 'BAD' | Predicted Label: Negative | Probabilities: [0.47927355766296387, 0.5207264423370361]
Text: 'BAD Movie ever' | Predicted Label: Negative | Probabilities: [0.4792587161064148, 0.5207412242889404]
Text: 'BEST MOVIE EVER' | Predicted Label: Negative | Probabilities: [0.47895047068595886, 0.5210494995117188]

'''