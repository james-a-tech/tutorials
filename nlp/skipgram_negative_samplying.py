import numpy as np
from collections import defaultdict
import random

class SkipGram:
    def __init__(self, vocabulary_size, embedding_dim=100, window_size=5, learning_rate=0.025):
        self.vocab_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # Initialize word embeddings (W) and context embeddings (C)
        self.W = np.random.uniform(-1, 1, (vocabulary_size, embedding_dim))
        self.C = np.random.uniform(-1, 1, (vocabulary_size, embedding_dim))
        
        # Build word frequency dictionary for negative sampling
        self.word_freq = defaultdict(int)
        
    def generate_training_data(self, corpus):
        """
        Generate training pairs from corpus
        corpus: list of tokenized sentences (each sentence is a list of word indices)
        """
        training_data = []
        
        # Count word frequencies for negative sampling
        for sentence in corpus:
            for word in sentence:
                self.word_freq[word] += 1
                
        # Generate skip-gram pairs
        for sentence in corpus:
            for i, target_word in enumerate(sentence):
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(sentence), i + self.window_size + 1)
                
                # Get context words
                context_words = sentence[window_start:i] + sentence[i+1:window_end]
                
                for context_word in context_words:
                    training_data.append((target_word, context_word))
                    
        return training_data
    
    def negative_sampling(self, num_samples):
        """Generate negative samples based on word frequency"""
        # Convert frequencies to probabilities with 0.75 power (as in Word2Vec)
        total_freq = sum([freq ** 0.75 for freq in self.word_freq.values()])
        word_probs = {word: (freq ** 0.75) / total_freq 
                      for word, freq in self.word_freq.items()}
        
        words = list(word_probs.keys())
        probs = list(word_probs.values())
        
        return np.random.choice(words, size=num_samples, p=probs)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def train(self, training_data, epochs=5, negative_samples=5):
        """Train the skip-gram model"""
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for target_word, context_word in training_data:
                # Forward pass
                # Get target word embedding
                h = self.W[target_word]
                
                # Positive sample (actual context word)
                pos_score = np.dot(self.C[context_word], h)
                pos_prob = self.sigmoid(pos_score)
                
                # Negative samples
                neg_words = self.negative_sampling(negative_samples)
                neg_scores = np.dot(self.C[neg_words], h)
                neg_probs = self.sigmoid(neg_scores)
                
                # Calculate loss
                pos_loss = -np.log(pos_prob + 1e-10)
                neg_loss = -np.sum(np.log(1 - neg_probs + 1e-10))
                total_loss += pos_loss + neg_loss
                
                # Backward pass
                # Gradients for positive sample
                pos_error = pos_prob - 1
                grad_C_pos = pos_error * h
                grad_W_pos = pos_error * self.C[context_word]
                
                # Gradients for negative samples
                neg_errors = neg_probs
                grad_C_neg = np.outer(neg_errors, h)
                grad_W_neg = np.sum(neg_errors[:, np.newaxis] * self.C[neg_words], axis=0)
                
                # Update weights
                self.C[context_word] -= self.learning_rate * grad_C_pos
                self.W[target_word] -= self.learning_rate * grad_W_pos
                self.C[neg_words] -= self.learning_rate * grad_C_neg
                self.W[target_word] -= self.learning_rate * grad_W_neg
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss/len(training_data):.4f}")
    
    def get_word_embedding(self, word_idx):
        """Get embedding for a specific word"""
        return self.W[word_idx]

# Example usage
def main():
    # Sample corpus (word indices)
    corpus = [
        [0, 1, 2, 3],  # "I like to play"
        [4, 5, 6, 7],  # "Dog runs very fast"
        [8, 9, 2, 10]  # "Cat wants to sleep"
    ]
    
    # Vocabulary size (assuming we have words indexed 0-10)
    vocab_size = 11
    
    # Initialize model
    model = SkipGram(vocabulary_size=vocab_size, 
                    embedding_dim=50, 
                    window_size=2)
    
    # Generate training data
    training_data = model.generate_training_data(corpus)
    
    # Train the model
    model.train(training_data, epochs=10, negative_samples=5)
    
    # Get embedding for a word (example: word index 2)
    embedding = model.get_word_embedding(2)
    print(f"Embedding for word 2: {embedding[:5]}...")  # Showing first 5 dimensions

if __name__ == "__main__":
    main()