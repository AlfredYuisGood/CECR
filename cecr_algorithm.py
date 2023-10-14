import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define a simple GNN model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Modify your retrieval functions to use GNN
def retrieve_item_embeddings(item_id, gnn_model):
    # Implementation of item embedding retrieval using GNN
    # You will need to provide your graph data and convert it to PyTorch tensors
    # For simplicity, I'm using random data here as an example.
    graph_data = torch.randn(item_id.shape[0], input_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)  # Replace with your actual edge_index
    item_embedding = gnn_model(graph_data, edge_index)
    return item_embedding

def retrieve_user_embeddings(user_id, gnn_model, user_item_interactions, user_social_connections):
    # Implementation of user embedding retrieval using GNN
    # Use your graph data and GNN model here to consider user-item interactions and social connections.
    # Replace the random data and edge_index with your actual data.
    
    # Retrieve item embeddings for user-item interactions
    item_embeddings = []
    for item_id in user_item_interactions[user_id]:
        item_emb = retrieve_item_embeddings(item_id, gnn_model)
        item_embeddings.append(item_emb)
    item_embeddings = torch.stack(item_embeddings, dim=0)
    
    # Retrieve user embeddings for social connections
    user_embeddings = []
    for friend_id in user_social_connections[user_id]:
        friend_emb = retrieve_user_embeddings(friend_id, gnn_model, user_item_interactions, user_social_connections)
        user_embeddings.append(friend_emb)
    user_embeddings = torch.stack(user_embeddings, dim=0)
    
    # Combine item and user embeddings to form the user representation
    user_representation = torch.cat((item_embeddings.mean(dim=0), user_embeddings.mean(dim=0)))
    
    return user_representation

def retrieve_attribute_embeddings(attribute_id, gnn_model):
    # Implementation of attribute embedding retrieval using GNN
    # Similar to item and user embeddings, use your graph data and GNN model here.
    # Replace the random data and edge_index with your actual data.
    graph_data = torch.randn(attribute_id.shape[0], input_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)  # Replace with your actual edge_index
    attribute_embedding = gnn_model(graph_data, edge_index)
    return attribute_embedding

# Define the MLP function for rating-aware representation zu,i
def mlp(item_emb, rating_emb):
    concatenated_emb = np.concatenate((item_emb, rating_emb))
    return np.tanh(np.dot(concatenated_emb, mlp_weights) + mlp_bias)

# Define the aggregation function for user-item interactions
def aggregate_user_interactions(user_items, item_embeddings):
    attentions = []
    for item_id in user_items:
        item_emb = item_embeddings[item_id]
        rating_emb = user_items[item_id]['rating_emb']  # Use the rating-aware representation
        attention = mlp(item_emb, rating_emb)
        attentions.append(attention)
    attentions = np.exp(attentions) / np.sum(np.exp(attentions))
    user_representation = np.dot(attentions, item_embeddings)
    return user_representation

# Define the aggregation function for user-user social relations
def aggregate_user_social_relations(friends, user_embeddings):
    attentions = []
    for friend_id in friends:
        friend_emb = user_embeddings[friend_id]
        attention = mlp(friend_emb, user_embeddings[friend_id])  # Use the rating-aware representation
        attentions.append(attention)
    attentions = np.exp(attentions) / np.sum(np.exp(attentions))
    user_representation = np.dot(attentions, user_embeddings)
    return user_representation

# Define the aggregation function for attribute-item relations
def aggregate_attribute_items(attribute_items, item_embeddings):
    attentions = []
    for item_id in attribute_items:
        item_emb = item_embeddings[item_id]
        attention = mlp(item_emb, item_embeddings[item_id])  # Use the rating-aware representation
        attentions.append(attention)
    attentions = np.exp(attentions) / np.sum(np.exp(attentions))
    attribute_representation = np.dot(attentions, item_embeddings)
    return attribute_representation

# Define the function to update user representation
def update_user_representation(eu, eu_plus, eu_minus, W3, W4, b):
    C_u_minus = sigmoid(np.dot(W3, np.concatenate((eu_minus, eu, eu_minus * eu))) + b)
    C_u_plus = sigmoid(np.dot(W4, np.concatenate((eu_plus, eu, eu_plus * eu))) + b)
    eu_prime = eu * (C_u_minus - C_u_plus)
    return eu_prime

# Define the function to update item representation
def update_item_representation(ea_plus, ei_minus, W5, b):
    C_i_minus = sigmoid(np.dot(W5, np.concatenate((ea_plus, ei_minus, ea_plus * ei_minus))) + b)
    ei_minus_prime = ei_minus * C_i_minus
    return ei_minus_prime

# Define the function to calculate overall preference representation
def calculate_overall_preference(eu_prime, ei_minus_prime):
    e_overall = eu_prime - ei_minus_prime
    return e_overall

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize user representation eu, positive attribute representation ea_plus, negative attribute representation ea_minus
eu = np.zeros((representation_size,))
ea_plus = np.zeros((representation_size,))
ea_minus = np.zeros((representation_size,))

# Update user representation eu_prime
eu_prime = update_user_representation(eu, ea_plus, ea_minus, W3, W4, b)

# Initialize item representation ei_minus
ei_minus = np.zeros((representation_size,))

# Update item representation ei_minus_prime
ei_minus_prime = update_item_representation(ea_plus, ei_minus, W5, b)

# Calculate overall preference representation e_overall
e_overall = calculate_overall_preference(eu_prime, ei_minus_prime)

# Counterfactual Explanation Generation
def generate_counterfactual_explanation(user_embedding, item_embedding, attributes, epsilon, gamma, learning_rate, lambda_val):
    delta = np.zeros(len(attributes))  # Initialize change vector
    
    while True:
        # Compute ranking scores
        s_ui = compute_ranking_score(user_embedding, item_embedding)
        s_ui_delta = compute_ranking_score(user_embedding, item_embedding + delta)
        s_ui_k = compute_ranking_score(user_embedding, item_k_embedding)
        
        # Calculate Explanation Strength (ES)
        es = s_ui - s_ui_delta
        if es <= epsilon:
            break
        
        # Calculate Explanation Complexity (EC)
        ec = np.count_nonzero(delta)  # Number of attribute changes
        
        # Optimization for delta
        delta = delta - learning_rate * (lambda_val * np.log(sigmoid(epsilon)) / delta
                                         + (np.linalg.norm(delta)**2 + gamma * np.linalg.norm(delta, ord=1)) / delta)
    
    return delta

# Augmenting Recommendation via Counterfactual Samples
def augment_recommendation_model(training_samples, delta, model, omega):
    augmented_samples = []
    
    for u, i, i_k in training_samples:
        if delta is not None:
            augmented_samples.append((u, i, i_k))
    
    # Train the target model
    target_model = train_model(augmented_samples, model, omega)
    
    return target_model

# Recommendation
def recommend_item_or_attribute(user_embedding, overall_embedding, item_embeddings, attribute_embeddings, K=5):
    item_scores = compute_preference_scores(item_embeddings, overall_embedding)
    attribute_scores = compute_preference_scores(attribute_embeddings, overall_embedding)
    
    if np.max(item_scores) > np.max(attribute_scores):
        # Recommend top K items
        top_item_indices = np.argsort(item_scores)[::-1][:K]
        recommended_items = [items[idx] for idx in top_item_indices]
        return recommended_items
    else:
        # Ask for a random attribute
        random_attribute_index = random.randint(0, len(attributes)-1)
        recommended_attribute = attributes[random_attribute_index]
        return recommended_attribute

# Helper functions
def compute_ranking_score(user_embedding, embedding):
    return np.dot(user_embedding, embedding)

def compute_preference_scores(embeddings, overall_embedding):
    return np.dot(embeddings, overall_embedding)

# Training code for the model
def train_model(training_samples, model, omega):
    # BPR loss minimization
    loss = -np.sum(np.log(sigmoid(epsilon))) + omega * np.linalg.norm(model)**2
    
    # Training code for the model
    
    return trained_model

# Define the CECR algorithm
def CECR_algorithm(K, epsilon, gamma, learning_rate, lambda_val, omega):
    # Initialize embeddings, parameters, and training samples
    
    # Training Phase 1: Train the model using the original dataset
    model = train_model(training_samples, g, omega)
    
    while True:
        # Receive user feedback on item or attribute during the conversation
        
        # Counterfactual Explanation Generation
        delta = generate_counterfactual_explanation(user_embedding, item_embedding, attributes,
                                                   epsilon, gamma, learning_rate, lambda_val)
        
        # Augmenting Recommendation via Counterfactual Samples
        model = augment_recommendation_model(training_samples, delta, model, omega)
        
        # Recommendation
        recommended_item_or_attribute = recommend_item_or_attribute(user_embedding, overall_embedding,
                                                                    item_embeddings, attribute_embeddings, K)
        
        # Ask for further feedback or terminate the conversation based on recommended_item_or_attribute

# Parse command-line arguments
parser = argparse.ArgumentParser(description="CECR Algorithm")
parser.add_argument("--K_values", nargs='+', type=int, default=[5], help="List of recommendation values (K)")
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value")
parser.add_argument("--gamma", type=float, default=0.01, help="Gamma value")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--lambda_val", type=float, default=0.1, help="Lambda value")
parser.add_argument("--omega", type=float, default=0.01, help="Omega value")

args = parser.parse_args()

if __name__ == "__main__":
    K_values = args.K_values
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    lambda_val = args.lambda_val
    omega = args.omega

    for K in K_values:
        CECR_algorithm(K, epsilon, gamma, learning_rate, lambda_val, omega)


# python cecr_algorithm.py --K 5 --epsilon 0.1 --gamma 0.01 --learning_rate 0.001 --lambda_val 0.1 --omega 0.01
