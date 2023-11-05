import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

# Define your data dimensions and other relevant parameters here
input_dim = 128  # Example input dimension, replace with your actual value
num_items = 3
num_attributes = 3
item_dim = 128
attribute_dim = 128
user_dim = 128
overall_dim = 128

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your GNN model class (Part 1)
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

# Modify your retrieval functions to use GNN (Part 1)
def retrieve_item_embeddings(item_id, gnn_model):
    # Implementation of item embedding retrieval using GNN
    # Replace with your actual graph data and edge_index
    graph_data = torch.randn(item_id.shape[0], input_dim).to(device)  # Replace with your actual graph data
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long).to(device)  # Replace with your actual edge_index
    item_embedding = gnn_model(graph_data, edge_index)
    return item_embedding

def retrieve_user_embeddings(user_id, gnn_model, user_item_interactions, user_social_connections):
    # Implementation of user embedding retrieval using GNN
    # Replace the random data and edge_index with your actual data
    
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
    # Replace with your actual graph data and edge_index
    graph_data = torch.randn(attribute_id.shape[0], input_dim).to(device)  # Replace with your actual graph data
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long).to(device)  # Replace with your actual edge_index
    attribute_embedding = gnn_model(graph_data, edge_index)
    return attribute_embedding

# Define the recommendation logic (Part 1)
def recommend_item_or_attribute(user_embedding, overall_embedding, item_embeddings, attribute_embeddings, K=5):
    item_scores = compute_preference_scores(item_embeddings, overall_embedding)
    attribute_scores = compute_preference_scores(attribute_embeddings, overall_embedding)
    
    if torch.max(item_scores) > torch.max(attribute_scores):
        # Recommend top K items
        _, top_item_indices = torch.topk(item_scores, k=K)
        recommended_items = [f"Item {idx}" for idx in top_item_indices]
        return recommended_items
    else:
        # Ask for an attribute
        return None  # Placeholder for attribute recommendation

# Define your optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training code for the model (Part 1)
def train_model(training_samples, model, optimizer, num_epochs, omega):
    # BPR loss minimization
    loss_function = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for BPR

    # Set the model in training mode
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_samples in training_samples:  # You may need to create batches depending on your data size
            user_id, positive_item_id, negative_item_id = batch_samples

            # Retrieve embeddings
            user_embedding = retrieve_user_embeddings(user_id, gnn_model, user_item_interactions, user_social_connections)
            positive_item_embedding = retrieve_item_embeddings(positive_item_id, gnn_model)
            negative_item_embedding = retrieve_item_embeddings(negative_item_id, gnn_model)

            # Calculate preference scores
            positive_score = torch.matmul(positive_item_embedding, overall_embedding)
            negative_score = torch.matmul(negative_item_embedding, overall_embedding)

            # Compute BPR loss
            loss = -torch.log(torch.sigmoid(positive_score - negative_score))
            
            # Zero gradients, backward pass, and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = total_loss / len(training_samples)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Return the trained model
    return model

# Define the augment_recommendation_model function (Part 1)
def augment_recommendation_model(training_samples, delta, model, omega):
    # Augment the recommendation model with counterfactual samples
    
    # Assuming delta is a dictionary containing updates to the model's parameters
    for param_name, param_value in delta.items():
        if param_name in model.state_dict():
            # Update the model's parameters with delta
            model.state_dict()[param_name] += omega * param_value
    
    return model

# Define the CECR algorithm (Part 2)
def CECR_algorithm(K, epsilon, gamma, learning_rate, lambda_val, omega):
    # Initialize embeddings, parameters, and training samples
    # Replace with your actual data and initialization logic
    user_embedding = torch.randn(user_dim).to(device)
    overall_embedding = torch.randn(overall_dim).to(device)
    item_embeddings = torch.randn(num_items, item_dim).to(device)
    attribute_embeddings = torch.randn(num_attributes, attribute_dim).to(device)
    
    while True:
        # Receive user feedback on item or attribute during the conversation
        # This is a placeholder for your actual user interaction logic
        user_feedback = input("Please provide feedback or type 'quit' to end the conversation: ")
        
        if user_feedback.lower() == 'quit':
            # Terminate the conversation if the user types 'quit'
            break
        
        # Counterfactual Explanation Generation
        delta = generate_counterfactual_explanation(user_embedding, item_embedding, epsilon, gamma, learning_rate, lambda_val)
        
        # Augmenting Recommendation via Counterfactual Samples
        model = augment_recommendation_model(training_samples, delta, model, omega)
        
        # Train the model with augmented data and delta parameter
        model = train_model(training_samples, model, optimizer, num_epochs, omega)
        
        # Recommendation logic
        recommended_items = recommend_item_or_attribute(user_embedding, overall_embedding, item_embeddings, attribute_embeddings, K)
        
        if recommended_items is not None:
            print("System: We recommend the following items:")
            for idx, item in enumerate(recommended_items, start=1):
                print(f"{idx}. {item}")
        else:
            print("System: Please provide feedback on attributes.")
        
        # Ask for further feedback or terminate the conversation based on user input
        user_input = input("Do you want more recommendations? (Type 'yes' or 'quit'): ")
        if user_input.lower() == 'quit':
            break

# User Simulator (Part 2)
# You may find other user simulator online, here just demo how to use user simulator
class UserSimulator:
    def __init__(self, items, attributes, user_representation, item_embedding, attribute_embedding):
        self.items = items
        self.attributes = attributes
        self.user_profile = {
            "user_representation": user_representation,
            "item_embedding": item_embedding,
            "attribute_embedding": attribute_embedding
        }

    def reset_user_profile(self):
        self.user_profile = {}

    def get_simulated_feedback(self, system_question):
        feedback = {}
        
        for question in system_question:
            if "attribute" in question.lower():
                attribute = random.choice(self.attributes)
                feedback[question] = attribute
            else:
                for item in self.items:
                    feedback[item] = random.choice(["like", "dislike"])
        
        return feedback

# Recommendation System (Part 2)
class RecommendationSystem:
    def __init__(self, user_embedding, overall_embedding, item_embeddings, attribute_embeddings):
        self.user_embedding = user_embedding
        self.overall_embedding = overall_embedding
        self.item_embeddings = item_embeddings
        self.attribute_embeddings = attribute_embeddings
        
    def get_recommendations(self, K):
        item_scores = torch.matmul(self.item_embeddings, self.overall_embedding)
        attribute_scores = torch.matmul(self.attribute_embeddings, self.overall_embedding)
        
        top_item_indices = torch.topk(item_scores, k=K).indices
        recommended_items = [f"Item {idx}" for idx in top_item_indices]
        
        return recommended_items

# Conversation Simulation (Part 2)
def simulate_conversation(UserSimulator, recommendation_system, K):
    print("Welcome to the Conversational Recommendation System!")
    
    while True:
        UserSimulator.reset_user_profile()
        
        for _ in range(3):
            system_question = ["What do you think about attribute X?", "How do you feel about attribute Y?"]
            
            user_feedback = UserSimulator.get_simulated_feedback(system_question)
            
            recommended_items = recommendation_system.get_recommendations(K)
            
            print("System: We recommend the following items:")
            for idx, item in enumerate(recommended_items, start=1):
                print(f"{idx}. {item}")
            
            user_input = input("System: Do you want to continue the conversation? (yes/no): ").strip().lower()
            if user_input != "yes":
                print("System: Conversation ended. Thank you!")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CECR Algorithm and Conversation")
    parser.add_argument("--K_values", nargs='+', type=int, default=[5], help="List of recommendation values (K)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value")
    parser.add_argument("--gamma", type=float, default=0.01, help="Gamma value")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lambda_val", type=float, default=0.1, help="Lambda value")
    parser.add_argument("--omega", type=float, default=0.01, help="Omega value")

    args = parser.parse_args()

    for K in args.K_values:
        CECR_algorithm(K, args.epsilon, args.gamma, args.learning_rate, args.lambda_val, args.omega)




# python cecr_algorithm.py --K_values 5 --epsilon 0.1 --gamma 0.01 --learning_rate 0.001 --lambda_val 0.1 --omega 0.01
