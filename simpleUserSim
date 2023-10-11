import random

# This file shows how to use UserSimulator

# User Simulator
class UserSimulator:
    def __init__(self, items, attributes):
        self.items = items
        self.attributes = attributes
        self.user_profile = {}  # User's historical preferences

    def reset_user_profile(self):
        self.user_profile = {}

    def get_simulated_feedback(self, system_question, recommended_items):
        # Simulate user feedback based on the system's question and recommendations
        feedback = {}
        
        for item in recommended_items:
            # Simulate user's reaction to recommended items (like or dislike)
            feedback[item] = random.choice(["like", "dislike"])
        
        # Simulate user's responses to system questions (attributes)
        for question in system_question:
            # Simulate user's response to attributes
            attribute = random.choice(self.attributes)
            feedback[question] = attribute
        
        return feedback

# Recommendation System (Placeholder)
class RecommendationSystem:
    def get_recommendations(self, user_profile):
        # Implement your recommendation algorithm here
        # Return a list of recommended items
        recommended_items = random.sample(user_profile["dislike"], k=3)
        return recommended_items

# Conversation Simulation
def simulate_conversation(user_simulator, recommendation_system):
    print("Welcome to the Conversational Recommendation System!")
    
    while True:
        # Reset the user profile at the beginning of the conversation
        user_simulator.reset_user_profile()
        
        # Start a conversation loop
        for _ in range(3):  # Simulate a conversation with 3 rounds
            # Assume the system asks a question about attributes
            system_question = ["What do you think about attribute X?", "How do you feel about attribute Y?"]
            
            # Simulate user's response
            user_feedback = user_simulator.get_simulated_feedback(system_question, user_simulator.items)
            
            # Update the user profile based on user feedback
            user_simulator.user_profile.update(user_feedback)
            
            # Get recommendations from the system
            recommended_items = recommendation_system.get_recommendations(user_simulator.user_profile)
            
            print("System: We recommend the following items:")
            for idx, item in enumerate(recommended_items, start=1):
                print(f"{idx}. {item}")
            
            # Simulate the next round of the conversation
            # ...
        
        # Ask the user if they want to continue the conversation
        user_input = input("System: Do you want to continue the conversation? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("System: Conversation ended. Thank you!")
            break

# Run the conversation simulation
items = ["item1", "item2", "item3"]
attributes = ["attribute1", "attribute2", "attribute3"]

user_simulator = UserSimulator(items, attributes)
recommendation_system = RecommendationSystem()

simulate_conversation(user_simulator, recommendation_system)
