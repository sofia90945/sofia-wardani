import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Dummy training data
menu_category = ['Beverages', 'Appetizers', 'Desserts', 'Main Course']
menu_item = ['Soda', 'Tiramisu', 'Chicken Alfredo', 'Spinach Artichoke Dip']
ingredients = ['confidential','Tomatoes', 'Basil', 'Garlic', 'Olive Oil']

# Create and train label encoders
le_menu_category = LabelEncoder().fit(menu_category)
le_menu_item = LabelEncoder().fit(menu_item)
le_ingredients = LabelEncoder().fit(ingredients)

# Save label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump({
        'menu_category': le_menu_category,
        'menu_item': le_menu_item,
        'ingredients': le_ingredients
    }, file)

# Dummy model training
model = RandomForestClassifier()
# Note: Replace this with your actual model training code and data
X_train = [[0, 0, 0, 10.0]]  # Example training data
y_train = [1]  # Example target
model.fit(X_train, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)
