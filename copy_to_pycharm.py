import shutil
import os

# ✅ Full path to Jupyter's model directory (where .pkl files are saved)
source_folder = r"C:/Users/Gauri/machine-learning-projects/Fake-News-Detection/fake_news_app"

# ✅ Model files to copy
model_file = "fake_news_model.pkl"
vectorizer_file = "vectorizer.pkl"

# ✅ Full path to your PyCharm project folder (where app.py expects the .pkl files)
destination_folder = r"C:/Users/Gauri/PyCharmProjects/fake_news_app/fake_news_app"

# ✅ Ensure the destination exists
os.makedirs(destination_folder, exist_ok=True)

# 🔁 Copy files
shutil.copy(os.path.join(source_folder, model_file), os.path.join(destination_folder, model_file))
shutil.copy(os.path.join(source_folder, vectorizer_file), os.path.join(destination_folder, vectorizer_file))

print("✅ Model and Vectorizer copied successfully to PyCharm project!")
