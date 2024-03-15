# # # import cv2
# # # import numpy as np
# # # import streamlit as st
# # # import pandas as pd
# # # from keras.models import model_from_json
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # from sklearn.metrics.pairwise import linear_kernel
# # # import random

# # # # Load the Kaggle datasets
# # # kaggle_books_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/booksmain.csv'
# # # kaggle_music_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/musicmain.csv'

# # # try:
# # #     kaggle_books_df = pd.read_csv(kaggle_books_path)
# # #     kaggle_music_df = pd.read_csv(kaggle_music_path)
# # # except pd.errors.ParserError as e:
# # #     st.error(f"Error parsing CSV file: {e}")
# # #     st.stop()

# # # # Load the emotion detection model
# # # json_file = open("facialemotionmodel.json", "r")
# # # model_json = json_file.read()
# # # json_file.close()
# # # model = model_from_json(model_json)
# # # model.load_weights("facialemotionmodel.h5")
# # # haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # # face_cascade = cv2.CascadeClassifier(haar_file)

# # # # Function to extract facial features and detect emotion
# # # def extract_features(image):
# # #     feature = np.array(image)
# # #     feature = feature.reshape(1, 48, 48, 1)
# # #     return feature / 255.0

# # # # Function to detect emotion from the captured image
# # # def detect_emotion(image):
# # #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# # #     emotions = []
# # #     try:
# # #         for (x, y, w, h) in faces:
# # #             face_image = gray[y:y + h, x:x + w]
# # #             face_image = cv2.resize(face_image, (48, 48))
# # #             img = extract_features(face_image)
# # #             pred = model.predict(img)
# # #             emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
# # #             emotions.append(emotion_label)
# # #         return emotions
# # #     except cv2.error:
# # #         return None

# # # # Function to get book recommendations based on emotion
# # # def get_book_recommendations(emotion):
# # #     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
# # #     return emotion_df['title'].tolist()

# # # # Function to get music recommendations based on emotion
# # # def get_music_recommendations(emotion):
# # #     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
# # #     return emotion_df['track_name'].tolist()

# # # # Convert emotion labels to the corresponding emotions
# # # EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# # # # Streamlit UI
# # # st.title("Rhythmic Reads Hub")

# # # # Add a button to capture image
# # # if st.button("Capture Image"):
# # #     # Capture image using webcam
# # #     webcam = cv2.VideoCapture(0)

# # #     # Capture and display the image
# # #     ret, frame = webcam.read()
# # #     if ret:
# # #         st.image(frame, channels="BGR")
# # #         captured_emotions = detect_emotion(frame)
# # #     webcam.release()

# # #     if captured_emotions:
# # #         # Convert emotion labels to emotions
# # #         detected_emotions = [EMOTIONS[label] for label in captured_emotions]

# # #         # Display detected emotions
# # #         st.write("Detected Emotions:", detected_emotions)

# # #         # Get book recommendation based on the first detected emotion
# # #         book_recommendation = []
# # #         for emotion in detected_emotions:
# # #             book_recommendation += get_book_recommendations(emotion)

# # #         # Display 5 random book recommendations
# # #         st.header("Recommended Books:")
# # #         if book_recommendation:
# # #             random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
# # #             for book in random_books:
# # #                 st.write(book)
# # #         else:
# # #             st.write("No books found for the detected emotions.")

# # #         # Get music recommendation based on the first detected emotion
# # #         music_recommendation = []
# # #         for emotion in detected_emotions:
# # #             music_recommendation += get_music_recommendations(emotion)

# # #         # Randomly select 5 music tracks
# # #         st.header("Music Recommendation:")
# # #         if music_recommendation:
# # #             random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
# # #             for music in random_music:
# # #                 st.write(music)
# # #         else:
# # #             st.write("No music found for the detected emotions.")
# # #     else:
# # #         st.error("No faces detected in the captured image.")

# # import cv2
# # import numpy as np
# # import streamlit as st
# # import pandas as pd
# # from keras.models import model_from_json
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import linear_kernel
# # import random

# # # Load the Kaggle datasets
# # kaggle_books_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/booksmain.csv'
# # kaggle_music_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/musicmain.csv'

# # try:
# #     kaggle_books_df = pd.read_csv(kaggle_books_path)
# #     kaggle_music_df = pd.read_csv(kaggle_music_path)
# # except pd.errors.ParserError as e:
# #     st.error(f"Error parsing CSV file: {e}")
# #     st.stop()

# # # Load the emotion detection model
# # json_file = open("facialemotionmodel.json", "r")
# # model_json = json_file.read()
# # json_file.close()
# # model = model_from_json(model_json)
# # model.load_weights("facialemotionmodel.h5")
# # haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # face_cascade = cv2.CascadeClassifier(haar_file)

# # # Function to extract facial features and detect emotion
# # def extract_features(image):
# #     feature = np.array(image)
# #     feature = feature.reshape(1, 48, 48, 1)
# #     return feature / 255.0

# # # Function to detect emotion from the captured image
# # def detect_emotion(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #     emotions = []
# #     try:
# #         for (x, y, w, h) in faces:
# #             face_image = gray[y:y + h, x:x + w]
# #             face_image = cv2.resize(face_image, (48, 48))
# #             img = extract_features(face_image)
# #             pred = model.predict(img)
# #             emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
# #             emotions.append(emotion_label)
# #         return emotions
# #     except cv2.error:
# #         return None

# # # Function to get book recommendations based on emotion
# # def get_book_recommendations(emotion):
# #     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
# #     return emotion_df['title'].tolist()

# # # Function to get music recommendations based on emotion
# # def get_music_recommendations(emotion):
# #     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
# #     return emotion_df['track_name'].tolist()

# # # Convert emotion labels to the corresponding emotions
# # EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# # # Streamlit UI
# # st.title("Rhythmic Reads Hub")

# # # Add a button to capture image
# # if st.button("Capture Image"):
# #     # Capture image using webcam
# #     webcam = cv2.VideoCapture(0)

# #     # Capture and display the image
# #     ret, frame = webcam.read()
# #     if ret:
# #         st.image(frame, channels="BGR")
# #         captured_emotions = detect_emotion(frame)
# #     webcam.release()

# #     if captured_emotions:
# #         # Convert emotion labels to emotions
# #         detected_emotions = [EMOTIONS[label] for label in captured_emotions]
# #         # Display detected emotions
# #         st.header("Detected Emotion:")
# #         st.write(detected_emotions[0])

# #         # Get book recommendation based on the first detected emotion
# #         book_recommendation = []
# #         for emotion in detected_emotions:
# #             book_recommendation += get_book_recommendations(emotion)

# #         # Display 5 random book recommendations
# #         st.header("Recommended Books:")
# #         if book_recommendation:
# #             random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
# #             for book in random_books:
# #                 st.write(book)
# #         else:
# #             st.write("No books found for the detected emotions.")

# #         # Get music recommendation based on the first detected emotion
# #         music_recommendation = []
# #         for emotion in detected_emotions:
# #             music_recommendation += get_music_recommendations(emotion)

# #         # Randomly select 5 music tracks
# #         st.header("Music Recommendation:")
# #         if music_recommendation:
# #             random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
# #             for music in random_music:
# #                 st.write(music)
# #         else:
# #             st.write("No music found for the detected emotions.")
# #     else:
# #         st.error("No faces detected in the captured image.")

# import cv2
# import numpy as np
# import streamlit as st
# import pandas as pd
# from keras.models import model_from_json
# import random

# # Load the Kaggle datasets
# kaggle_books_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/booksmain.csv'
# kaggle_music_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/musicmain.csv'

# # kaggle_books_path = 'C:/DesktopProjects/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/booksmain.csv'
# # kaggle_music_path = 'C:/DesktopProjects/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/musicmain.csv'

# try:
#     kaggle_books_df = pd.read_csv(kaggle_books_path)
#     kaggle_music_df = pd.read_csv(kaggle_music_path)
# except pd.errors.ParserError as e:
#     st.error(f"Error parsing CSV file: {e}")
#     st.stop()

# # Load the emotion detection model
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("facialemotionmodel.h5")
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# # Function to extract facial features and detect emotion
# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0

# # Function to detect emotion from the captured image
# def detect_emotion(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     emotions = []
#     try:
#         for (x, y, w, h) in faces:
#             face_image = gray[y:y + h, x:x + w]
#             face_image = cv2.resize(face_image, (48, 48))
#             img = extract_features(face_image)
#             pred = model.predict(img)
#             emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
#             emotions.append(emotion_label)
#         return emotions
#     except cv2.error:
#         return None

# # Function to get book recommendations based on emotion
# def get_book_recommendations(emotion):
#     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
#     return emotion_df['title'].tolist()

# # Function to get music recommendations based on emotion
# def get_music_recommendations(emotion):
#     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
#     return emotion_df['track_name'].tolist()

# # Convert emotion labels to the corresponding emotions
# EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# # Streamlit UI
# st.title("Rhythmic Reads Hub Chatbot")

# # Add radio button for image selection
# option = st.radio("Select an option:", ("Upload Image", "Capture Image"))

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)
#         captured_emotions = detect_emotion(image)
#         if captured_emotions:
#             # Convert emotion labels to emotions
#             detected_emotions = [EMOTIONS[label] for label in captured_emotions]

#             # Display detected emotions
#             st.header("Detected Emotion:")
#             st.write(detected_emotions[0])

#             # Get book recommendation based on the first detected emotion
#             book_recommendation = []
#             for emotion in detected_emotions:
#                 book_recommendation += get_book_recommendations(emotion)

#             # Display 5 random book recommendations
#             st.header("Recommended Books:")
#             if book_recommendation:
#                 random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#                 for book in random_books:
#                     st.write(book)
#             else:
#                 st.write("No books found for the detected emotions.")

#             # Get music recommendation based on the first detected emotion
#             music_recommendation = []
#             for emotion in detected_emotions:
#                 music_recommendation += get_music_recommendations(emotion)

#             # Randomly select 5 music tracks
#             st.header("Music Recommendation:")
#             if music_recommendation:
#                 random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#                 for music in random_music:
#                     st.write(music)
#             else:
#                 st.write("No music found for the detected emotions.")
#         else:
#             st.error("No faces detected in the uploaded image.")

# elif option == "Capture Image":
#     # Add a button to capture image
#     if st.button("Capture Image"):
#         # Capture image using webcam
#         webcam = cv2.VideoCapture(0)

#         # Capture and display the image
#         ret, frame = webcam.read()
#         if ret:
#             st.image(frame, channels="BGR")
#             captured_emotions = detect_emotion(frame)
#         webcam.release()

#         if captured_emotions:
#             # Convert emotion labels to emotions
#             detected_emotions = [EMOTIONS[label] for label in captured_emotions]

#             # Display detected emotions
#             st.header("Detected Emotion:")
#             st.write(detected_emotions[0])

#             # Get book recommendation based on the first detected emotion
#             book_recommendation = []
#             for emotion in detected_emotions:
#                 book_recommendation += get_book_recommendations(emotion)

#             # Display 5 random book recommendations
#             st.header("Recommended Books:")
#             if book_recommendation:
#                 random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#                 for book in random_books:
#                     st.write(book)
#             else:
#                 st.write("No books found for the detected emotions.")

#             # Get music recommendation based on the first detected emotion
#             music_recommendation = []
#             for emotion in detected_emotions:
#                 music_recommendation += get_music_recommendations(emotion)

#             # Randomly select 5 music tracks
#             st.header("Music Recommendation:")
#             if music_recommendation:
#                 random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#                 for music in random_music:
#                     st.write(music)
#             else:
#                 st.write("No music found for the detected emotions.")
#         else:
#             st.error("No faces detected in the captured image.")

#  

# import cv2
# import numpy as np
# import streamlit as st
# import pandas as pd
# from keras.models import model_from_json
# import random

# # Load the Kaggle datasets
# kaggle_books_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/booksmain.csv'
# kaggle_music_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/musicmain.csv'

# try:
#     kaggle_books_df = pd.read_csv(kaggle_books_path)
#     kaggle_music_df = pd.read_csv(kaggle_music_path)
# except pd.errors.ParserError as e:
#     st.error(f"Error parsing CSV file: {e}")
#     st.stop()

# # Load the emotion detection model
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("facialemotionmodel.h5")
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# # Function to extract facial features and detect emotion
# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0

# # Function to detect emotion from the image
# def detect_emotion(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     emotions = []
#     try:
#         for (x, y, w, h) in faces:
#             face_image = gray[y:y + h, x:x + w]
#             face_image = cv2.resize(face_image, (48, 48))
#             img = extract_features(face_image)
#             pred = model.predict(img)
#             emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
#             emotions.append(emotion_label)
#         return emotions
#     except cv2.error:
#         return None

# # Function to get book recommendations based on emotion
# def get_book_recommendations(emotion):
#     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
#     return emotion_df['title'].tolist()

# # Function to get music recommendations based on emotion
# def get_music_recommendations(emotion):
#     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
#     return emotion_df['track_name'].tolist()

# # Convert emotion labels to the corresponding emotions
# EMOTIONS = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

# # Streamlit UI
# st.title("Rhythmic Reads Hub")

# # Add radio button for image selection
# option = st.radio("Select an option:", ("Upload Image", "Capture Image", "Select Emotion"))

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)
#         captured_emotions = detect_emotion(image)
#         if captured_emotions:
#             # Convert emotion labels to emotions if they are valid
#             detected_emotions = [EMOTIONS[label] for label in captured_emotions if label in EMOTIONS]

#             if detected_emotions:
#                 # Display detected emotions
#                 st.header("Detected Emotions:")
#                 for emotion in detected_emotions:
#                     st.success(emotion)

#                 # Get book recommendation based on the detected emotions
#                 book_recommendation = []
#                 for emotion in detected_emotions:
#                     book_recommendation += get_book_recommendations(emotion)

#                 # Display 5 random book recommendations
#                 st.header("Recommended Books:")
#                 if book_recommendation:
#                     random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#                     for book in random_books:
#                         st.write(book)
#                 else:
#                     st.write("No books found for the detected emotions.")

#                 # Get music recommendation based on the detected emotions
#                 music_recommendation = []
#                 for emotion in detected_emotions:
#                     music_recommendation += get_music_recommendations(emotion)

#                 # Display music recommendation
#                 st.header("Music Recommendation:")
#                 if music_recommendation:
#                     random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#                     for music in random_music:
#                         st.write(music)
#                 else:
#                     st.write("No music found for the detected emotions.")
#             else:
#                 st.error("Invalid emotions detected.")
#         else:
#             st.error("No faces detected in the uploaded image.")

# elif option == "Capture Image":
#     # Add a button to capture image
#     if st.button("Capture Image"):
#         # Capture image using webcam
#         webcam = cv2.VideoCapture(0)

#         # Capture and display the image
#         ret, frame = webcam.read()
#         if ret:
#             st.image(frame, channels="BGR")
#             captured_emotions = detect_emotion(frame)
#         webcam.release()

#         if captured_emotions:
#             # Convert emotion labels to emotions if they are valid
#             detected_emotions = [EMOTIONS[label] for label in captured_emotions if label in EMOTIONS]

#             if detected_emotions:
#                 # Display detected emotions
#                 st.header("Detected Emotions:")
#                 for emotion in detected_emotions:
#                     st.success(emotion)

#                 # Get book recommendation based on the detected emotions
#                 book_recommendation = []
#                 for emotion in detected_emotions:
#                     book_recommendation += get_book_recommendations(emotion)

#                 # Display 5 random book recommendations
#                 st.header("Recommended Books:")
#                 if book_recommendation:
#                     random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#                     for book in random_books:
#                         st.write(book)
#                 else:
#                     st.write("No books found for the detected emotions.")

#                 # Get music recommendation based on the detected emotions
#                 music_recommendation = []
#                 for emotion in detected_emotions:
#                     music_recommendation += get_music_recommendations(emotion)

#                 # Display music recommendation
#                 st.header("Music Recommendation:")
#                 if music_recommendation:
#                     random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#                     for music in random_music:
#                         st.write(music)
#                 else:
#                     st.write("No music found for the detected emotions.")
#             else:
#                 st.error("Invalid emotions detected.")
#         else:
#             st.error("No faces detected in the captured image.")

# elif option == "Select Emotion":
#     selected_emotion = st.selectbox("Select an emotion:", list(EMOTIONS.values()))

#     # Get book recommendation based on the selected emotion
#     book_recommendation = get_book_recommendations(selected_emotion)

#     # Display 5 random book recommendations
#     st.header("Recommended Books:")
#     if book_recommendation:
#         random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#         for book in random_books:
#             st.write(book)
#     else:
#         st.write("No books found for the selected emotion.")

#     # Get music recommendation based on the selected emotion
#     music_recommendation = get_music_recommendations(selected_emotion)

#     # Randomly select 5 music tracks
#     st.header("Music Recommendation:")
#     if music_recommendation:
#         random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#         for music in random_music:
#             st.write(music)
#     else:
#         st.write("No music found for the selected emotion.")

import cv2
import numpy as np
import streamlit as st
import pandas as pd
from keras.models import model_from_json
import random

# Load the Kaggle datasets
kaggle_books_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/booksmain.csv'
kaggle_music_path = 'C:/Users/shreya binod choubey/Desktop/MusicAndBookRecommender/Bookmusic recommd/musicmain.csv'

try:
    kaggle_books_df = pd.read_csv(kaggle_books_path)
    kaggle_music_df = pd.read_csv(kaggle_music_path)
except pd.errors.ParserError as e:
    st.error(f"Error parsing CSV file: {e}")
    st.stop()

# Load the emotion detection model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract facial features and detect emotion
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to detect emotion from the captured image
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []
    try:
        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
            emotions.append(emotion_label)
        return emotions
    except cv2.error:
        return None

# Function to get book recommendations based on emotion
def get_book_recommendations(emotion):
    emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
    return emotion_df['title'].tolist()

# Function to get music recommendations based on emotion
def get_music_recommendations(emotion):
    emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
    return emotion_df['track_name'].tolist()

# Convert emotion labels to the corresponding emotions
EMOTIONS = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Streamlit UI
st.title("Rhythmic Reads Hub")

# Add radio button for image selection
option = st.radio("Select an option:", ("Upload Image", "Capture Image", "Select Emotion"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        captured_emotions = detect_emotion(image)
        if captured_emotions:
            # Convert emotion labels to emotions
            detected_emotions = [list(EMOTIONS.keys())[list(EMOTIONS.values()).index(label)] for label in captured_emotions]

            # Display detected emotions
            st.header("Detected Emotion:")
            st.write(detected_emotions[0])

            # Get book recommendation based on the first detected emotion
            book_recommendation = []
            for emotion in detected_emotions:
                book_recommendation += get_book_recommendations(emotion)

            # Display 5 random book recommendations
            st.header("Recommended Books:")
            if book_recommendation:
                random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
                for book in random_books:
                    st.write(book)
            else:
                st.write("No books found for the detected emotions.")

            # Get music recommendation based on the first detected emotion
            music_recommendation = []
            for emotion in detected_emotions:
                music_recommendation += get_music_recommendations(emotion)

            # Randomly select 5 music tracks
            st.header("Music Recommendation:")
            if music_recommendation:
                random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
                for music in random_music:
                    st.write(music)
            else:
                st.write("No music found for the detected emotions.")
        else:
            st.error("No faces detected in the uploaded image.")

elif option == "Capture Image":
    # Add a button to capture image
    if st.button("Capture Image"):
        # Capture image using webcam
        webcam = cv2.VideoCapture(0)

        # Capture and display the image
        ret, frame = webcam.read()
        if ret:
            st.image(frame, channels="BGR")
            captured_emotions = detect_emotion(frame)
        webcam.release()

        if captured_emotions:
            # Convert emotion labels to emotions
            detected_emotions = [list(EMOTIONS.keys())[list(EMOTIONS.values()).index(label)] for label in captured_emotions]

            # Display detected emotions
            st.header("Detected Emotion:")
            st.write(detected_emotions[0])

            # Get book recommendation based on the first detected emotion
            book_recommendation = []
            for emotion in detected_emotions:
                book_recommendation += get_book_recommendations(emotion)

            # Display 5 random book recommendations
            st.header("Recommended Books:")
            if book_recommendation:
                random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
                for book in random_books:
                    st.write(book)
            else:
                st.write("No books found for the detected emotions.")

            # Get music recommendation based on the first detected emotion
            music_recommendation = []
            for emotion in detected_emotions:
                music_recommendation += get_music_recommendations(emotion)

            # Randomly select 5 music tracks
            st.header("Music Recommendation:")
            if music_recommendation:
                random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
                for music in random_music:
                    st.write(music)
            else:
                st.write("No music found for the detected emotions.")
        else:
            st.error("No faces detected in the captured image.")

elif option == "Select Emotion":
    selected_emotion = st.selectbox("Select an emotion:", list(EMOTIONS.keys()))

    # Get book recommendation based on the selected emotion
    book_recommendation = get_book_recommendations(selected_emotion)

    # Display 5 random book recommendations
    st.header("Recommended Books:")
    if book_recommendation:
        random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
        for book in random_books:
            st.write(book)
    else:
        st.write("No books found for the selected emotion.")

    # Get music recommendation based on the selected emotion
    music_recommendation = get_music_recommendations(selected_emotion)

    # Randomly select 5 music tracks
    st.header("Music Recommendation:")
    if music_recommendation:
        random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
        for music in random_music:
            st.write(music)
    else:
        st.write("No music found for the selected emotion.")