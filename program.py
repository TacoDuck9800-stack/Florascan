import streamlit as st
with st.spinner("Loading..."):
	import matplotlib.pyplot as plt
	import numpy as np
	import PIL
	import tensorflow as tf
	import pathlib

	from tensorflow import keras
	from tensorflow.keras import layers
	from tensorflow.keras.models import Sequential

@st.cache_resource
def loadAIPics():
    with st.spinner():
        # Points directly to your local PlantVillage color folder
        data_dir = pathlib.Path("archive/plantvillage dataset/color")
        return data_dir




batch_size = 32
img_height = 180
img_width = 180

# start of AI
@st.cache_resource(show_spinner=False)
def trainModel():
	with st.spinner("Setting up dataset (this only happens once)...", show_time=True):
		data_dir = loadAIPics()

	with st.spinner("Making datasets"):
		train_ds = tf.keras.utils.image_dataset_from_directory( # trained dataset; trains the AI on the images
			data_dir, # directory path where the images are at
			validation_split=0.2, # 20% of the images will be used for validation
			subset="training", # the other 80% will be for training
			seed=123, # random number generator. If stays the same, the same images will always be used to train
			image_size=(img_height, img_width), # image size needed
			batch_size=batch_size) # 32 images at a time

		val_ds = tf.keras.utils.image_dataset_from_directory( # validation dataset
			data_dir,
			validation_split=0.2,
			subset="validation", # validation subset
			seed=123,
			image_size=(img_height, img_width),
			batch_size=batch_size)		

	class_names = train_ds.class_names # the directory names of the folders

	AUTOTUNE = tf.data.AUTOTUNE

	# train dataset; keeps trained images saved after first use
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # shuffles 1000 times, loads next images in background
	# validation dataset; will preload in background with computer specs
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


	normalization_layer = layers.Rescaling(1./255) # 0 - 1 instead of 0 - 255 RBG
	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # applies 1/255 RGB thing


	num_classes = len(class_names)
	with st.spinner("Training AI..."):
		model = Sequential([
			layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # 0 - 1 RGB
			layers.Conv2D(16, 3, padding='same', activation='relu'), # looks for shapes and edges
			layers.MaxPooling2D(),
			layers.Conv2D(32, 3, padding='same', activation='relu'), # looks for textures
			layers.MaxPooling2D(),
			layers.Conv2D(64, 3, padding='same', activation='relu'), # looks for complex shapes, like petals
			layers.MaxPooling2D(),
			layers.Flatten(), # converts image into list of numbers
			layers.Dense(128, activation='relu'), # passes the list into 128 'neurons' that check out the image
			layers.Dense(num_classes) # finally outputs what it thinks the image is
		])	

		model.compile(optimizer='adam',
					loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # checks how wrong it is so it can train better
					metrics=['accuracy']) # tracks percent of guesses that are correct
		
	# Epochs
	with st.spinner("Epochs..."):
		epochs=2
		history = model.fit(
			train_ds,
			validation_data=val_ds,
			epochs=epochs
		)

	return model, class_names, data_dir, train_ds

model, class_names, data_dir, train_ds = trainModel()

# end of AI

# treatment dictionary based on your class names
treatment_info = {
    "Apple___Apple_scab": {
        "status": "Apple Scab, Caused by Fungus 🍄",
        "advice": "Clean up fallen leaves that might have fungus on them and prune the tree to open it up for more air, so less moisture can reach the fungus."
    },
    "Apple___Black_rot": {
        "status": "Black Rot 🍎",
        "advice": "Prune infected branches, remove mummified fruit from the tree and ground, and apply protective fungicides during the growing season."
    },
    "Apple___Cedar_Apple_Rust": {
        "status": "Cedar Apple Rust 🌿",
        "advice": "Remove nearby juniper or cedar galls if possible, and apply protective fungicides in spring when spores are active."
    },
    "Apple___healthy": {
        "status": "Healthy Apple 🍏",
        "advice": "Your apple tree looks wonderful! Maintain regular watering, proper sunlight, and routine seasonal pruning."
    },
    "Blueberry___healthy": {
        "status": "Healthy Blueberry 🫐",
        "advice": "Optimal health! Keep the soil acidic, well-drained, and consistently moist."
    },
    "Cherry_(including_sour)___healthy": {
        "status": "Powdery Mildew (Cherry) 🍒",
        "advice": "Improve air circulation by pruning crowded branches, and apply organic sulfur or bio-fungicides at the first sign of white spots."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "status": "Healthy Cherry 🍒",
        "advice": "The cherry tree is thriving! Keep up good orchard hygiene and balanced watering."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "status": "Cercospora Leaf Spot (Gray Leaf Spot) 🌽",
        "advice": "Practice crop rotation, till under crop residue after harvest, and plant resistant corn hybrids."
    },
    "Corn_(maize)___Common_Rust": {
        "status": "Common Rust (Corn) 🌽",
        "advice": "Plant resistant hybrids and apply foliar fungicides if infection is severe early in the season."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "status": "Northern Leaf Blight 🌽",
        "advice": "Rotate crops annually, manage crop residue, and select resistant corn varieties."
    },
    "Corn_(maize)___healthy": {
        "status": "Healthy Corn 🌽",
        "advice": "Looking strong and healthy! Maintain standard nutrient levels and monitor soil moisture."
    },
    "Grape___Black_Rot": {
        "status": "Black Rot (Grape) 🍇",
        "advice": "Remove mummified berries and infected canes, ensure proper canopy management for sunlight penetration, and apply fungicides."
    },
    "Grape___Esca_(Black_Measles)": {
        "status": "Esca (Black Measles) 🍇",
        "advice": "Prune out infected wood during dry weather, avoid large pruning wounds, and remove dead vines to prevent spreading."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "Grape Leaf Blight 🍇",
        "advice": "Manage vine canopy for airflow, clear away fallen leaves, and apply appropriate preventative fungicides."
    },
    "Grape___healthy": {
        "status": "Healthy Grape 🍇",
        "advice": "Vines look vibrant and healthy! Continue proper trellis maintenance and seasonal pruning."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "status": "Citrus Greening (HLB) 🍊",
        "advice": "Control Asian citrus psyllid insects, immediately remove infected trees to protect the grove, and plant certified disease-free trees."
    },
    "Peach___Bacterial_spot": {
        "status": "Bacterial Spot (Peach) 🍑",
        "advice": "Plant resistant peach varieties, apply copper sprays during the dormant season, and avoid excessive nitrogen fertilizer."
    },
    "Peach___healthy": {
        "status": "Healthy Peach 🍑",
        "advice": "Your peach tree is doing great! Maintain proper fruit thinning and orchard sanitation."
    },
    "Pepper,_bell___Bacterial_Spot": {
        "status": "Bacterial Spot (Pepper) 🫑",
        "advice": "Use certified disease-free seeds, practice crop rotation, and apply copper-based bactericides if spots appear."
    },
    "Pepper,_bell___healthy": {
        "status": "Healthy Pepper 🫑",
        "advice": "Looking great! Keep up consistent watering at the base and balanced soil nutrition."
    },
    "Potato___Early_blight": {
        "status": "Early Blight (Potato) 🥔",
        "advice": "Practice crop rotation, use certified seed tubers, avoid overhead sprinkler watering, and apply preventative fungicides."
    },
    "Potato___Late_blight": {
        "status": "Late Blight (Potato) 🥔",
        "advice": "Destroy infected plants and tubers immediately to stop outbreaks, avoid wet foliage, and use registered fungicides."
    },
    "Potato___healthy": {
        "status": "Healthy Potato 🥔",
        "advice": "Plants look strong! Ensure proper hilling of soil and balanced watering schedules."
    },
    "Raspberry___healthy": {
        "status": "Healthy Raspberry 🍇",
        "advice": "Thriving berry bushes! Keep old canes pruned down after harvest and maintain a good layer of mulch."
    },
    "Soybean___health": {
        "status": "Healthy Soybean 🌱",
        "advice": "Crops look healthy! Maintain standard weed control and regular field scouting."
    },
    "Squash___Powdery_mildew": {
        "status": "Powdery Mildew (Squash) 🥒",
        "advice": "Ensure good plant spacing, water at the soil level to keep leaves dry, and spray with neem oil or potassium bicarbonate."
    },
    "Strawberry___Leaf_scorch": {
        "status": "Leaf Scorch (Strawberry) 🍓",
        "advice": "Remove and destroy severely infected leaves, ensure proper field drainage, and avoid overhead irrigation."
    },
    "Strawberry___healthy": {
        "status": "Healthy Strawberry 🍓",
        "advice": "Plants look wonderful! Keep fruit clean and off the soil using straw mulch."
    },
    "Tomato___Bacterial_Spot": {
        "status": "Bacterial Spot (Tomato) 🍅",
        "advice": "Use clean, disease-free seeds, avoid working with plants while they are wet, and apply copper sprays."
    },
    "Tomato___Early_Blight": {
        "status": "Early Blight (Tomato) 🍅",
        "advice": "Prune lower leaves near the soil, mulch around the base to prevent soil splashing, and apply a copper-based fungicide."
    },
    "Tomato___healthy": {
		"status": "Healthy (Tomato) 🍅",
		"advice": "Tomato looks wonderful! Keep growing it as you have been!"
	}
}


st.title("Hello Plant Lover! 🌼")
st.write("Want to know your plant's condition? Upload an image! 🪴")

uploadedFile = st.file_uploader( # Prompts the user 
	"Upload Image Here! // Hit + to change images", # Text above image uploader
	type=["jpg", "png", "jpeg"], # What file types are allowed
	max_upload_size=15 # In megabytes
	# maybe let the user upload multiple images of the same thing for better accuracy
)

if uploadedFile is not None:
    # 1. Process and predict the image
    img = PIL.Image.open(uploadedFile).convert("RGB").resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # 2. Get the predicted class name and confidence
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 3. Display the image and prediction
    st.image(img, caption="Analyzed Crop Image")
    st.write(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")

    # 4. Look up and display the treatment advice
    if predicted_class in treatment_info:
        info = treatment_info[predicted_class]
        
        if "Healthy" in info["status"]:
            st.success(f"**Status:** {info['status']}\n\n**Recommendation:** {info['advice']}")
        else:
            st.warning(f"**Status:** {info['status']}\n\n**Treatment Plan:** {info['advice']}")
    else:
        st.info("No specific treatment data available for this class yet.")