🌾 DroughtScanUAV
This is a real-time drought detection tool utilizing UAV (Unmanned Aerial Vehicle) captured images. By using the VARI index (Vegetation Index for Drought), it helps in assessing the health of vegetation and detecting drought conditions based on RGB images taken by UAVs. The project employs techniques like image stitching, vegetation classification, and interactive visualizations to analyze and present the drought status clearly and visually.

📁 Features
This section outlines the key capabilities of the project:

UAV Image Stitching: The project can stitch together multiple images taken by UAVs into one large, seamless image, providing a wider view of the area under study.

VARI Index Calculation: The VARI index is used to analyze vegetation health. This index helps detect changes in vegetation due to drought, making it a critical tool for agricultural monitoring.

Vegetation Classification: The script classifies vegetation into various categories such as Healthy, Moderate, Sparse, and Stressed based on the health of plants as identified through the UAV images.

Interactive Plots: The tool produces several visual outputs:

Heatmaps: Color-coded maps to show the intensity of drought or vegetation health across the area.

Pie Charts: Visual representation of different vegetation categories to show their proportion.

3D Topography: A 3D surface plot that gives a more in-depth view of the terrain and its vegetation health.

▶️ Run
This section provides step-by-step instructions on how to run the project on your local machine:

Install Dependencies: First, you need to install the required Python libraries. This is done by running the following command:pip install -r requirements.txt
This will install all the libraries listed in the requirements.txt file, such as OpenCV, NumPy, Plotly, and others required for the project.

Place Your Drone Images: Once the dependencies are installed, place the UAV images you want to analyze in the input_images/ folder. These images will be processed by the script.

Run the Script: Execute the main Python script to start the analysis:python drought_scan_uav.py
This command will trigger the analysis process, which includes stitching the images, calculating the VARI index, classifying vegetation, and generating visual outputs.

📊 Output
Once the script runs, it generates the following results:

Stitched UAV Image: A high-resolution stitched image of the area covered by UAV images.

VARI Heatmap: A heatmap visualizing the vegetation health based on the VARI index, helping you see which areas are more affected by drought.

Vegetation Classification Map: A visual map showing the classification of vegetation (Healthy, Moderate, Sparse, Stressed).

Pie Chart & 3D Surface Plot: The pie chart provides a breakdown of the proportions of each vegetation category, while the 3D surface plot offers a topographical view of the area, highlighting drought conditions.

🛠️ Tech Stack
This section lists the technologies and libraries used to build the project:

Python: The primary programming language for scripting.

OpenCV: Used for image processing and stitching.

NumPy: For numerical operations and handling arrays.

Plotly: A library for creating interactive visualizations like heatmaps and 3D plots.

Matplotlib: Used for generating static visualizations like charts.

