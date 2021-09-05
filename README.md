# GitHub Collaboration Network and Candidate Recommendation System for Recruiters
Software engineering recruiters receive hundreds or even thousands of applications for a particular job role. Recruiters have to go through individual resumes to filter the best set of candidates. This process is time-consuming, and when the applicant list is big, it is impossible to go through all the profiles resulting in the possible exclusion of suitable candidates. It is to be noted that many of the resumes contain a GitHub profile link, and these profiles can be beneficial in determining if the candidate can be a good fit for a job role. As such, we pulled the publicly available dataset from GitHub, mined it, and built models to find similarities between users. We also developed metrics to measure users' contributions and a recommendation system using these metrics. Recruiters looking to hire software engineering candidates can use this recommendation system to filter candidates based on the recommendations. To build the recommendation system, we use a network-based approach and a machine learning approach. Finally, we evaluate the system on a real dataset collected from Github public API and present the result.

# Step 1 - Setup Instructions
1. Clone the repository
2. Install Python3
3. Run - python3 -m pip install -r requirements.txt

# Step 2 - Download already downloaded network and ML data
1. Download the files in from this google drive link: https://drive.google.com/drive/folders/1go2xQTel_xZk9ZQqCgojtnHBzU7vk2Of?usp=sharing

# Step 3 - Run code and Download the data again
1. Go to Github and generate api tokens for its public api.
2. Open step1_download_github_public_data.py in any notebook.
3. Add the username and token generated from github in variable called "credentials" in the file.
4. Save and Close the file.
5. Open command line and run - python step1_download_github_public_data.py

The script will download necessary file and dump the network data as a python pickle file.

# Step 4 - Train machine learning models
1. Load step2_train_machine_learning_model.py.ipynb in Jupyternotebook.
2. Run the code line by line (if necessary use already downloaded data).

# Step 5 - Run the Recommendation System
If you just want to run the recommendation system, you can skip step 3 and step 4, and directly run the step 5, then follow the instructions on step 5.

1. Open terminal
2. Run python main.py
3. Follow the instructions printed on the terminal.
4. If necessary download already prepared data from Step 2 and run the code.
