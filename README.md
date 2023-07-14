# Products Recommendation

## Context

I worked on this project for my internship for _WimbeeTech_ in Tunisia in 2023

## Project Description

This project is based on a competition on kaggle. 

Every month, the Spanish bank SANTANDER offers the same 24 products to each of its customers.  My application studies the profile of the customer(s) and the month concerned. Using this data, it is able to rank the 24 products in order of recommendation.

You can have more details on  _https://www.kaggle.com/competitions/santander-product-recommendation_




## Installation

1. Clone the repository: `git clone https://github.com/Azizo27/Recommendation.git`
2. Install Python on your device
3. Install the dependencies: `pip install -r requirements.txt`
4. Download and unzip **train_ver2.csv.zip** from kaggle 


Note: Make sure you have Python 3.x and pip installed on your system: `python --version`


## Configuration

1. Create a cleaned and renamed train file from train_ver2.csv _(more details in LoadCsv.py file)_.
2. Create all models for each month _(more details in CreatingModelProduct.py file)_


## Usage

### First Usage: One Customer

1. Start the web application: `flask run`
2. Go to _http://127.0.0.1:5000/_
3. Choose for which month you want to do prediction
4. Write the customer data
5. Click on send button
6. Wait few minutes and you will get ranked recommendation

### Second Usage: Many Customers

1. Go to main.py file
2. Load a cleaned and renamed version of the dataframe you want to predict
3. Choose for which month you want to do prediction
4. Open terminal and run : `python3 main.py`

## Contributing

Contributions are welcome! If you have any bug reports, feature requests, or improvements, please submit them as GitHub issues or pull requests. 


## Credits

All the open-source libraries used in this project are listed in requirements.txt file

