HERCODE 4.0 Real-Time Anti-Money Laundering (AML) Detection Model for Financial Transactionos Using Predefined Schemas:

Solution by Grace Poole

Work instructions:
1.	To run the code, use an Ubuntu shell. I would recommend using WSL on Command Prompt after enabling Bios settings upon restart of the computer you are using. 
2.	Note, the versions and packages from requirements.txt file are already installed upon composing the docker image which can be composed by running:
        In order for the solution to work you will need to ensure you have installed docker desktop on your computer and ticked the box for Expose daemon on tcp://localhost:2375 without TLS in the settings. You may need to log into your docker profile from the shell you are using also using docker login. 
        Kafka can be installed from the kafka_2.12-3.9.0 folder inside the AML folder and then installing kafka with
	- pip install kafka-python although this should be automatically handled by the docker image to install kafka-python==2.0.4
-	docker compose up --build
3.	For the optional functionality of querying the ML prediction model by invoking the API from the front end, ensure to serve the IP:
-	python3 -m http.server 8000
Once the website is running, access the web page from the following link:
-	http://localhost:8000/index.html
4.	Run the Flask API using the command:
-	python3 AMLAPI.py
5.	Now the website will be able to send queries to the API. 
Refresh and ensure the website is open. 
Enter in the transaction information on the front end (Should look something like this)
 
Once a transaction is sent, the response will display. Note that a result of 1 is indicative of a money laundering attempt whereas a result of 0 is indicative that there is no cause for concern from that give transaction.

Note: Ensure the port that is being used is available – By default I am using port 5000 for the Flask API but this can be edited in the Flask App or more ideally you would be killing anything using that port currently using the following commands:
-	sudo lsof -i :5000
-	sudo kill -9 <PID>

6.	The curl command that is triggering the prediction from either the web gui or the Kafka live stream is structured in the same way as the following example (Where different values can be used of course). 
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "Time": "12:46:14",
  "Date": "10/7/2022",
  "Sender_account": "5748489103",
  "Receiver_account": "5971760311",
  "Amount": "3250002.35",
  "Payment_currency": "UK pounds",
  "Received_currency": "UK dollar",
  "Sender_bank_location": "UK",
  "Receiver_bank_location": "USA",
  "Payment_type": "Cross-border"
}'
This example will produce the following output:
{
  "Is_laundering": 1
}
It is a known, randomly selected datapoint from the dataset of approximately 9millions transaction samples that I am using as input data.
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "Time": "20:11:10",
  "Date": "10/7/2022",
  "Sender_account": "976070224",
  "Receiver_account": "7951108994",
  "Amount": "7872.17",
  "Payment_currency": "UK pounds",
  "Received_currency": "UK pounds",
  "Sender_bank_location": "UK",
  "Receiver_bank_location": "UK",
  "Payment_type": "Credit card"
}'
{
  "Is_laundering": 0
}

This is an example of a case that is not laundering. 


The full output from my current configuration use to train the ML model can be viewed in the output.txt file. 
Average Cross-Validation Accuracy: 0.7964652483724449
Accuracy: 0.789367088607595
              precision    recall  f1-score   support

           0       0.76      0.85      0.81      2015
           1       0.83      0.72      0.77      1935

    accuracy                           0.79      3950
   macro avg       0.79      0.79      0.79      3950
weighted avg       0.79      0.79      0.79      3950
Theses are the metrics I obtained to view the accuracy. The accuracy can be increased by tweaking various parameters but I have found it seems to plateau at this level. 

These are the parameters of the ML Model I am using:
# Initializing RandomForestClassifier with class balancing
rf = RandomForestClassifier(
    n_estimators=100,  # Increase number of trees for more accuracy
    max_depth=30,      # Increase depth
    min_samples_split=10,  # Minimum samples to split a node
    min_samples_leaf=5,    # Minimum samples to be in a leaf node
    class_weight='balanced',
    random_state=42
)

	Note that when I max the number of trees to 300 the accuracy remains arounds 0.79 so I believe with this model that is the highest I can really achieve. 
7.	To demonstrate that my design can handle a stream of live input data I have given it the functionality of handling kafka producer and consumer data.
8.	Once the docker image has been composed up and after ensuring the Flask API is running it is then possible to run the producer:
-	python3 producer.py
And then a few moments after that to run the consumer:
-	python3 consumer.py

The producer creates the random data from randomly sampling elements from my original SAML-D dataset which are then fed into the consumer for the prediction whilst printing what the input data was also for reference. If the Flask API is being invoked directly from the frontend then the API will use the predict app (instead of the transaction app) which has been defined inside the AMLAPI.py file. The predict app goes straight to performing the prediction from the input data.

Please note that I retrieved the sample source data from Kaggle on the following URL:
https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml


