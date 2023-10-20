
Download Data and Model here:

1. Download and unzip Amazon-1M dataset as part of [Amazon-1M.zip](https://drive.google.com/file/d/1aC8YB5y8SAYANy6ToXdrgkpYuMlPvd5G/view?usp=sharing).
2. Download and unzip trained model as part of [Amazon-1M-model.zip](https://drive.google.com/file/d/16Tvd66jOFQaq6cDNdKIDW0EVEwXz4vCq/view?usp=sharing).


Running the inference code on Amazon 1M data:

1. Set the environment variables in setup.sh file. 
2. Run `source setup.sh`
3. Run `python src/evaluate_XPERT.py configs/evaluation.yaml`

Data format:

1. item_features.txt contains the 768-dimensional embeddings of the Amazon product titles which were exracted from a pretrained 6-layered DistilBERT base mode.
The format of each row is: <item_id> <item_embedding>

2. final_data_test.txt and final_data_train.txt contains the test and train data respectively in the following format:
<user_id>   <label>   <label_time>  <history>
label = List of comma separated: <product_id> which are treated as the label
label_time = Timestamp of last reviewed product_id among labels
history = List of space separated: <product_id>:<timestamp> which are the user history

feat_data_bxml and user_data_test contains binarized files extracted from the files above, and are shared for fast inference.



