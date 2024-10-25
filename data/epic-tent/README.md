## Set up the EPIC-Tent-O dataset

To create the train and test files, first clone the official repository [EPIC_Tent2019](https://github.com/youngkyoonjang/EPIC_Tent2019) using the following command:

```shell
git clone https://github.com/youngkyoonjang/EPIC_Tent2019.git
```

Then, run the following commands to generate the `train.json` and `test.json` files:

```shell
# Create train.json
python create_annotation_file.py -s train_split.json -o train.json

# Create test.json
python create_annotation_file.py -s test_split.json -o test.json
```

To verify that the files you generated match those used in the experiments, we have saved the hashes of our `train.json` and `test.json` files in `train_hash` and `test_hash`, respectively. Use the following commands to verify:

```shell
# Check hash for train.json
python check_hash.py -h train_hash -f train.json

# Check hash for test.json
python check_hash.py -h test_hash -f test.json
```

## License

THE LICENSE REMAINS THE ORIGINAL, WITH NO CHANGES.