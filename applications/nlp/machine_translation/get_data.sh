# Data collected from: https://paracrawl.eu/
echo 'Grab a cup of coffee. This may take a while to complete.'

echo '(1 / 4) Creating corpus directory...'
if [ ! -d "./corpus" ]; then
	mkdir ./corpus
else
	echo "Output directory found, skipping this step."
fi
echo 'Done.'

echo '(2 / 4) Collecting data...'
GZ_FILE=./corpus/en-fi.txt.gz

if [[ -f "$GZ_FILE" || -f "./corpus/en-fi.txt" ]]; then
	echo ".gz or extracted file found, skipping this step."
else
	wget https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-fi.txt.gz -O "$GZ_FILE"
fi

echo 'Done.'

echo '(3 / 4) Extracting data...'

if [ -f "./corpus/en-fi.txt" ]; then
	echo "Extracted file found, skipping this step."
else
	gzip -d ./corpus/en-fi.txt.gz
fi

echo 'Done.'

echo '(4 / 4) Building Byte Pair Encoding vocabulary...'
python ./prepare_data.py
echo 'Done.'
