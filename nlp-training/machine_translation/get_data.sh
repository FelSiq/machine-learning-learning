# Data collected from: https://paracrawl.eu/
echo 'Creating corpus directory...'
mkdir ./corpus
echo 'Done.'

echo 'Collecting data...'
wget https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-fi.txt.gz -O ./corpus/en-fi.txt.gz
echo 'Done.'

echo 'Extracting data...'
gzip -d corpus/en-fi.txt.gz
echo 'Done.'

echo 'Building Byte Pair Encoding vocabulary...'
python prepare_data.py
echo 'Done.'
