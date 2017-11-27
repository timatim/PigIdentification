echo "Starting to download JDD Pig Identification dataset..."
while read p; do
	echo "Downloading ${p}..."
	wget "${p}"
done < jdd_dataset_A.txt

mv Pig_Identification_Qualification_Test_A.zip Pig_Identification_Qualification_Test_A.z04
mv Pig_Identification_Qualification_Train.zip Pig_Identification_Qualification_Train.z09

# concatenate, repair, and unzip
cat Pig_Identification_Qualification_Test_A.z* > Pig_Identification_Qualification_Test_A.zip
zip -FF Pig_Identification_Qualification_Test_A.zip --out test-full.zip
unzip test-full.zip

cat Pig_Identification_Qualification_Train.z* > Pig_Identification_Qualification_Train.zip
zip -FF Pig_Identification_Qualification_Train.zip --out train-full.zip
unzip train-full.zip
