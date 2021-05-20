zip -r $2 $1
aws s3 cp $2.zip s3://dl4s-datasets/