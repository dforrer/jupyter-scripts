!#/bin/bash

# Run with '. startup-spark.sh' to set the paths in the current shell

echo "Setting SPARK_HOME and PYTHONPATH"

export SPARK_HOME=/home/farmer/spark
PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.9-src.zip:$PYTHONPATH
export PYTHONPATH

echo "Starting Spark..."

~/spark/sbin/start-all.sh

