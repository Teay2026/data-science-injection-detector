"""One Spark session for the whole project."""

import os
import subprocess
import sys

from pyspark.sql import SparkSession


def get_spark(memory="4g"):
    # Workers must run the same Python as the notebook.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    # Local mode needs no network. Avoids a warning on some machines.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    # Spark 3.5 needs Java 17. On macOS, pick it if JAVA_HOME is not set.
    if sys.platform == "darwin" and "JAVA_HOME" not in os.environ:
        java17 = subprocess.run(["/usr/libexec/java_home", "-v", "17"], capture_output=True, text=True)
        if java17.returncode == 0:
            os.environ["JAVA_HOME"] = java17.stdout.strip()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("injection-detector")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
