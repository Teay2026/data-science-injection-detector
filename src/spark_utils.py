"""One Spark session for the whole project."""

import os
import subprocess
import sys
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = str(Path(__file__).resolve().parents[1])


def get_spark(memory="4g"):
    # Workers must run the same Python as the notebook.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    # Workers must also find our src folder: the pandas UDFs of src/cleaning.py run there.
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if ROOT not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([ROOT] + [p for p in paths if p])
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
        # On Windows the Python workers keep this timeout (15 s by default) on their
        # connection to Spark: a step slower than that kills them ("Connection reset").
        .config("spark.python.authenticate.socketTimeout", "1h")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
