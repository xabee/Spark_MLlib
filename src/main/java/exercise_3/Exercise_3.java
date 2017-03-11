package exercise_3;

import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import com.google.common.collect.Lists;

public class Exercise_3 {

	public static void networkAnomalyDetection(JavaSparkContext sc) {
		JavaRDD<String> dataset = sc.textFile("src/main/resources/3_kddcup.csv");
		
	}
}
