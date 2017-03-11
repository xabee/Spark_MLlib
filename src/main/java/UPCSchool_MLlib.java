import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import exercise_2.Exercise_2;


public class UPCSchool_MLlib {
	
	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", "G:\\winutil\\");

		
		SparkConf conf = new SparkConf().setAppName("UPCSchool-MLlib").setMaster("local[*]");
		JavaSparkContext ctx = new JavaSparkContext(conf);
		

		// Exercise_1.spamDetection(ctx);
		Exercise_2.musicRecommendation(ctx);
		//Exercise_3.networkAnomalyDetection(ctx);
	}

}
