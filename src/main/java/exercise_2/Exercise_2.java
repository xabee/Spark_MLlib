package exercise_2;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;

import scala.Tuple2;

import com.google.common.collect.Lists;

public class Exercise_2 {
	
	static class FeaturesToString implements Function<Tuple2<Object, double[]>, String> {
		private static final long serialVersionUID = 1L;
		@Override
	    public String call(Tuple2<Object, double[]> element) {
	      return element._1() + "," + Arrays.toString(element._2());
	    }
	  }

	public static void musicRecommendation(JavaSparkContext sc) {
			
		JavaRDD<String> rawUserArtistData = sc.textFile("src/main/resources/2_user_artist_data_1m.txt");
		JavaRDD<String> rawArtistData = sc.textFile("src/main/resources/2_artist_data.txt");
		JavaRDD<String> rawArtistAlias = sc.textFile("src/main/resources/2_artist_alias.txt");

		
		/*************************************************
		 *********** DISPLAY STATISTICS ******************
		 *************************************************/
		
		MultivariateStatisticalSummary userIDs_stats = Statistics.colStats( rawUserArtistData.map(linea -> Vectors.dense( Double.parseDouble(linea.split(" ")[0]) ) ).rdd() );
		
		MultivariateStatisticalSummary artistIDs_stats = Statistics.colStats(rawUserArtistData.map(linea -> Vectors.dense( Double.parseDouble(linea.split(" ")[1]) ) ).rdd() );
				
		System.out.println("Statistics for user IDs: [min] "+userIDs_stats.
				min()+", [max] "+userIDs_stats.max()+"; [count] "+userIDs_stats.count());
		System.out.println("Statistics for artist IDs: [min] "+	artistIDs_stats.
				min()+", [max] "+artistIDs_stats.max()+"; [count] "+artistIDs_stats.count());
		
		// Statistics for user IDs: [min] [1000002.0], [max] [1000320.0]; [count] 60183
		// Statistics for artist IDs: [min] [1.0], [max] [1.0787933E7]; [count] 60183
		
		/*************************************************
		 ************* PREPROCESSING *********************
		 *************************************************/

		JavaPairRDD<Integer,String> id_name = rawArtistData.flatMapToPair(linea -> {
			List<Tuple2<Integer,String>> idArtist = Lists.newArrayList();
			String[] parts = linea.split("\t");
			if (parts.length>=2 && Utils.isInteger(parts[0]) && !parts[1].trim().isEmpty()) {
				Tuple2<Integer,String> artist = new Tuple2<Integer,String>(Integer.parseInt(parts[0]), parts[1] );
				idArtist.add(artist);
			}
			return idArtist;
		});
		
		JavaPairRDD<Integer, Integer> artistAlias = rawArtistAlias.flatMapToPair(linea -> {
			List<Tuple2<Integer, Integer>> ididArtist = Lists.newArrayList();
			String[] parts = linea.split("\t");
			if (parts.length>=2 && Utils.isInteger(parts[0]) && Utils.isInteger(parts[1])) {
				Tuple2<Integer, Integer> artist = new Tuple2<Integer,Integer>(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]) );
				ididArtist.add(artist);
			}
			return ididArtist;
		});
		
		
		/*************************************************
		 ************** BUILD MODEL **********************
		 *************************************************/
		
		final Broadcast<JavaPairRDD<Integer,Integer>> bArtistAlias = sc.broadcast(artistAlias.cache());
		
		JavaRDD<Rating> trainData = rawUserArtistData.sample(true,0.1).map( linea -> {
			String[] parts = linea.split(" ");
			List<Integer> canonId = bArtistAlias.value().lookup(Integer.parseInt(parts[1]) );
			if(!canonId.isEmpty()){
				return new Rating (Integer.parseInt(parts[0]), canonId.get(0), Double.parseDouble(parts[1]));
			} else {
				return new Rating (Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Double.parseDouble(parts[1]));
			}
		}).cache();
		
		MatrixFactorizationModel model = ALS.trainImplicit(trainData.rdd(), 10, 5, 0.01, 1.0);
		
		/*************************************************
		 ********** SPOT CHECK RECOMMENDATIONS ***********
		 *************************************************/
		
		Rating[] users_rec = model.recommendUsers(606, 5);
		String out = "";
		for (Rating r :users_rec) {
			//System.out.println("User: "+r.user() );
			out = out + "\n" + "User: "+r.user();
			Rating[] prod_rec = model.recommendProducts(r.user(), 5);
			for (Rating rp : prod_rec) {
				//System.out.println("\t" + id_name.lookup(rp.product()) );
				out = out + "\n" + "\t" + id_name.lookup(rp.product());
			}
		}
		System.out.println(out);
		

		
	}
}
