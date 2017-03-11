package exercise_1;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.clearspring.analytics.util.Lists;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;


public class Exercise_1 {

	public static void spamDetection(JavaSparkContext jsc) {
		// Load the training set labels
		JavaRDD<String> labels = jsc.textFile("src/main/resources/1_spam-mail.tr.label");

		// Split each line into two string components (ID and Label).
		// Replace the nulls for the proper code
		JavaPairRDD<Integer,Integer> predictionById = labels.mapToPair(label -> {
			List<String> etiqueta = Arrays.asList(label.split(","));
			return new Tuple2<Integer, Integer>(Integer.parseInt(etiqueta.get(0)),Integer.parseInt(etiqueta.get(1)));
		});
		
		// Broadcast variables allow the programmer to keep a read-only variable cached on each machine
		// rather than shipping a copy of it with tasks
		final Broadcast<JavaPairRDD<Integer,Integer>> broadcastPredictionById = jsc.broadcast(predictionById.cache());

		// Regex to extract the ID
		final Pattern pattern = Pattern.compile(".*TRAIN_([0-9]*).*");
		
		// Get training set files
		JavaPairRDD<String,String> allEmails = jsc.wholeTextFiles("src/main/resources/1_TR");
		
		System.out.println("allEmails: " + allEmails.count());
		
		// Map to 0/1 ham/spam
		JavaPairRDD<Integer,String> labeledEmails = allEmails.flatMapToPair(email -> {
			List<Tuple2<Integer,String>> labeledEmail = Lists.newArrayList();
			//      Using class Matcher, match the regular expression with the email
			//      If there is a match, obtain the ID from the regular expression with the group method.
			//      Further obtain the label using the lookup method from the broadcast variable
			//      Finally, add a new Tuple2<Integer,String> with the pair <label,email> to labeledEmail
			Matcher mclass=pattern.matcher(email._1);
			System.out.println("------------------->>>>>>>>>>>>>> START matcher!");

			if(mclass.find())
			{
				int mailID=Integer.parseInt(mclass.group(1));
				int label =broadcastPredictionById.value().lookup(mailID).get(0);
				Tuple2<Integer,String> correo = new Tuple2<Integer,String>(label,email._2);
				labeledEmail.add(correo);
				System.out.println("---->> positive match! " + email._1);
			}
			

			return labeledEmail;
		});
		
		System.out.println("Labeled emails: " + labeledEmails.count());

		labeledEmails.cache();
		System.out.println(labeledEmails.toString());
		
		// Store into the ham RDD only emails of type 0
		JavaRDD<String> ham = labeledEmails.filter(email -> email._1==0).map(email -> email._2);
		System.out.println("ham: " + ham.count());
		
		// Store into the spam RDD only emails of type 1
		JavaRDD<String> spam = labeledEmails.filter(email -> email._1==1).map(email -> email._2);
		System.out.println("spam: " + spam.count());

		// Similarly to the training set, obtain a JavaRDD<String> of all emails from the test set	
		JavaRDD<String> testEmails = jsc.wholeTextFiles("src/main/resources/1_TT").map(email->email._2);

		/**
		 * MLlib
		 */
		// Create a HashingTF instance to map email text to vectors of 100 features.
		final HashingTF tf = new HashingTF(100);

		JavaRDD<LabeledPoint> positiveExamples = spam.map(email -> {
			// split the email by spaces and apply the transform method from tf to the array
			List<String> words = Arrays.asList(email.split(" "));
			tf.transform(words);
			
			LabeledPoint pos = new LabeledPoint(1.0, tf.transform(words));
			// return a new LabeledPoint with label 1 (ham) and the result of the tf transformation
			return pos;
		});
		JavaRDD<LabeledPoint> negativeExamples = ham.map(email -> {
			// split the email by spaces and apply the transform method from tf to the array
			List<String> words = Arrays.asList(email.split(" "));
			tf.transform(words);
			
			LabeledPoint neg = new LabeledPoint(0.0, tf.transform(words));

			// return a new LabeledPoint with label 0 (spam) and the result of the tf transformation
			return neg;
		});

		// generate the trainingData RDD as the union of positive and negative examples
		JavaRDD<LabeledPoint> trainingData = negativeExamples.union(positiveExamples);
		
		System.out.println("Positive: " + positiveExamples.count());
		System.out.print("Negative: " + negativeExamples.count());
		
		// Cache for Logistic Regression, iterative algorithm
		trainingData.cache();

		// Create for Logistic Regression learner, LBFGS optimizer
		LogisticRegressionWithSGD lrLearner = new LogisticRegressionWithSGD();
		
		// Run learning algorithm on training data
		LogisticRegressionModel model = lrLearner.run(trainingData.rdd());

		Vector posTestExample = tf.transform(Arrays.asList("O M G GET cheap stuff by sending money to ...".split(" ")));
		Vector negTestExample = tf.transform(Arrays.asList("Hi Dad, I started studying Spark the other ...".split(" ")));

		System.out.println("Prediction for positive test example: "+ model.predict(posTestExample));
		System.out.println("Prediction for negative test example: "+ model.predict(negTestExample));
		
	}

}