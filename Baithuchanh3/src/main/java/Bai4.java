import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Bai4 {

    public static class RatingMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) {
                return;
            }

            String[] parts = line.split(",");
            if (parts.length < 3) {
                return;
            }

            String userId = parts[0].trim();
            String movieId = parts[1].trim();
            String ratingStr = parts[2].trim();

            try {
                Double.parseDouble(ratingStr);
            } catch (NumberFormatException e) {
                return;
            }

            outKey.set(movieId);
            outValue.set(userId + "," + ratingStr);
            context.write(outKey, outValue);
        }
    }

    public static class AgeGroupReducer extends Reducer<Text, Text, NullWritable, Text> {
        private Map<String, Integer> userAgeMap;
        private Map<String, String> movieTitleMap;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            String usersPath = conf.get("users.path");
            String moviesPath = conf.get("movies.path");

            if (usersPath == null || usersPath.isEmpty()) {
                throw new IOException("Thieu tham so users.path");
            }
            if (moviesPath == null || moviesPath.isEmpty()) {
                throw new IOException("Thieu tham so movies.path");
            }

            userAgeMap = JobUtils.loadUserAge(usersPath, conf);
            movieTitleMap = JobUtils.loadMovieTitles(moviesPath, conf);
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Map<String, Double> sumMap = new LinkedHashMap<>();
            Map<String, Integer> countMap = new LinkedHashMap<>();
            sumMap.put("0-18", 0.0);
            sumMap.put("18-35", 0.0);
            sumMap.put("35-50", 0.0);
            sumMap.put("50+", 0.0);
            countMap.put("0-18", 0);
            countMap.put("18-35", 0);
            countMap.put("35-50", 0);
            countMap.put("50+", 0);

            for (Text value : values) {
                String[] parts = value.toString().split(",");
                if (parts.length < 2) {
                    continue;
                }

                Integer age = userAgeMap.get(parts[0].trim());
                if (age == null) {
                    continue;
                }

                double rating;
                try {
                    rating = Double.parseDouble(parts[1].trim());
                } catch (NumberFormatException e) {
                    continue;
                }

                String group = JobUtils.ageGroup(age);
                sumMap.put(group, sumMap.get(group) + rating);
                countMap.put(group, countMap.get(group) + 1);
            }

            String movieTitle = movieTitleMap.getOrDefault(key.toString(), "Unknown Movie (" + key + ")");

            double avg0_18 = countMap.get("0-18") > 0 ? sumMap.get("0-18") / countMap.get("0-18") : 0.0;
            double avg18_35 = countMap.get("18-35") > 0 ? sumMap.get("18-35") / countMap.get("18-35") : 0.0;
            double avg35_50 = countMap.get("35-50") > 0 ? sumMap.get("35-50") / countMap.get("35-50") : 0.0;
            double avg50 = countMap.get("50+") > 0 ? sumMap.get("50+") / countMap.get("50+") : 0.0;

            String out = String.format(
                    Locale.US,
                    "%s: [0-18: %.2f, 18-35: %.2f, 35-50: %.2f, 50+: %.2f]",
                    movieTitle, avg0_18, avg18_35, avg35_50, avg50
            );
            context.write(NullWritable.get(), new Text(out));
        }
    }

    public static int runJob(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: Bai4 <ratings_input_path> <output_path> <users_path> <movies_path>");
            return 1;
        }

        Configuration conf = new Configuration();
        conf.set("users.path", args[2]);
        conf.set("movies.path", args[3]);
        JobUtils.deleteOutputPath(conf, args[1]);

        Job job = Job.getInstance(conf, "Bai4 - Rating by Age Group");
        job.setJarByClass(Bai4.class);

        job.setMapperClass(RatingMapper.class);
        job.setReducerClass(AgeGroupReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(runJob(args));
    }
}
