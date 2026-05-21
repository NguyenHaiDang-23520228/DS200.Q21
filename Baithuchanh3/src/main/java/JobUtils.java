import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public final class JobUtils {

    private JobUtils() {
    }

    public static Map<String, String> loadMovieTitles(String moviesPath, Configuration conf) throws IOException {
        Map<String, String> movieMap = new HashMap<>();
        Path path = new Path(moviesPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                int firstComma = line.indexOf(',');
                int lastComma = line.lastIndexOf(',');
                if (firstComma <= 0 || lastComma <= firstComma) {
                    continue;
                }

                String id = line.substring(0, firstComma).trim();
                String title = line.substring(firstComma + 1, lastComma).trim();
                movieMap.put(id, title);
            }
        }
        return movieMap;
    }

    public static Map<String, String> loadMovieGenres(String moviesPath, Configuration conf) throws IOException {
        Map<String, String> movieGenresMap = new HashMap<>();
        Path path = new Path(moviesPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                int firstComma = line.indexOf(',');
                int lastComma = line.lastIndexOf(',');
                if (firstComma <= 0 || lastComma <= firstComma) {
                    continue;
                }

                String movieId = line.substring(0, firstComma).trim();
                String genres = line.substring(lastComma + 1).trim();
                movieGenresMap.put(movieId, genres);
            }
        }
        return movieGenresMap;
    }

    public static Map<String, String> loadUserGender(String usersPath, Configuration conf) throws IOException {
        Map<String, String> userGenderMap = new HashMap<>();
        Path path = new Path(usersPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    continue;
                }

                userGenderMap.put(parts[0].trim(), parts[1].trim().toUpperCase());
            }
        }
        return userGenderMap;
    }

    public static Map<String, Integer> loadUserAge(String usersPath, Configuration conf) throws IOException {
        Map<String, Integer> userAgeMap = new HashMap<>();
        Path path = new Path(usersPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length < 3) {
                    continue;
                }

                try {
                    userAgeMap.put(parts[0].trim(), Integer.parseInt(parts[2].trim()));
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return userAgeMap;
    }

    public static Map<String, String> loadUserOccupation(String usersPath, Configuration conf) throws IOException {
        Map<String, String> userOccupationMap = new HashMap<>();
        Path path = new Path(usersPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length < 4) {
                    continue;
                }

                userOccupationMap.put(parts[0].trim(), parts[3].trim());
            }
        }
        return userOccupationMap;
    }

    public static Map<String, String> loadOccupationNames(String occupationPath, Configuration conf) throws IOException {
        Map<String, String> occupationMap = new HashMap<>();
        Path path = new Path(occupationPath);
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    continue;
                }

                occupationMap.put(parts[0].trim(), parts[1].trim());
            }
        }
        return occupationMap;
    }

    public static String ageGroup(int age) {
        if (age < 18) {
            return "0-18";
        }
        if (age < 35) {
            return "18-35";
        }
        if (age < 50) {
            return "35-50";
        }
        return "50+";
    }

    public static void deleteOutputPath(Configuration conf, String outputPath) throws IOException {
        Path path = new Path(outputPath);
        FileSystem fs = path.getFileSystem(conf);
        if (fs.exists(path)) {
            fs.delete(path, true);
        }
    }
}
