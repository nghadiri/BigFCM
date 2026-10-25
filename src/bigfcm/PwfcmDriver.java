
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Random;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PwfcmDriver {
  private Integer dimension_of_points = 0;
  Vector<String> tuples_a = null;
  private Integer num_of_points = 0;
  private Integer num_of_centroids = 0;
  private Double[][] mat_points = null;
  private Double epsilon = 0.00;
  private Double epsilon_last = 0.00;
  private Double fuzziness = 0.00;
  private Double[][] cluster_center = null;
  private Double[] center_weight = null;
  private Double[] center_weight_in = null;
  private Double[] dsqr = null;
  private Double[] numer3 = null;
  private Double[] center_weight_from_prev_iter = null;
  private Double[] total_center_weight_from_prev_and_curr_operation = null;
  private Double[][] center_points = null;
  public int reminder = 0;
  public int divider = 0;
  private Double[][] cluster_center_prev = null;
  int fiterations = 0;

  public void initialize_vars(int num_centers, int dims, double fuzziness,
      double epsilon, int iter) {
    num_of_centroids = num_centers;
    dimension_of_points = dims;
    this.fuzziness = fuzziness;
    this.epsilon = epsilon;
    this.epsilon_last = epsilon;
    this.fiterations = iter;

  }

  public void last_run(String path_file) throws IOException {
    prepare_str(path_file);
    wfcm();
    Path pt = new Path(path_file + "/real_out");
    FileSystem fs = FileSystem.get(new Configuration());
    BufferedWriter br =
        new BufferedWriter(new OutputStreamWriter(fs.create(pt, true)));

    for (int i = 0; i < num_of_centroids; i++) {
      StringBuilder str_builder = new StringBuilder();
      for (int j = 0; j < dimension_of_points; j++) {
        str_builder.append(Double.toString(cluster_center[i][j]));
        str_builder.append(" ");
      }
      str_builder.append("\n");
      br.write(str_builder.toString());
    }
    br.close();
    double sum = 0;
    for (int i = 0; i < num_of_points; i++) {
      sum += center_weight_in[i];
    }
    System.out.println(sum);
  }

  private void prepare_str(String path_file) throws IOException {
    tuples_a = new Vector<String>();
    Path ofile = new Path(path_file + "/part-r-00000");
    FileSystem fs = FileSystem.get(new Configuration());
    BufferedReader br =
        new BufferedReader(new InputStreamReader(fs.open(ofile)));

    String l = br.readLine();
    while (l != null) {
      String[] str = l.split("\\s+");
      int sizea = str.length;
      if (sizea == dimension_of_points + 1 | sizea == dimension_of_points + 2) {
        tuples_a.add(l);
      }
      l = br.readLine();
    }
    br.close();
    num_of_points = tuples_a.size();
    divider = num_of_points;
    mat_points = new Double[num_of_points][dimension_of_points];
    center_weight = new Double[num_of_points];
    center_weight_in = new Double[num_of_points];
    cluster_center = new Double[num_of_centroids][dimension_of_points];
    dsqr = new Double[num_of_centroids];
    numer3 = new Double[num_of_centroids];
    int size_inner;
    int u = 0;
    double parsed_double = 0.0;
    for (int i = 0; i < num_of_points; i++) {
      String[] str_sp = tuples_a.get(i).split("\\s+");
      size_inner = str_sp.length;
      if (size_inner == dimension_of_points + 2) {
        u = -1;
        for (String mover : str_sp) {
          if (u == -1) {
            u = 0;
            continue;
          }
          parsed_double = Double.parseDouble(mover);
          if (u == dimension_of_points) {
            center_weight_in[i] = parsed_double;
          } else {
            mat_points[i][u] = parsed_double;
            if (i < num_of_centroids) {
              cluster_center[i][u] = parsed_double;
            }
          }
          u++;
        }
      }
    }

  }


  private Double cal_Euc_distnce(Integer i, Integer j) {
    int k;
    Double sum = 0.0;
    for (k = 0; k < dimension_of_points; k++) {
      double x = cluster_center[j][k];
      double y = mat_points[i][k];
      if (y - x == 0) {
        sum += Math.pow(0.00001, 2);
      } else {
        sum += Math.pow(y - x, 2);
      }

    }
    return Math.sqrt(sum);
  }// end of calculate the norm


  private Double update_centroids_weight() {
    double max_diff = 0;
    double[][] temp_centroid =
        new double[num_of_centroids][dimension_of_points];
    double denom3;
    double u = 0.0;
    double temp_diff = 0.0;
    double new_center = 0.0;
    for (int q = 0; q < num_of_centroids; q++) {
      center_weight[q] = 0.0;
      for (int w = 0; w < dimension_of_points; w++) {
        temp_centroid[q][w] = 0.0;
      }
    }
    for (int k = 0; k < num_of_points; k++) {
      denom3 = 0.0;
      for (int i = 0; i < num_of_centroids; i++) {
        dsqr[i] = cal_Euc_distnce(k, i);
        numer3[i] = Math.pow(dsqr[i], 2 / ((double) fuzziness - 1));
        denom3 += 1 / (double) numer3[i];
      }// end of for i to number of centroids

      for (int i = 0; i < num_of_centroids; i++) {
        u = Math.pow(denom3 * numer3[i], -1 * fuzziness);
        for (int d = 0; d < dimension_of_points; d++) {
          temp_centroid[i][d] += mat_points[k][d] * u * center_weight_in[k];
        }// end of
        center_weight[i] += u * center_weight_in[k];
      }// end of for i to number of centroids
    }// end of for k to number of points
    for (int i = 0; i < num_of_centroids; i++) {
      for (int d = 0; d < dimension_of_points; d++) {
        new_center = temp_centroid[i][d] / center_weight[i];
        temp_diff = Math.abs(cluster_center[i][d] - new_center);
        if (temp_diff > max_diff) {
          max_diff = temp_diff;
        }
        cluster_center[i][d] = new_center;
      }
    }
    return max_diff;

  }

  boolean wfcm() {
    int iteration = 0;
    double max_diff = 0.0;
    do {
      max_diff = update_centroids_weight();
      iteration++;
    } while (max_diff > epsilon_last && iteration < fiterations);
    return true;
  }

  // /prepare str 2

  private void prepare_str_2(String path_file, int count, int num_center)
      throws IOException {
    tuples_a = new Vector<String>();
    Path ofile = new Path(path_file);
    FileSystem fs = FileSystem.get(new Configuration());
    BufferedReader br =
        new BufferedReader(new InputStreamReader(fs.open(ofile)));
    int counter = 0;
    String l = br.readLine();
    Random rand = new Random();
    int randomNum;
    while (l != null & counter < count) {
      String[] str = l.split(",");
      // System.out.println(l);
      // String[] str = l.split(" ");
      randomNum = rand.nextInt(2);
      int sizea = str.length;

      if (sizea == dimension_of_points & randomNum == 1) {
        tuples_a.add(l);
        counter++;
      }
      l = br.readLine();
    }
    br.close();
    num_of_points = tuples_a.size();
    divider = num_of_points;
    num_of_centroids = num_center;
    mat_points = new Double[num_of_points][dimension_of_points];
    dsqr = new Double[num_of_centroids];
    numer3 = new Double[num_of_centroids];
    cluster_center = new Double[num_of_centroids][dimension_of_points];
    center_weight = new Double[num_of_centroids];
    cluster_center_prev = new Double[num_of_centroids][dimension_of_points];
    center_weight_from_prev_iter = new Double[num_of_centroids];
    total_center_weight_from_prev_and_curr_operation =
        new Double[2 * num_of_centroids];
    center_points = new Double[2 * num_of_centroids][dimension_of_points];

    int rand_center = 0;
    int y = 0;
    for (int i = 0; i < num_of_points; i++) {
      String[] str_sp = tuples_a.get(i).split(",");
      // String[] str_sp = tuples_a.get(i).split(" ");
      rand_center = -1;
      if (y < num_of_centroids) {
        rand_center = rand.nextInt() % 1;
      }
      for (int k = 0; k < dimension_of_points; k++) {
        if (rand_center == 0) {
          cluster_center[y][k] = Double.parseDouble(str_sp[k]);
          // cluster_center[y][k] =
          // Double.parseDouble(str_sp[k + 1].split(":")[1]);
        }
        mat_points[i][k] = Double.parseDouble(str_sp[k]);
        // mat_points[i][k] = Double.parseDouble(str_sp[k + 1].split(":")[1]);
        if (rand_center == 0 && k == dimension_of_points - 1)
          y++;
      }
    }

  }

  public void first_run(String path_file, int count, int num_center,
      String out_file, int Instance) throws IOException {
    prepare_str_2(path_file, count, num_center);

    if (Instance == 1) {
      divider = num_center;
    } else {
      divider = Integer.MAX_VALUE;
    }
    reminder = num_of_points / divider;
    int u;
    boolean flag_inner = false;
    for (u = 0; u < reminder; u++) {
      // System.out.println("U:" + u);
      flag_inner = true;
      fcm(u * divider, u * divider + divider);
      // round_normal_center_weight();
      if (u != 0) {
        converge_prev_cur_cntrs();
      }
      update_centers_weight_for_next_iter();
    }
    int modular = num_of_points % divider;
    if (modular != 0) {
      fcm(u * divider, u * divider + modular);
      // round_normal_center_weight();
      if (flag_inner) {
        converge_prev_cur_cntrs();
      }
    }
    Path pt = new Path(out_file + "/initialize_center");
    FileSystem fs = FileSystem.get(new Configuration());
    BufferedWriter br =
        new BufferedWriter(new OutputStreamWriter(fs.create(pt, true)));

    for (int i = 0; i < num_of_centroids; i++) {
      StringBuilder str_builder = new StringBuilder();
      for (int j = 0; j < dimension_of_points; j++) {
        // System.out.println(cluster_center[i][j]);
        str_builder.append(Double.toString(cluster_center[i][j]));
        str_builder.append(" ");
      }
      str_builder.append("\n");
      br.write(str_builder.toString());
    }
    br.close();
  }

  boolean fcm(int init_loop, int finalize_loop) {
    int iteration = 0;
    double max_diff = 0.0;
    do {
      max_diff = update_centroids_first(init_loop, finalize_loop);
      iteration++;
    } while (max_diff > epsilon && iteration < fiterations);
    return true;
  }

  private Double update_centroids_first(int init_loop, int finalize_loop) {
    double max_diff = 0;
    double[][] temp_centroid =
        new double[num_of_centroids][dimension_of_points];
    double denom3;
    double u = 0.0;
    double temp_diff = 0.0;
    double new_center = 0.0;
    for (int q = 0; q < num_of_centroids; q++) {
      center_weight[q] = 0.0;
      for (int w = 0; w < dimension_of_points; w++) {
        temp_centroid[q][w] = 0.0;
      }
    }
    for (int k = init_loop; k < finalize_loop; k++) {
      denom3 = 0.0;
      for (int i = 0; i < num_of_centroids; i++) {
        dsqr[i] = cal_Euc_distnce(k, i);
        numer3[i] = Math.pow(dsqr[i], 2 / ((double) fuzziness - 1));
        denom3 += 1 / (double) numer3[i];
      }// end of for i to number of centroids

      for (int i = 0; i < num_of_centroids; i++) {
        u = Math.pow(denom3 * numer3[i], -1 * fuzziness);
        for (int d = 0; d < dimension_of_points; d++) {
          temp_centroid[i][d] += mat_points[k][d] * u;
        }// end of
        center_weight[i] += u;
      }// end of for i to number of centroids
    }// end of for k to number of points
    for (int i = 0; i < num_of_centroids; i++) {
      for (int d = 0; d < dimension_of_points; d++) {
        new_center = temp_centroid[i][d] / center_weight[i];
        temp_diff = Math.abs(cluster_center[i][d] - new_center);
        if (temp_diff > max_diff) {
          max_diff = temp_diff;
        }
        cluster_center[i][d] = new_center;
      }
    }
    return max_diff;

  }

  private Double update_centroids_w() {
    double max_diff = 0;
    double[][] temp_centroid =
        new double[num_of_centroids][dimension_of_points];
    double denom3;
    double u = 0.0;
    double temp_diff = 0.0;
    double new_center = 0.0;
    for (int q = 0; q < num_of_centroids; q++) {
      center_weight[q] = 0.0;
      for (int w = 0; w < dimension_of_points; w++) {
        temp_centroid[q][w] = 0.0;
      }
    }
    for (int k = 0; k < num_of_centroids * 2; k++) {
      denom3 = 0.0;
      for (int i = 0; i < num_of_centroids; i++) {
        dsqr[i] = cal_Euc_distnce_for_centroid(k, i);
        numer3[i] = Math.pow(dsqr[i], 2 / ((double) fuzziness - 1));
        denom3 += 1 / (double) numer3[i];
      }// end of for i to number of centroids

      for (int i = 0; i < num_of_centroids; i++) {
        u = Math.pow(denom3 * numer3[i], -1 * fuzziness);
        for (int d = 0; d < dimension_of_points; d++) {
          temp_centroid[i][d] +=
              center_points[k][d] * u
                  * total_center_weight_from_prev_and_curr_operation[k];
        }// end of
        center_weight[i] +=
            u * total_center_weight_from_prev_and_curr_operation[k];
      }// end of for i to number of centroids
    }// end of for k to number of points
    for (int i = 0; i < num_of_centroids; i++) {
      for (int d = 0; d < dimension_of_points; d++) {
        new_center = temp_centroid[i][d] / center_weight[i];
        temp_diff = Math.abs(cluster_center[i][d] - new_center);
        if (temp_diff > max_diff) {
          max_diff = temp_diff;
        }
        cluster_center[i][d] = new_center;
      }
    }
    return max_diff;

  }

  private Double cal_Euc_distnce_for_centroid(Integer i, Integer j) {
    int k;
    Double sum = 0.0;
    for (k = 0; k < dimension_of_points; k++) {
      double x = cluster_center[j][k];
      double y = center_points[i][k];
      if (y - x == 0) {
        sum += Math.pow(0.00001, 2);
      } else {
        sum += Math.pow(y - x, 2);
      }

    }
    return Math.sqrt(sum);
  }// end of calculate the norm


  boolean wwfcm() {
    int iteration = 0;
    double max_diff = 0.0;
    do {
      max_diff = update_centroids_w();
      iteration++;
    } while (max_diff > epsilon && iteration < fiterations);
    return true;
  }

  void update_centers_weight_for_next_iter() {
    for (int i = 0; i < num_of_centroids; i++) {
      for (int j = 0; j < dimension_of_points; j++) {
        cluster_center_prev[i][j] = cluster_center[i][j];
      }
      center_weight_from_prev_iter[i] = center_weight[i];
    }
  }

  void converge_prev_cur_cntrs() {
    for (int i = 0; i < num_of_centroids; i++) {
      for (int j = 0; j < dimension_of_points; j++) {
        center_points[i][j] = cluster_center[i][j];
      }
      total_center_weight_from_prev_and_curr_operation[i] = center_weight[i];
    }
    for (int i = num_of_centroids; i < 2 * num_of_centroids; i++) {
      for (int j = 0; j < dimension_of_points; j++) {
        center_points[i][j] = cluster_center_prev[i - num_of_centroids][j];
      }
      total_center_weight_from_prev_and_curr_operation[i] =
          center_weight_from_prev_iter[i - num_of_centroids];
    }
    wwfcm();
    // round_normal_center_weight();
  }

  void round_normal_center_weight() {
    for (int i = 0; i < num_of_centroids; i++) {
      center_weight[i] = (double) Math.round(center_weight[i]);
    }

  }

  void RunJob(Configuration conf, String input, String output,
      String centers_num_str, String dimensions_str, String Epsilon_str,
      String fuzziness_str, String Seperator_str, String num_of_instance,
      String iterations_str) throws IllegalArgumentException, IOException,
      ClassNotFoundException, InterruptedException {

    int driver_num_instance = Integer.valueOf(num_of_instance);
    int driver_centers = Integer.valueOf(centers_num_str);
    conf.setInt("centers", driver_centers);
    int driver_dims = Integer.valueOf(dimensions_str);
    conf.setInt("dimensions", driver_dims);
    Double driver_epsilon = Double.valueOf(Epsilon_str);
    conf.setDouble("Epsilon", driver_epsilon);
    Double driver_fuzziness = Double.valueOf(fuzziness_str);
    conf.setDouble("fuzziness", driver_fuzziness);
    int driver_iterations = Integer.valueOf(iterations_str);
    conf.setInt("iterations", driver_iterations);
    conf.setInt("Thompson", driver_num_instance);
    conf.set("Seprator", Seperator_str);
    Job job = Job.getInstance(conf, "JobName");
    job.setJarByClass(PwfcmDriver.class);
    job.setMapperClass(PwfcmMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);
    job.setCombinerClass(PwfcmCombiner.class);

    job.setReducerClass(PwfcmReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    FileSystem fs = FileSystem.get(conf);
    RemoteIterator<LocatedFileStatus> fileStatusListIterator =
        fs.listFiles(new Path(input), false);
    String str = null;
    while (fileStatusListIterator.hasNext()) {
      LocatedFileStatus fileStatus = fileStatusListIterator.next();
      str = fileStatus.getPath().toString();
    }
    initialize_vars(driver_centers, driver_dims, driver_fuzziness,
        driver_epsilon, driver_iterations);

    long first_StartTime = System.currentTimeMillis();
    first_run(str, driver_num_instance, driver_centers,
        "/usr/local/hadoop/sample", 0);
    long first_EndTime = System.currentTimeMillis();

    long Second_StartTime = System.currentTimeMillis();
    first_run(str, driver_num_instance, driver_centers,
        "/usr/local/hadoop/sample", 1);
    long Second_EndTime = System.currentTimeMillis();
    
    if (first_EndTime - first_StartTime > Second_EndTime - Second_StartTime) {
      conf.setInt("Instance", 1);
    } else {
      conf.setInt("Instance", 1);
    }

    job.addCacheFile(new Path("/usr/local/hadoop/sample/initialize_center")
        .toUri());

    if (!job.waitForCompletion(true)) {
      System.out.println("finished unnormally!");
      return;
    }

    last_run(output);
    System.out.println("finished normally!");
  }

  public static void main(String[] args) throws Exception {

    String input = args[0];
    String output = args[1];
    String centers_num_str = args[2];
    String dimensions_str = args[3];
    String Epsilon_str = args[4];
    String fuzziness_str = args[5];
    String Seperator_str = args[6];
    String num_of_instance = args[7];
    String iterations_str = args[8];
    Configuration conf = new Configuration();
    PwfcmDriver P = new PwfcmDriver();
    P.RunJob(conf, input, output, centers_num_str, dimensions_str, Epsilon_str,
        fuzziness_str, Seperator_str, num_of_instance, iterations_str);

  }
}
