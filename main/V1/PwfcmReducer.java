import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;


public class PwfcmReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
  // private Vector<PwfcmTuple> tuples = new Vector<PwfcmTuple>();
  private Vector<String> tuples_a = null;
  private Integer dimension_of_points = 0;
  private Integer num_of_points;
  private Integer num_of_centroids = 0;
  private Double[][] mat_points = null;
  private Double epsilon = 0.00;
  private Double fuzziness = 0.0;
  private Double[][] cluster_center = null;
  private Double[] center_weight = null;
  private Double[] center_weight_from_combiner = null;
  private Double[] dsqr = null;
  private Double[] numer3 = null;
  int fiteration = 0;

  public void reduce(IntWritable _key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
    dimension_of_points = context.getConfiguration().getInt("dimensions", 0);
    num_of_centroids = context.getConfiguration().getInt("centers", 0);
    fuzziness = context.getConfiguration().getDouble("fuzziness", 0.0);
    epsilon = context.getConfiguration().getDouble("Epsilon", 0.0);
    fiteration = context.getConfiguration().getInt("iterations", 0);
    StringBuilder out_str = new StringBuilder();
    tuples_a = new Vector<String>();
    for (Text text_list : values) {
      tuples_a.add(text_list.toString());
    }
    num_of_points = tuples_a.size();
    prepare_mat();
    // process values
    URI[] url = context.getCacheFiles();
    String temp = url[0].toString();
    Path ofile = new Path(temp);
    FileSystem fs = FileSystem.get(new Configuration());
    BufferedReader br =
        new BufferedReader(new InputStreamReader(fs.open(ofile)));

    String l = br.readLine();
    int ii = 0;
    int jj;
    while (l != null) {
      String[] split_line = l.split(" ");
      jj = 0;
      for (String mover : split_line) {
        cluster_center[ii][jj] = Double.parseDouble(mover);
        jj++;
      }
      l = br.readLine();
      ii++;
    }
    br.close();

    boolean status_1 = fcm();
    // round_normal_center_weight();
    // ///////////sending key and values
    if (status_1) {
      for (int i = 0; i < num_of_centroids; i++) {
        out_str.setLength(0);
        for (int j = 0; j < dimension_of_points; j++) {
          out_str.append(Double.toString(cluster_center[i][j]));
          out_str.append(" ");
        }
        out_str.append(Double.toString(center_weight[i]));
        context.write(_key, new Text(out_str.toString()));
      }
    }
  }

  // ///preparing matrix

  private void prepare_mat() {
    mat_points = new Double[num_of_points][dimension_of_points];
    center_weight_from_combiner = new Double[num_of_points];
    String[] split_str = null;
    for (int k = 0; k < num_of_points; k++) {
      split_str = tuples_a.get(k).split(" ");
      for (int i = 0; i < dimension_of_points; i++) {
        mat_points[k][i] = Double.parseDouble(split_str[i]);
      }
      center_weight_from_combiner[k] =
          Double.parseDouble(split_str[dimension_of_points]);
    }

    cluster_center = new Double[num_of_centroids][dimension_of_points];
    center_weight = new Double[num_of_centroids];
    dsqr = new Double[num_of_centroids];
    numer3 = new Double[num_of_centroids];
  }// /end of prepare function
   // /////////////////prepare memberships
   // ////////////////////////////new method////////////////////////

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

  private Double update_centroids() {
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
          temp_centroid[i][d] +=
              mat_points[k][d] * u * center_weight_from_combiner[k];
        }// end of
        center_weight[i] += u * center_weight_from_combiner[k];
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
  boolean fcm() {
    int iteration = 0;
    double max_diff = 0.0;
    do {
      max_diff = update_centroids();
      iteration++;
    } while (max_diff > epsilon && iteration < fiteration);
    return true;
  }

  void round_normal_center_weight() {
    for (int i = 0; i < num_of_centroids; i++) {
      center_weight[i] = (double) Math.round(center_weight[i]);
    }
  }

}
