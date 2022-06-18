import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;


public class PwfcmMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
  public static int counter = 0;
  private String str = null;
  private String spl = null;
  private int set_num_dim = 0;
  private String[] str_token = null;
  private StringBuilder str_builder = new StringBuilder();
  public void map(LongWritable ikey, Text ivalue, Context context)
      throws IOException, InterruptedException {
    set_num_dim = context.getConfiguration().getInt("dimensions", 0);
    spl = context.getConfiguration().get("Seprator", " ");
    str = ivalue.toString();
    str_token = str.split(spl);
    int size_str_tok = str_token.length;
    if (size_str_tok == set_num_dim) {
      for (int i = 0; i < size_str_tok; i++) {
        str_builder.append(str_token[i]);
        str_builder.append(" ");
      }
      context.write(new IntWritable(counter % 1),
          new Text(str_builder.toString()));
      str_builder.setLength(0);
      counter++;
    }
  }
}// end of class
