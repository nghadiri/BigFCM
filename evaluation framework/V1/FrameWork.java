import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.hadoop.fs.FileSystem;
import java.util.Arrays;
import java.util.Random;
import java.util.Stack;

public class FrameWork {
  
  int dimension =0;
  int number_Of_instance =0;
  Double[][] instance_points = null;
  BufferedWriter writer = null;
  BufferedWriter writerOutPut = null;
  BufferedWriter writerOutPut2 = null;
  BufferedReader reader = null;
  BufferedReader readerEpsilon = null;
  BufferedReader readerCenters = null;
  String[] SamplePath = null;
  String[] InputSamplePath = null;
  String[] InputSampleSeqPath = null;
  String ClustersDirectory = null;
  int NumberOfSample = 0;
  double epsilon = 0.0;
  int iterations = 0;
  Configuration conf = null;
  PwfcmDriver BigFCM = null;
  BufferedReader reader_in_seq = null;
  FileSystem fs = null;
  int counter = 0;
  String[] EpsilonArray = null;
  String[] CentersArray = null;

  boolean FillEpsilonArray(String EpsilonPath) throws IOException {
    Stack<String> st= new Stack<>();
    readerEpsilon= new BufferedReader(new FileReader(EpsilonPath));
    String line;
    while ((line = readerEpsilon.readLine()) != null) {
      
      st.push(line);
    }
    readerEpsilon.close();
    int EpsilonArraySize = st.size();
    EpsilonArray= new String[EpsilonArraySize];
    for (int i=0;i<EpsilonArraySize;i++)
    {
      EpsilonArray[i]=st.pop();
    }
    
    return true;
  }

  boolean FillCentersArray(String CentersPath) throws IOException {
    Stack<String> st = new Stack<>();
    readerCenters = new BufferedReader(new FileReader(CentersPath));
    String line;
    while ((line = readerCenters.readLine()) != null) {

      st.push(line);
    }
    readerCenters.close();
    int CentersArraySize = st.size();
    CentersArray = new String[CentersArraySize];
    for (int i = 0; i < CentersArraySize; i++) {
      CentersArray[i] = st.pop();
    }

    return true;
  }


  boolean InitSampAlg(int NumOFIns, int dim) {
    // Declare Matrix Point and Allocate it
    number_Of_instance = NumOFIns;
    dimension = dim;
    instance_points = new Double[number_Of_instance][dimension];
    return true;
  }

  boolean RunSamplingAlg(String DataSetPath, int NumOFIns) throws IOException
  {
    reader = new BufferedReader(new FileReader(DataSetPath));
    String line;
    int counter =0;
    String[] c;
    while ((line = reader.readLine()) != null  && counter!=NumOFIns) {
      c = line.split(",");
      for (int i=0;i<c.length;i++)
      {
        instance_points[counter][i]=Double.parseDouble(c[i]);
        
      }
      counter++;
    }
    
    Random rand = new Random();
    int GenRandom = 0;
    while((line = reader.readLine()) != null)
    {
      GenRandom = rand.nextInt(counter);
      if (GenRandom < number_Of_instance) {
        c = line.split(",");
        for (int i = 0; i < c.length; i++) {
          instance_points[GenRandom][i] = Double.parseDouble(c[i]);
        }
      }
      counter++;
    }
    
    return true;
  }

  boolean WriteToFile(String WPath, int count) throws IOException {
    WPath = WPath + "Sample" + Integer.toString(count);
    SamplePath[count] = WPath;
    writer = new BufferedWriter(new FileWriter(SamplePath[count]));

    for (int i = 0; i < number_Of_instance; i++) {
      for (int j = 0; j < dimension; j++) {
        writer.write(Double.toString(instance_points[i][j]));
        if (j < dimension - 1) {
          writer.write(",");
        }
      }
      writer.newLine();
    }
    writer.close();
    return true;
  }

  boolean run_Sampling(String DatasetPath, int NumOFInstance, int repNUM,
      String sampleDirPath, int dimensionInput) throws IOException {
    SamplePath = new String[repNUM];
    for (int i = 0; i < repNUM; i++) {
      InitSampAlg(NumOFInstance, dimensionInput);
      RunSamplingAlg(DatasetPath, NumOFInstance);
      WriteToFile(sampleDirPath, i);
    }
    return true;
  }

  boolean run(String DataSetPath, int NumOFInstance, int repNum,
      String SampleDirPath, int dimensionInput, String centers_num_str,
      String Epsilon_str, String fuzziness_str, String Seperator_str,
      String num_of_instance, String iterations_str, int repeatPerEachSamples,
      String EpsilonPath, String CentersPath)
      throws Exception {

    FillEpsilonArray(EpsilonPath);
    System.out.println(Arrays.toString(EpsilonArray));
    FillCentersArray(CentersPath);
    System.out.println(Arrays.toString(CentersArray));
    writerOutPut =
        new BufferedWriter(new FileWriter(SampleDirPath
            + "finalResult-ChangingCenters.csv"));
    writerOutPut2 =
        new BufferedWriter(new FileWriter(SampleDirPath
            + "finalResult-ChangingEpsilon.csv"));
    long startTime;
    long endTime;
    conf = new Configuration();
    fs = FileSystem.get(conf);
    BigFCM = new PwfcmDriver();
    run_Sampling(DataSetPath, NumOFInstance, repNum, SampleDirPath,
        dimensionInput);

    ConvertTextToSequential(repNum, SampleDirPath, Seperator_str);
    CopyFromLocalToHost(repNum);
    for (int i = 0; i < repNum; i++) {
      for (int z = 0; z < CentersArray.length; z++) {
        for (int u = 0; u < repeatPerEachSamples; u++) {
          startTime = System.currentTimeMillis();
          System.out
              .println(RunMahoutKmeans(u, i, Integer.parseInt(CentersArray[z]),
                  Double.parseDouble(Epsilon_str),
                  Integer.parseInt(iterations_str)));
          endTime = System.currentTimeMillis();
          writerOutPut.write("MahoutKmeans" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + CentersArray[z] + "," + Epsilon_str
              + "," + Long.toString(endTime - startTime));
          writerOutPut.newLine();
          writerOutPut.flush();

          startTime = System.currentTimeMillis();
          System.out
              .println(RunMahoutFuzzyKmeans(u, i,
                  Integer.parseInt(CentersArray[z]),
                  Double.parseDouble(Epsilon_str),
                  Integer.parseInt(iterations_str)));
          endTime = System.currentTimeMillis();
          writerOutPut.write("MahoutFKmeans" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + CentersArray[z] + "," + Epsilon_str
              + "," + Long.toString(endTime - startTime));
          writerOutPut.newLine();
          writerOutPut.flush();
          startTime = System.currentTimeMillis();
          RunBigFCM(u, i, dimensionInput, Epsilon_str, fuzziness_str,
              CentersArray[z], Seperator_str, num_of_instance, iterations_str);
          endTime = System.currentTimeMillis();
          writerOutPut.write("BIGFCM" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + CentersArray[z] + "," + Epsilon_str
              + "," + Long.toString(endTime - startTime));
          writerOutPut.newLine();
          writerOutPut.flush();
        }
      }

    }
    writerOutPut.close();
    for (int i = 0; i < repNum; i++) {
      for (int z = 0; z < EpsilonArray.length; z++) {
        for (int u = 0; u < repeatPerEachSamples; u++) {
          startTime = System.currentTimeMillis();
          System.out.println(RunMahoutKmeans(u, i,
              Integer.parseInt(centers_num_str),
              Double.parseDouble(EpsilonArray[z]),
              Integer.parseInt(iterations_str)));

          endTime = System.currentTimeMillis();
          writerOutPut2.write("MahoutKmeans" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + centers_num_str + ","
              + EpsilonArray[z] + "," + Long.toString(endTime - startTime));
          writerOutPut2.newLine();
          writerOutPut2.flush();
          startTime = System.currentTimeMillis();
          System.out.println(RunMahoutFuzzyKmeans(u, i,
              Integer.parseInt(centers_num_str),
              Double.parseDouble(EpsilonArray[z]),
              Integer.parseInt(iterations_str)));

          endTime = System.currentTimeMillis();
          writerOutPut2.write("MahoutFKmeans" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + centers_num_str + ","
              + EpsilonArray[z] + "," + Long.toString(endTime - startTime));
          writerOutPut2.newLine();
          writerOutPut2.flush();
          startTime = System.currentTimeMillis();

          RunBigFCM(u, i, dimensionInput, EpsilonArray[z], fuzziness_str,
              centers_num_str, Seperator_str, num_of_instance, iterations_str);
          endTime = System.currentTimeMillis();
          writerOutPut2.write("BIGFCM" + "," + Integer.toString(i) + ","
              + Integer.toString(u) + "," + centers_num_str + ","
              + EpsilonArray[z] + "," + Long.toString(endTime - startTime));
          writerOutPut2.newLine();
          writerOutPut2.flush();
        }
      }

    }
    writerOutPut2.close();
    System.out.println("Finished!");
    return true;
  }

  boolean CopyFromLocalToHost(int NumRep) throws IOException {
    Path WorkingDirectoryPath = fs.getWorkingDirectory();
    ClustersDirectory = new String();
    ClustersDirectory = WorkingDirectoryPath + "/clusters";
    fs.mkdirs(new Path(ClustersDirectory));
    InputSamplePath = new String[NumRep];
    for (int i = 0; i < NumRep; i++) {
      InputSamplePath[i] =
          WorkingDirectoryPath + "/" + "sample-" + Integer.toString(i);
      fs.mkdirs(new Path(InputSamplePath[i]));
      fs.copyFromLocalFile(new Path(SamplePath[i]),
          new Path(InputSamplePath[i]));
    }
    return true;
  }

  boolean ConvertTextToSequential(int repNum, String pathFolder, String Delim)
      throws IllegalArgumentException, IOException {
    InputSampleSeqPath = new String[repNum];
    Path WorkingDirectoryPath = fs.getWorkingDirectory();
    for (int k = 0; k < repNum; k++) {

      InputSampleSeqPath[k] =
          WorkingDirectoryPath + "/" + "sampleseq-" + Integer.toString(k);
      fs.mkdirs(new Path(InputSampleSeqPath[k]));
      reader_in_seq = new BufferedReader(new FileReader(SamplePath[k]));
      SequenceFile.Writer writer =
          SequenceFile.createWriter(
              conf,
              Writer.file(new Path(InputSampleSeqPath[k] + "/" + "sampleseq"
                  + Integer.toString(k))), Writer.keyClass(LongWritable.class),
              Writer.valueClass(VectorWritable.class));
      String line;
      long counter = 0;
      while ((line = reader_in_seq.readLine()) != null) {
        String[] c = line.split(Delim);
        if (c.length == dimension) {
          double[] d = new double[c.length];
          for (int i = 0; i < c.length; i++) {
            d[i] = Double.parseDouble(c[i]);
          }
          Vector vec = new RandomAccessSparseVector(c.length);
          vec.assign(d);
          VectorWritable writable = new VectorWritable();
          writable.set(vec);
          writer.append(new LongWritable(counter++), writable);
        }
      }
      writer.close();
    }
    return true;
  }

  String RunMahoutKmeans(int RPS, int index, int num_of_centroid,
      double Epsilon,
      int iteration) throws ClassNotFoundException, IllegalArgumentException,
      IOException, InterruptedException {
    Path centroids =
        RandomSeedGenerator.buildRandom(conf, new Path(
            InputSampleSeqPath[index]), new Path(ClustersDirectory),
            num_of_centroid,
            new EuclideanDistanceMeasure());
    String output =
        fs.getWorkingDirectory() + "/outputseq-MahoutKM-"
            + Integer.toString(index) + "-" + Integer.toString(RPS) + "-"
            + Double.toString(Epsilon) + "-"
            + Integer.toString(num_of_centroid) + "-"
            + Integer.toString(counter);
    KMeansDriver.run(conf, new Path(
InputSampleSeqPath[index]), centroids,
        new Path(output), Epsilon, iteration, true, 0.0, false);
    counter++;
    return output;
  }
  
  String RunMahoutFuzzyKmeans(int RPS, int index, int num_of_centroid,
      double Epsilon,
      int iteration) throws Exception {
    Path centroids =
        RandomSeedGenerator.buildRandom(conf, new Path(
            InputSampleSeqPath[index]), new Path(ClustersDirectory),
            num_of_centroid,
            new EuclideanDistanceMeasure());
    String output =
        fs.getWorkingDirectory() + "/outputseq-MahoutFKM-"
            + Integer.toString(index) + "-" + Integer.toString(RPS) + "-"
            + Double.toString(Epsilon) + "-"
            + Integer.toString(num_of_centroid) + "-"
            + Integer.toString(counter);
    FuzzyKMeansDriver.run(conf, new Path(
InputSampleSeqPath[index]), centroids,
        new Path(output), Epsilon, iteration, (float) 2.0,false, false, 0.0, false);
    ClusterDumper clusterDumper =
        new ClusterDumper(new Path(output, "clusters-*-final"), new Path(
            output, "clusteredPoints"));
    clusterDumper.printClusters(null);
    counter++;
    return output;
  }

  String RunBigFCM(int RPS, int index, int dimensionInput, String Epsilon_str,
      String fuzziness_str, String centers_num_str,
      String Seperator_str, String num_of_instance,
 String iterations_str)
      throws IllegalArgumentException, ClassNotFoundException, IOException,
      InterruptedException {

    String output =
        fs.getWorkingDirectory() + "/outputBIGFCM" + "-"
            + Integer.toString(index) + "-" + Integer.toString(RPS) + "-"
            + centers_num_str + "-"
 + Epsilon_str + "-"
            + Integer.toString(counter);
    BigFCM.RunJob(conf, InputSamplePath[index], output, centers_num_str,
        Integer.toString(dimensionInput), Epsilon_str, fuzziness_str,
        Seperator_str, num_of_instance, iterations_str);
    counter++;
    return output;
  }

  public static void main(String[] args) throws Exception {
    FrameWork f = new FrameWork();
    f.run(args[0], Integer.parseInt(args[1]), Integer.parseInt(args[2]),
        args[3], Integer.parseInt(args[4]), args[5], args[6], args[7], args[8],
        args[9], args[10], Integer.parseInt(args[11]), args[12], args[13]);

  }
}


