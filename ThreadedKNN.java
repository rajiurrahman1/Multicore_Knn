/*
 * Implementation of multi threaded K-nearest neighbour algorithm
 * Here, the algorithm has been parallelized two times, 
 * first time for distance calculation between train and test data and second time for finding k neighbours who are nearest
 * 
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
/**
 *
 * @author Rajiur Rahman
 * Department of CS, Wayne State University
 * rajiurrahman.bd@gmail.com
 * 
 */

class RunnableKnn implements Runnable {
    private Thread t;
    int minInxex, maxIndex, numTestRow, numCol;
    private String threadName;
    double[][] trainInsideThread = {{1.0}, {2.0}};
    double[][] testInsideThread  = {{1.0}, {2.0}};
    double[][] distancesInsideThread = {{1.0}, {2.0}};
    
    RunnableKnn( String name, int mnIndex, int mxIndex, int ntr, int c){
        this.minInxex = mnIndex;
        this.maxIndex = mxIndex;
        this.numTestRow = ntr;
        this.numCol = c;
        this.threadName = name;
    }    
    public void run(){
        String threadName = t.getName();
        System.out.println("Inside thread - " + threadName+ "\tWorking with data rows "+ minInxex+"-"+maxIndex);
        this.distancesInsideThread = getDistance();
        
        
    }
    public double[][] getDistanceFromThread() throws InterruptedException{
        t.join();
        return this.distancesInsideThread;
    }    
    public void start() throws InterruptedException{
      if (t == null){
         t = new Thread (this, threadName);
         t.start ();
      }
    }
    public void setTrainData(int startIndex, int endIndex, int numCol, double[][] data){
        this.trainInsideThread = new double[(endIndex-startIndex)][numCol];
        for(int i=0; i<(endIndex-startIndex); i++){
            for(int j=0; j<numCol; j++){
                this.trainInsideThread[i][j] = data[startIndex+i][j];
            }
        }
    }    
    public void setTestData(int numTestRow, int numCol, double[][] data){
        this.testInsideThread = new double[numTestRow][numCol];
        for(int i=0; i<numTestRow; i++){
            for(int j=0; j<numCol; j++){
                this.testInsideThread[i][j] = data[i][j];
            }
        }
    }
    public void initializeDistanceMatrix(int minIndex, int maxIndex, int numTestRow){
        this.distancesInsideThread = new double[numTestRow][maxIndex-minIndex];
    }
    private double[][] getDistance() {
        // this method will set the distance variable
        // each row represents each test instance, each column for each training instance 
        for(int i=0; i<this.distancesInsideThread.length; i++){
            for(int j=0; j<(this.maxIndex-this.minInxex); j++){
                this.distancesInsideThread[i][j] = getDistanceRows(i,j);
            }
        }
        
        return this.distancesInsideThread;
    }
    private double getDistanceRows(int testID, int trainID) {
        double d1 = 0.0;
        for(int k=0; k<this.numCol; k++){
            d1+= Math.pow( (this.trainInsideThread[trainID][k] - this.testInsideThread[testID][k] ) , 2);
        }
        
        return Math.sqrt(d1);
    }    
}

class SortDistances implements Runnable{
    private Thread t;
    private String threadName;
    int k; 
    int numTrainRows;
    double[][] rxDistances = {{1.0}, {2.0}};
    int[][] nnIndices = {{1}, {2}};

    public SortDistances(String tName, int k1, int ntr) {
        this.threadName = tName;
        this.k = k1;
        this.numTrainRows = ntr;
    }
    public void run(){
        //find k nearest neighbor for each row - here each row is each test instances
        
        System.out.println("Inside thread "+t.getName() + " for calculating k nearest neighbors from indices");
        for(int i=0; i<this.nnIndices.length; i++){
            for(int j=0; j<this.k; j++){
                this.nnIndices[i][j] = getNearestNeighborIndex(i);
                
            }        
        }
        
    }
    public void start() throws InterruptedException{
        if (t == null){
            t = new Thread (this, threadName);
            t.start ();
      }
    }
    public int[][] getNearestNeighborFromThread() throws InterruptedException{
        t.join();
        return this.nnIndices;
    } 
    public void setDistance(double[][] distance, int minIndex, int maxInidex, int numTrainRows){
        this.rxDistances = new double[(maxInidex-minIndex)][numTrainRows];
        for(int i=0; i<(maxInidex-minIndex); i++ ){
            for(int j=0; j<numTrainRows; j++){
                this.rxDistances[i][j] = distance[minIndex+i][j];
            }
        }
        
    }
    public void initializeNNIndices(int minIindex, int maxIndex, int k){
        this.nnIndices = new int[(maxIndex-minIindex)][k];
    }
    private int getNearestNeighborIndex(int testRow) {
        double minValue = Double.POSITIVE_INFINITY;
        int minIndex = 0;
        for(int i=0; i<this.numTrainRows; i++){
            if( this.rxDistances[testRow][i] < minValue ){
                minIndex = i;
                minValue = this.rxDistances[testRow][i];
            }
        }
//        System.out.println("Min index: "+minIndex);
        this.rxDistances[testRow][minIndex] = Double.POSITIVE_INFINITY;
        
        return minIndex;
    }
}

public class ThreadedKNN {
    private static boolean isNumeric(String str){  
        try{
            double d = Double.parseDouble(str);  
        }  
        catch(NumberFormatException nfe){  
            return false;  
        }  
        return true;  
    }    
    private static double[][] readData(int numRow, int numCol, String dataFileName ) throws FileNotFoundException{
        
        //helping link http://stackoverflow.com/questions/22185683/read-txt-file-into-2d-array
        double [][] matrix = new double[numRow][numCol];
        String filename = dataFileName;
        File inFile = new File(filename);
        Scanner in = new Scanner(inFile);
     
        int lineCount = 0;
        while (in.hasNextLine()) {
            String[] currentLine = in.nextLine().trim().split("\\s+"); 
            for (int i = 0; i < currentLine.length; i++) {
                if(isNumeric(currentLine[i])){
//                    System.out.println(i + "-" + currentLine[i]);
                    matrix[lineCount][i] = Double.parseDouble(currentLine[i]);    
                }
                else{
                      matrix[lineCount][i] = (double)0.0;    
                }
            }
            lineCount++;
        }
        
        return matrix;
    } 
    private static void dumpMatrix(double[][] matrix, int k, int numCol) {
        System.out.println("\nPrinting Matrix");
        for(int i=0; i<k; i++){
            for(int j=0; j<numCol; j++){
                System.out.print(matrix[i][j]+" ");
            }
            System.out.println();
        }
    }   
    private static void dumpIntMatrix(int[][] matrix, int k, int numCol) {
        System.out.println("\nPrinting Integer Matrix");
        for(int i=0; i<k; i++){
            for(int j=0; j<numCol; j++){
                System.out.print(matrix[i][j]+" ");
            }
            System.out.println();
        }
    } 
    public static double[][] accumulateDistances(double[][] distances, double[][] rDistances, int numTestRow, int minIndex, int maxIndex){
        //add a part of the r_x_distance values to the main distance variable
        for(int i=0; i<numTestRow; i++){
            for(int j=minIndex; j<maxIndex; j++){
                distances[i][j] = rDistances[i][j-minIndex];
            }
        }
        
        
        return distances;
    }        
    public static int[][] accumulateNerestNeighborIndices(int[][]nnIndices, int[][] sIndices, int minRows, int maxRows, int k){
        for(int i=minRows; i<maxRows; i++){
            for(int j=0; j<k; j++){
                nnIndices[i][j] = sIndices[(i-minRows)][j];
            }
        }
        return nnIndices;
    }    
    private static int predictLabel(double[][] trainData, int[][] nnIndices, int numCol, int testRow, int k) {
        int sum1 =0;
        int pLabel = 0;
        for(int i=0; i<k; i++){
            int currentIndex = nnIndices[testRow][i];
            sum1 += trainData[currentIndex][numCol];
        }
        if(2*sum1 > k){
            pLabel = 1;
        }
//        System.out.println("Sum: "+sum1+ "\tLabel: "+pLabel);
        return pLabel;
    }
    
    
    //usage java -Xmx14g -jar blahblah.jar 10k_train.txt 10k_test.txt 8000 2000 10000 5
    public static void main(String[] args) throws FileNotFoundException, InterruptedException{
        System.out.println("Hello world from Multi thread K-NN!\n");
        long startTime = System.currentTimeMillis();
//        final int numTrainRow = 69, numTestRow = 31, numCol = 1000, k=5;
//        final String trainDataFileName = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\train.txt";
//        final String testDataFileName  = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\test.txt";
        
//        final int numTrainRow = 20, numTestRow = 4, numCol = 4, k=5;
//        final String trainDataFileName = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\train1.txt";
//        final String testDataFileName  = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\test1.txt";
        
//        final int numTrainRow = 8000, numTestRow = 2000, numCol = 10000, k=5;
//        final String trainDataFileName = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\10k_train.txt";
//        final String testDataFileName  = "C:\\Dropbox\\Big_Data_Dropbox\\Others\\JavaThreadMultiCore\\TreadTest1\\10k_test.txt";
        
        final String trainDataFileName = args[0].trim();
        final String testDataFileName = args[1].trim();
        final int numTrainRow = Integer.parseInt(args[2]);
        final int numTestRow = Integer.parseInt(args[3]);
        final int numCol = Integer.parseInt(args[4]);
        final int k = Integer.parseInt(args[5]);
        
        //the last column will be the class label //split the trianing data and copy the full test data 
        // call the threads whih will calculate the distances from training to test instances in parallel then accumulate the distances
        double[][] trainData = readData(numTrainRow, numCol+1, trainDataFileName);
        double[][] testData = readData(numTestRow, numCol+1, testDataFileName);
        double[][] distance = new double[numTestRow][numTrainRow]; 
        int[][] nnIndices = new int[numTestRow][k];
        int[] predictedLabels = new int[numTestRow];
        
        //define from which row to which row each thread will work on
        int minIndex1 = 0, maxIndex1 = 615;
        int minIndex2 = 615, maxIndex2 = 1230;
        int minIndex3 = 1230, maxIndex3 = 1845;
        int minIndex4 = 1845, maxIndex4 = 2460;
        int minIndex5 = 2460, maxIndex5 = 3075;
        int minIndex6 = 3075, maxIndex6 = 3690;
        int minIndex7 = 3690, maxIndex7 = 4305;
        int minIndex8 = 4305, maxIndex8 = 4920;
        int minIndex9 = 4920, maxIndex9 = 5535;
        int minIndex10 = 5535, maxIndex10 = 6150;
        int minIndex11 = 6150, maxIndex11 = 6765;
        int minIndex12 = 6765, maxIndex12 = 7380;
        int minIndex13 = 7380, maxIndex13 = 8000;
        
        //declare runnable objects as the threads which will simultaneously work on the training data
        RunnableKnn R1 = new RunnableKnn( "Thread-1", minIndex1, maxIndex1, numTestRow, numCol);        
        RunnableKnn R2 = new RunnableKnn( "Thread-2", minIndex2, maxIndex2, numTestRow, numCol);        
        RunnableKnn R3 = new RunnableKnn( "Thread-3", minIndex3, maxIndex3, numTestRow, numCol);        
        RunnableKnn R4 = new RunnableKnn( "Thread-4", minIndex4, maxIndex4, numTestRow, numCol);        
        RunnableKnn R5 = new RunnableKnn( "Thread-5", minIndex5, maxIndex5, numTestRow, numCol);        
        RunnableKnn R6 = new RunnableKnn( "Thread-6", minIndex6, maxIndex6, numTestRow, numCol);        
        RunnableKnn R7 = new RunnableKnn( "Thread-7", minIndex7, maxIndex7, numTestRow, numCol);        
        RunnableKnn R8 = new RunnableKnn( "Thread-8", minIndex8, maxIndex8, numTestRow, numCol);        
        RunnableKnn R9 = new RunnableKnn( "Thread-9", minIndex9, maxIndex9, numTestRow, numCol);        
        RunnableKnn R10 = new RunnableKnn( "Thread-10", minIndex10, maxIndex10, numTestRow, numCol);        
        RunnableKnn R11 = new RunnableKnn( "Thread-11", minIndex11, maxIndex11, numTestRow, numCol);        
        RunnableKnn R12 = new RunnableKnn( "Thread-12", minIndex12, maxIndex12, numTestRow, numCol);        
        RunnableKnn R13 = new RunnableKnn( "Thread-13", minIndex13, maxIndex13, numTestRow, numCol);        

        //split train data into cores
        R1.setTrainData(minIndex1, maxIndex1, numCol+1, trainData);        
        R2.setTrainData(minIndex2, maxIndex2, numCol+1, trainData);        
        R3.setTrainData(minIndex3, maxIndex3, numCol+1, trainData);        
        R4.setTrainData(minIndex4, maxIndex4, numCol+1, trainData);        
        R5.setTrainData(minIndex5, maxIndex5, numCol+1, trainData);        
        R6.setTrainData(minIndex6, maxIndex6, numCol+1, trainData);        
        R7.setTrainData(minIndex7, maxIndex7, numCol+1, trainData);        
        R8.setTrainData(minIndex8, maxIndex8, numCol+1, trainData);        
        R9.setTrainData(minIndex9, maxIndex9, numCol+1, trainData);        
        R10.setTrainData(minIndex10, maxIndex10, numCol+1, trainData);        
        R11.setTrainData(minIndex11, maxIndex11, numCol+1, trainData);        
        R12.setTrainData(minIndex12, maxIndex12, numCol+1, trainData);        
        R13.setTrainData(minIndex13, maxIndex13, numCol+1, trainData);
        
        //copy the same test data to all the threads
        R1.setTestData(numTestRow, numCol+1, testData);        
        R2.setTestData(numTestRow, numCol+1, testData);        
        R3.setTestData(numTestRow, numCol+1, testData);        
        R4.setTestData(numTestRow, numCol+1, testData);        
        R5.setTestData(numTestRow, numCol+1, testData);        
        R6.setTestData(numTestRow, numCol+1, testData);        
        R7.setTestData(numTestRow, numCol+1, testData);        
        R8.setTestData(numTestRow, numCol+1, testData);        
        R9.setTestData(numTestRow, numCol+1, testData);        
        R10.setTestData(numTestRow, numCol+1, testData);        
        R11.setTestData(numTestRow, numCol+1, testData);        
        R12.setTestData(numTestRow, numCol+1, testData);        
        R13.setTestData(numTestRow, numCol+1, testData);        
        
        //initialize the distance variable inside thread to appropriate size
        R1.initializeDistanceMatrix(minIndex1, maxIndex1, numTestRow);        
        R2.initializeDistanceMatrix(minIndex2, maxIndex2, numTestRow);        
        R3.initializeDistanceMatrix(minIndex3, maxIndex3, numTestRow);        
        R4.initializeDistanceMatrix(minIndex4, maxIndex4, numTestRow);        
        R5.initializeDistanceMatrix(minIndex5, maxIndex5, numTestRow);        
        R6.initializeDistanceMatrix(minIndex6, maxIndex6, numTestRow);        
        R7.initializeDistanceMatrix(minIndex7, maxIndex7, numTestRow);        
        R8.initializeDistanceMatrix(minIndex8, maxIndex8, numTestRow);        
        R9.initializeDistanceMatrix(minIndex9, maxIndex9, numTestRow);        
        R10.initializeDistanceMatrix(minIndex10, maxIndex10, numTestRow);        
        R11.initializeDistanceMatrix(minIndex11, maxIndex11, numTestRow);        
        R12.initializeDistanceMatrix(minIndex12, maxIndex12, numTestRow);        
        R13.initializeDistanceMatrix(minIndex13, maxIndex13, numTestRow);        
        
        //now all the variables necessary are set in the threds, call the threads to start working
        R1.start();                
        R2.start();
        R3.start();
        R4.start();
        R5.start();
        R6.start();
        R7.start();
        R8.start();
        R9.start();
        R10.start();
        R11.start();
        R12.start();
        R13.start();
        
        //now get the distances back from thread
        double[][] r1Distance = R1.getDistanceFromThread();
        double[][] r2Distance = R2.getDistanceFromThread();
        double[][] r3Distance = R3.getDistanceFromThread();
        double[][] r4Distance = R4.getDistanceFromThread();
        double[][] r5Distance = R5.getDistanceFromThread();
        double[][] r6Distance = R6.getDistanceFromThread();
        double[][] r7Distance = R7.getDistanceFromThread();
        double[][] r8Distance = R8.getDistanceFromThread();
        double[][] r9Distance = R9.getDistanceFromThread();
        double[][] r10Distance = R10.getDistanceFromThread();
        double[][] r11Distance = R11.getDistanceFromThread();
        double[][] r12Distance = R12.getDistanceFromThread();
        double[][] r13Distance = R13.getDistanceFromThread();
        
        //accumulate all the distances to a single distance variable
        distance = accumulateDistances(distance, r1Distance, numTestRow, minIndex1, maxIndex1);
        distance = accumulateDistances(distance, r2Distance, numTestRow, minIndex2, maxIndex2);
        distance = accumulateDistances(distance, r3Distance, numTestRow, minIndex3, maxIndex3);
        distance = accumulateDistances(distance, r4Distance, numTestRow, minIndex4, maxIndex4);
        distance = accumulateDistances(distance, r5Distance, numTestRow, minIndex5, maxIndex5);
        distance = accumulateDistances(distance, r6Distance, numTestRow, minIndex6, maxIndex6);
        distance = accumulateDistances(distance, r7Distance, numTestRow, minIndex7, maxIndex7);
        distance = accumulateDistances(distance, r8Distance, numTestRow, minIndex8, maxIndex8);
        distance = accumulateDistances(distance, r9Distance, numTestRow, minIndex9, maxIndex9);
        distance = accumulateDistances(distance, r10Distance, numTestRow, minIndex10, maxIndex10);
        distance = accumulateDistances(distance, r11Distance, numTestRow, minIndex11, maxIndex11);
        distance = accumulateDistances(distance, r12Distance, numTestRow, minIndex12, maxIndex12);
        distance = accumulateDistances(distance, r13Distance, numTestRow, minIndex13, maxIndex13);
        
        //next step split the distance variable row wise to threads and find the nearest k neighbor's indices
        int minRows1 = 0, maxRows1 = 153;
        int minRows2 = 153, maxRows2 = 306;
        int minRows3 = 306, maxRows3 = 459;
        int minRows4 = 459, maxRows4 = 612;
        int minRows5 = 612, maxRows5 = 765;
        int minRows6 = 765, maxRows6 = 918;
        int minRows7 = 918, maxRows7 = 1071;
        int minRows8 = 1071, maxRows8 = 1224;
        int minRows9 = 1224, maxRows9 = 1377;
        int minRows10 = 1377, maxRows10 = 1530;
        int minRows11 = 1530, maxRows11 = 1683;
        int minRows12 = 1683, maxRows12 = 1836;
        int minRows13 = 1836, maxRows13 = 2000;
        
        SortDistances S1 = new SortDistances("thread 1", k, numTrainRow);
        SortDistances S2 = new SortDistances("thread 2", k, numTrainRow);
        SortDistances S3 = new SortDistances("thread 3", k, numTrainRow);
        SortDistances S4 = new SortDistances("thread 4", k, numTrainRow);
        SortDistances S5 = new SortDistances("thread 5", k, numTrainRow);
        SortDistances S6 = new SortDistances("thread 6", k, numTrainRow);
        SortDistances S7 = new SortDistances("thread 7", k, numTrainRow);
        SortDistances S8 = new SortDistances("thread 8", k, numTrainRow);
        SortDistances S9 = new SortDistances("thread 9", k, numTrainRow);
        SortDistances S10 = new SortDistances("thread 10", k, numTrainRow);
        SortDistances S11 = new SortDistances("thread 11", k, numTrainRow);
        SortDistances S12 = new SortDistances("thread 12", k, numTrainRow);
        SortDistances S13 = new SortDistances("thread 13", k, numTrainRow);
        
        
        S1.setDistance(distance, minRows1, maxRows1, numTrainRow);
        S2.setDistance(distance, minRows2, maxRows2, numTrainRow);
        S3.setDistance(distance, minRows3, maxRows3, numTrainRow);
        S4.setDistance(distance, minRows4, maxRows4, numTrainRow);
        S5.setDistance(distance, minRows5, maxRows5, numTrainRow);
        S6.setDistance(distance, minRows6, maxRows6, numTrainRow);
        S7.setDistance(distance, minRows7, maxRows7, numTrainRow);
        S8.setDistance(distance, minRows8, maxRows8, numTrainRow);
        S9.setDistance(distance, minRows9, maxRows9, numTrainRow);
        S10.setDistance(distance, minRows10, maxRows10, numTrainRow);
        S11.setDistance(distance, minRows11, maxRows11, numTrainRow);
        S12.setDistance(distance, minRows12, maxRows12, numTrainRow);
        S13.setDistance(distance, minRows13, maxRows13, numTrainRow);
        
        
        S1.initializeNNIndices(minRows1, maxRows1, numTrainRow);
        S2.initializeNNIndices(minRows2, maxRows2, numTrainRow);
        S3.initializeNNIndices(minRows3, maxRows3, numTrainRow);
        S4.initializeNNIndices(minRows4, maxRows4, numTrainRow);
        S5.initializeNNIndices(minRows5, maxRows5, numTrainRow);
        S6.initializeNNIndices(minRows6, maxRows6, numTrainRow);
        S7.initializeNNIndices(minRows7, maxRows7, numTrainRow);
        S8.initializeNNIndices(minRows8, maxRows8, numTrainRow);
        S9.initializeNNIndices(minRows9, maxRows9, numTrainRow);
        S10.initializeNNIndices(minRows10, maxRows10, numTrainRow);
        S11.initializeNNIndices(minRows11, maxRows11, numTrainRow);
        S12.initializeNNIndices(minRows12, maxRows12, numTrainRow);
        S13.initializeNNIndices(minRows13, maxRows13, numTrainRow);
        
        
                
        S1.start();        
        S2.start();
        S3.start();
        S4.start();
        S5.start();
        S6.start();
        S7.start();
        S8.start();
        S9.start();
        S10.start();
        S11.start();
        S12.start();
        S13.start();
        
               
        int[][] s1Indices = S1.getNearestNeighborFromThread();
        int[][] s2Indices = S2.getNearestNeighborFromThread();
        int[][] s3Indices = S3.getNearestNeighborFromThread();
        int[][] s4Indices = S4.getNearestNeighborFromThread();
        int[][] s5Indices = S5.getNearestNeighborFromThread();
        int[][] s6Indices = S6.getNearestNeighborFromThread();
        int[][] s7Indices = S7.getNearestNeighborFromThread();
        int[][] s8Indices = S8.getNearestNeighborFromThread();
        int[][] s9Indices = S9.getNearestNeighborFromThread();
        int[][] s10Indices = S10.getNearestNeighborFromThread();
        int[][] s11Indices = S11.getNearestNeighborFromThread();
        int[][] s12Indices = S12.getNearestNeighborFromThread();
        int[][] s13Indices = S13.getNearestNeighborFromThread();

        
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s1Indices, minRows1, maxRows1, k);      
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s2Indices, minRows2, maxRows2, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s3Indices, minRows3, maxRows3, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s4Indices, minRows4, maxRows4, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s5Indices, minRows5, maxRows5, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s6Indices, minRows6, maxRows6, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s7Indices, minRows7, maxRows7, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s8Indices, minRows8, maxRows8, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s9Indices, minRows9, maxRows9, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s10Indices, minRows10, maxRows10, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s11Indices, minRows11, maxRows11, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s12Indices, minRows12, maxRows12, k);
        nnIndices = accumulateNerestNeighborIndices(nnIndices, s13Indices, minRows13, maxRows13, k);

        //the painful parallelization is done __God Bless America__
        //its time for predicting the class labels
//        for(int i=0; i< numTestRow; i++){
//            int pLabel = predictLabel(trainData, nnIndices, numCol, i, k);
//            System.out.println(pLabel);
//        }
        
        System.out.println("\n\nK-nn finished running!!\n");
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("\n\nTotal time taken: " + totalTime + "\n");
        
    }


    
    
    
    
}
