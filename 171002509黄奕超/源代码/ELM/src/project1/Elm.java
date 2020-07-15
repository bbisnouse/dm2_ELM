package project1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;

public class Elm {
	private DenseMatrix train_set;
	private DenseMatrix test_set;
	private int numTrainData;
	private int numTestData;
	private DenseMatrix InputWeight;
	private double TrainingAccuracy;
	private int NumberofHiddenNeurons;
	private int NumberofOutputNeurons;						//also the number of classes
	private int NumberofInputNeurons;						//also the number of attribution
	private String func;
	private int []label;		
	//this class label employ a lazy and easy method,any class must written in 0,1,2...so the preprocessing is required
	
	//the blow variables in both train() and test()
	private DenseMatrix  BiasofHiddenNeurons;
	private DenseMatrix  OutputWeight;
	private DenseMatrix  testP;
	private DenseMatrix  Y;
	private DenseMatrix  T;
	public Elm(int h) {
		// TODO Auto-generated constructor stub
		NumberofHiddenNeurons = h;
		TrainingAccuracy = 0;
		NumberofOutputNeurons=1;
	}
	public double getTrainingAccuracy() {
		return TrainingAccuracy;
	}
	public DenseMatrix loadmatrix(String filename) throws IOException{
		//���������С�ľ���
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)) );
		String firstlineString = reader.readLine();
		String []strings = firstlineString.split("(,*)\\s+");
		int m = Integer.parseInt(strings[0]);
		int n = Integer.parseInt(strings[1]);
		DenseMatrix matrix = new DenseMatrix(m, n);
		firstlineString = reader.readLine();
		int i = 0;
		while (i<m) {
			String []datatrings = firstlineString.split("(,*)\\s+");
			for (int j = 0; j < n; j++) {
				matrix.set(i, j, Double.parseDouble(datatrings[j]));
			}
			i++;
			firstlineString = reader.readLine();
		}
		return matrix;
	}
	
	public void train(String TrainingData_File) throws NotConvergedException{
		try {
			train_set = loadmatrix(TrainingData_File);
		}catch(IOException e) {
			e.printStackTrace();
		}
		train();
	}
	private void train() throws NotConvergedException{
		
		numTrainData = train_set.numRows();
		NumberofInputNeurons = train_set.numColumns() - 1;
		//������������֮���Ȩ�غ�ƫ�ƿ�������趨
		InputWeight = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);
		//�������һ��h��n-1�еľ���
		DenseMatrix transT = new DenseMatrix(numTrainData, 1);
		//tT:m��1��
		DenseMatrix transP = new DenseMatrix(numTrainData, NumberofInputNeurons);
		//tP:m��n-1��
		for (int i = 0; i < numTrainData; i++) {
			//���һ��������tT����
			transT.set(i, 0, train_set.get(i, NumberofInputNeurons));
			for (int j = 0; j < NumberofInputNeurons; j++)
				transP.set(i, j, train_set.get(i, j));
		}
		T = new DenseMatrix(1,numTrainData);
		//T:1��m��
		DenseMatrix P = new DenseMatrix(NumberofInputNeurons,numTrainData);
		//P:n-1��m��
		
		transT.transpose(T);
		transP.transpose(P);
		//ת�þ���
		
		// Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
		
		BiasofHiddenNeurons = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);
		//�������һ��h��1�еľ���
		
		DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
		//tH:h��m��
		
		InputWeight.mult(P, tempH);
		//�����
		
		DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
		//BH:h��m��
		
		for (int j = 0; j < numTrainData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
	
		tempH.add(BiasMatrix);
		DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainData);
		//h��m��
		
		//��������
		for (int j = 0; j < NumberofHiddenNeurons; j++) {
			for (int i = 0; i < numTrainData; i++) {
				double temp = tempH.get(j, i);
				temp = 1.0f/ (1 + Math.exp(-temp));
				H.set(j, i, temp);
			}
		}

		DenseMatrix Ht = new DenseMatrix(numTrainData,NumberofHiddenNeurons);
		//m��h��
		H.transpose(Ht);
		//ת��
		Inverse invers = new Inverse(Ht);
		DenseMatrix pinvHt = invers.getMPInverse();			//NumberofHiddenNeurons*numTrainData
		//DenseMatrix pinvHt = invers.getMPInverse(0.000001); //fast method, PLEASE CITE in your paper properly: 
		
		OutputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
		//OutputWeight=pinv(H') * T';  
		pinvHt.mult(transT, OutputWeight);
		
		
		DenseMatrix Yt = new DenseMatrix(numTrainData,NumberofOutputNeurons);
		Ht.mult(OutputWeight,Yt);
		Y = new DenseMatrix(NumberofOutputNeurons,numTrainData);
		Yt.transpose(Y);
		
		double MSE = 0;
		for (int i = 0; i < numTrainData; i++) {
			MSE += (Yt.get(i, 0) - transT.get(i, 0))*(Yt.get(i, 0) - transT.get(i, 0));
		}
		TrainingAccuracy = Math.sqrt(MSE/numTrainData);
		//ѵ�����ݵ�ģ�ͽ������ʵ������ƽ����ƽ��ֵ��Ȼ����ƽ����֮��Ľ����
	}
	
	public void test(String TestingData_File) {
		try {
			test_set = loadmatrix(TestingData_File);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		numTestData = test_set.numRows();
		//m��
		DenseMatrix ttestP = new DenseMatrix(numTestData, NumberofInputNeurons);
		//ttP:m��n-1��
		for (int i = 0; i < numTestData; i++) {
			for (int j = 0; j < NumberofInputNeurons; j++)
				ttestP.set(i, j, test_set.get(i, j));
		}
		
		testP = new DenseMatrix(NumberofInputNeurons,numTestData);
		ttestP.transpose(testP);
		
		DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		//����㵽������Ȩ��
		InputWeight.mult(testP, tempH_test);
		DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		for (int j = 0; j < numTestData; j++) {
			for (int i = 0; i < NumberofHiddenNeurons; i++) {
				BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
			}
		}
		
		tempH_test.add(BiasMatrix2);
		DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons, numTestData);
		//��������
		for (int j = 0; j < NumberofHiddenNeurons; j++) {
			for (int i = 0; i < numTestData; i++) {
				double temp = tempH_test.get(j, i);
				temp = 1.0f/ (1 + Math.exp(-temp));
				H_test.set(j, i, temp);
			}
		}
		
		DenseMatrix transH_test = new DenseMatrix(numTestData,NumberofHiddenNeurons);
		H_test.transpose(transH_test);
		DenseMatrix Yout = new DenseMatrix(numTestData,NumberofOutputNeurons);
		transH_test.mult(OutputWeight,Yout);
		
		DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
		Yout.transpose(testY);

		
		for (int i = 0; i < numTestData; i++) {
			System.out.print(Yout.get(i, 0));
			if(i!=numTestData-1)System.out.print(", ");
		}
	}

}
